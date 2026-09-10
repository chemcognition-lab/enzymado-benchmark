"""Shared row, grouping, and assignment logic for dataset splits."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import random

SPLIT_NAMES = ("train", "val", "test")
SEQUENCE_COLUMN_ALIASES = (
    "protein_sequence",
    "sequence",
    "wildtype_sequence",
    "Sequence",
    "seq",
    "SEQ",
)
ID_COLUMN_ALIASES = (
    "protein_id",
    "uniprot_id",
    "sequence_id",
    "dataset_id",
    "id",
    "ID",
)


def sanitize_output_tag(output_tag: str | None) -> str | None:
    """Return a filesystem-safe filename fragment or ``None`` when no tag is set."""
    if output_tag is None:
        return None

    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in output_tag.strip()
    ).strip("._-")
    if not cleaned:
        raise ValueError("Output tag must contain at least one alphanumeric character.")
    return cleaned


def validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
    """Validate train/val/test fractions and return them as a split map."""
    fractions = {
        "train": float(train_fraction),
        "val": float(val_fraction),
        "test": float(test_fraction),
    }
    total = sum(fractions.values())

    if any(value < 0 for value in fractions.values()):
        raise ValueError("Split fractions must be non-negative.")

    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0. Received train+val+test={total:.6f}."
        )

    if fractions["train"] == 0:
        raise ValueError("The train split fraction must be greater than zero.")

    return fractions


def clean_sequence(sequence: str | None) -> str:
    """Strip whitespace and uppercase a protein sequence."""
    if sequence is None:
        return ""
    return "".join(str(sequence).split()).upper()


def detect_sequence_column(
    fieldnames: list[str],
    sequence_col: str | None = None,
) -> str:
    """Resolve the sequence column explicitly or from common aliases."""
    if sequence_col:
        if sequence_col not in fieldnames:
            raise ValueError(
                f"Sequence column '{sequence_col}' is not present in the dataset."
            )
        return sequence_col

    for alias in SEQUENCE_COLUMN_ALIASES:
        if alias in fieldnames:
            return alias

    raise ValueError(
        "Unable to detect the sequence column automatically. "
        f"Tried aliases: {SEQUENCE_COLUMN_ALIASES}."
    )


def detect_id_column(fieldnames: list[str]) -> str | None:
    """Return the first identifier-like column present in a dataset."""
    for alias in ID_COLUMN_ALIASES:
        if alias in fieldnames:
            return alias
    return None


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV into fieldnames and row dictionaries."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"{csv_path} does not have a CSV header.")

    return fieldnames, rows


def build_group_index(
    rows: list[dict[str, str]],
    sequence_col: str,
    group_col: str | None = None,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Group rows by sequence or by an explicit grouping column."""
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_sequence: dict[str, str] = {}

    for row_index, row in enumerate(rows):
        sequence = clean_sequence(row.get(sequence_col))
        if not sequence:
            raise ValueError(
                f"Row {row_index} is missing a valid sequence in column "
                f"'{sequence_col}'."
            )

        group_key = (
            sequence if group_col is None else str(row.get(group_col, "")).strip()
        )
        if group_col is not None and not group_key:
            raise ValueError(
                f"Row {row_index} is missing a valid grouping value in column "
                f"'{group_col}'."
            )

        group_to_indices[group_key].append(row_index)
        existing_sequence = group_to_sequence.get(group_key)
        if existing_sequence is None:
            group_to_sequence[group_key] = sequence
        elif existing_sequence != sequence:
            raise ValueError(
                f"Group '{group_key}' contains multiple sequences. "
                "Use a grouping column where each group maps to a single protein "
                "sequence."
            )

    return dict(group_to_indices), group_to_sequence


def assign_items_to_splits(
    item_weights: dict[str, int],
    fractions: dict[str, float],
    seed: int = 42,
) -> dict[str, str]:
    """Greedily assign weighted groups to splits while approximating target ratios."""
    items = list(item_weights.items())
    positive_splits = [split for split in SPLIT_NAMES if fractions[split] > 0]
    if len(items) < len(positive_splits):
        raise ValueError(
            "Not enough groups to populate every requested split. "
            f"Groups={len(items)}, positive splits={len(positive_splits)}."
        )

    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda pair: pair[1], reverse=True)

    total_weight = sum(item_weights.values())
    target_weights = {split: total_weight * fractions[split] for split in SPLIT_NAMES}
    current_weights = {split: 0 for split in SPLIT_NAMES}
    assignments: dict[str, str] = {}

    split_priority = sorted(
        positive_splits,
        key=lambda split: target_weights[split],
        reverse=True,
    )
    # Seed each requested split with one group so small splits are not left empty.
    seeded_items = items[: len(split_priority)]
    remaining_items = items[len(split_priority) :]
    for (item_name, item_weight), split in zip(
        seeded_items, split_priority, strict=True
    ):
        assignments[item_name] = split
        current_weights[split] += item_weight

    for item_name, item_weight in remaining_items:
        candidate_splits = list(SPLIT_NAMES)
        rng.shuffle(candidate_splits)
        best_split = candidate_splits[0]
        best_score: tuple[float, float] | None = None

        for split in candidate_splits:
            if fractions[split] == 0:
                continue

            proposed_weights = current_weights.copy()
            proposed_weights[split] += item_weight
            distance = sum(
                abs(proposed_weights[name] - target_weights[name])
                for name in SPLIT_NAMES
            )
            overflow = max(0.0, proposed_weights[split] - target_weights[split])
            score = (distance, overflow)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assignments[item_name] = best_split
        current_weights[best_split] += item_weight

    return assignments


def apply_group_assignments(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    group_to_indices: dict[str, list[int]],
    assignments: dict[str, str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Attach a split label to every row according to its group assignment."""
    output_fieldnames = list(fieldnames)
    if "split" not in output_fieldnames:
        output_fieldnames.append("split")

    split_rows = [dict(row) for row in rows]
    for group_key, row_indices in group_to_indices.items():
        split_name = assignments[group_key]
        for row_index in row_indices:
            split_rows[row_index]["split"] = split_name

    return output_fieldnames, split_rows


def summarize_split_rows(
    rows: list[dict[str, str]],
    sequence_col: str,
) -> dict[str, dict[str, int]]:
    """Summarize row and unique-protein counts for each split."""
    summary: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        split_rows = [row for row in rows if row.get("split") == split_name]
        proteins = {clean_sequence(row.get(sequence_col)) for row in split_rows}
        summary[split_name] = {
            "rows": len(split_rows),
            "proteins": len(proteins),
        }
    return summary
