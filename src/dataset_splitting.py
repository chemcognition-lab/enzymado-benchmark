"""Utilities for generating random or similarity-aware dataset split CSV files."""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import heapq
import json
import math
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import sklearn.model_selection

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
STANDARDIZATION_EXCLUDED_COLUMNS = {
    "active_site",
    "binding_positions",
    "ec_number",
    "id",
    "ligand_id",
    "ligand_name",
    "mutation",
    "n_measurements",
    "ph",
    "resolved_split",
    "smiles",
    "source_split",
    "split",
    "substrate_concentration",
    "temperature",
}
STANDARDIZATION_EXCLUDED_SUFFIXES = (
    "_count",
    "_id",
    "_sem",
    "_std",
)
KNOWN_REGRESSION_TARGET_COLUMNS = {
    "activity",
    "ddg",
    "dtm",
    "expression",
    "kcat",
    "km",
    "melting_point",
    "phopt_median",
    "specific_activity",
    "sum_aromatic_products",
    "target",
    "thermostability",
}


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
            raise ValueError(f"Sequence column '{sequence_col}' is not present in the dataset.")
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
                f"Row {row_index} is missing a valid sequence in column '{sequence_col}'."
            )

        group_key = sequence if group_col is None else str(row.get(group_col, "")).strip()
        if group_col is not None and not group_key:
            raise ValueError(
                f"Row {row_index} is missing a valid grouping value in column '{group_col}'."
            )

        group_to_indices[group_key].append(row_index)
        existing_sequence = group_to_sequence.get(group_key)
        if existing_sequence is None:
            group_to_sequence[group_key] = sequence
        elif existing_sequence != sequence:
            raise ValueError(
                f"Group '{group_key}' contains multiple sequences. "
                "Use a grouping column where each group maps to a single protein sequence."
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
    target_weights = {
        split: total_weight * fractions[split]
        for split in SPLIT_NAMES
    }
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
    for (item_name, item_weight), split in zip(seeded_items, split_priority, strict=True):
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


def parse_finite_float(value: str | None) -> float | None:
    """Parse a finite float from CSV text, returning ``None`` for blanks or invalid values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Return whether all present values in a column are finite numbers."""
    saw_value = False
    for row in rows:
        raw_value = row.get(column)
        if raw_value is None or not str(raw_value).strip():
            continue
        if parse_finite_float(raw_value) is None:
            return False
        saw_value = True
    return saw_value


def is_binary_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Return whether a numeric column contains only binary 0/1 values."""
    values: set[float] = set()
    for row in rows:
        parsed = parse_finite_float(row.get(column))
        if parsed is None:
            continue
        if parsed not in {0.0, 1.0}:
            return False
        values.add(parsed)
    return bool(values)


def should_exclude_standardization_column(column: str) -> bool:
    """Return whether a column is clearly an id, feature, label metadata, or split marker."""
    normalized = column.lower()
    if normalized in STANDARDIZATION_EXCLUDED_COLUMNS:
        return True
    if "sequence" in normalized:
        return True
    return normalized.endswith(STANDARDIZATION_EXCLUDED_SUFFIXES)


def resolve_standardization_columns(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    standardization_cols: tuple[str, ...] | None = None,
) -> tuple[list[str], dict[str, str], str]:
    """Choose scalar columns to standardize, preferring explicit user input."""
    if standardization_cols is not None:
        resolved_cols = list(dict.fromkeys(standardization_cols))
        missing = [column for column in resolved_cols if column not in fieldnames]
        if missing:
            raise ValueError(f"Standardization columns are missing from the dataset: {missing}.")
        non_numeric = [
            column
            for column in resolved_cols
            if not is_numeric_column(rows, column)
        ]
        if non_numeric:
            raise ValueError(
                "Standardization columns must contain only finite numeric values "
                f"or blanks. Non-numeric columns: {non_numeric}."
            )
        return resolved_cols, {}, "explicit"

    numeric_candidates: list[str] = []
    skipped_columns: dict[str, str] = {}
    for column in fieldnames:
        if not is_numeric_column(rows, column):
            continue
        if is_binary_numeric_column(rows, column):
            skipped_columns[column] = "binary_numeric_column"
            continue
        if should_exclude_standardization_column(column):
            skipped_columns[column] = "excluded_id_feature_or_metadata_column"
            continue
        numeric_candidates.append(column)

    known_targets = [
        column
        for column in numeric_candidates
        if column.lower() in KNOWN_REGRESSION_TARGET_COLUMNS
    ]
    if known_targets:
        return known_targets, skipped_columns, "auto_known_target"
    if len(numeric_candidates) == 1:
        return numeric_candidates, skipped_columns, "auto_single_numeric_column"

    for column in numeric_candidates:
        skipped_columns[column] = "ambiguous_numeric_column_use_standardize_col"
    return [], skipped_columns, "auto"


def make_standardized_column_name(
    column: str,
    used_fieldnames: set[str],
) -> str:
    """Return a non-conflicting standardized target column name."""
    base_name = f"{column}_standardized"
    candidate = base_name
    suffix = 2
    while candidate in used_fieldnames:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_fieldnames.add(candidate)
    return candidate


def format_standardized_value(value: float) -> str:
    """Format standardized values compactly while keeping round-trip precision useful."""
    return format(value, ".12g")


def build_standardized_dataset_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    target_columns: list[str],
) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    """Append z-scored target columns fit only on train rows."""
    output_fieldnames = list(fieldnames)
    used_fieldnames = set(output_fieldnames)
    standardized_column_by_target = {
        column: make_standardized_column_name(column, used_fieldnames)
        for column in target_columns
    }
    output_fieldnames.extend(standardized_column_by_target.values())

    standardized_rows = [dict(row) for row in rows]
    columns_metadata: dict[str, object] = {}
    for column in target_columns:
        train_values = [
            parsed
            for row in rows
            if row.get("split") == "train"
            for parsed in [parse_finite_float(row.get(column))]
            if parsed is not None
        ]
        if not train_values:
            raise ValueError(
                f"Cannot standardize column '{column}' because train has no numeric values."
            )

        train_mean = sum(train_values) / len(train_values)
        train_variance = sum((value - train_mean) ** 2 for value in train_values)
        train_variance /= len(train_values)
        train_std = math.sqrt(train_variance)
        scale = 1.0 if train_std == 0.0 else train_std
        standardized_column = standardized_column_by_target[column]

        value_counts_by_split = {split_name: 0 for split_name in SPLIT_NAMES}
        missing_counts_by_split = {split_name: 0 for split_name in SPLIT_NAMES}
        for row in standardized_rows:
            split_name = row.get("split")
            parsed = parse_finite_float(row.get(column))
            if split_name in value_counts_by_split:
                if parsed is None:
                    missing_counts_by_split[split_name] += 1
                else:
                    value_counts_by_split[split_name] += 1
            row[standardized_column] = (
                ""
                if parsed is None
                else format_standardized_value((parsed - train_mean) / scale)
            )

        columns_metadata[column] = {
            "standardized_column": standardized_column,
            "fit_split": "train",
            "fit_statistics": {
                "mean": train_mean,
                "std": train_std,
                "uncertainty": train_std,
                "scale": scale,
                "ddof": 0,
                "n": len(train_values),
                "zero_std_replaced_with_one": train_std == 0.0,
            },
            "applied_to_splits": list(SPLIT_NAMES),
            "numeric_values_by_split": value_counts_by_split,
            "missing_or_invalid_by_split": missing_counts_by_split,
        }

    metadata = {
        "method": "z_score",
        "fit_split": "train",
        "leakage_guard": (
            "Mean and uncertainty/std are fit on train rows only; the same "
            "train-derived values are applied to val and test."
        ),
        "columns": columns_metadata,
    }
    return output_fieldnames, standardized_rows, metadata


def generate_random_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_col: str | None = None,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Generate random but group-aware split assignments."""
    fractions = validate_split_fractions(train_fraction, val_fraction, test_fraction)
    group_to_indices, _ = build_group_index(rows, sequence_col, group_col=group_col)
    group_weights = {
        group_key: len(indices)
        for group_key, indices in group_to_indices.items()
    }
    assignments = assign_items_to_splits(group_weights, fractions, seed=seed)
    output_fieldnames, split_rows = apply_group_assignments(
        rows,
        fieldnames,
        group_to_indices,
        assignments,
    )
    return output_fieldnames, split_rows, summarize_split_rows(split_rows, sequence_col)


def build_rank_strata(
    values: list[float],
    n_bins: int,
) -> list[int]:
    """Assign equal-frequency target-rank strata for regression splitting."""
    if n_bins < 2:
        raise ValueError("Stratification requires at least two bins.")

    ranked_values = sorted((value, index) for index, value in enumerate(values))
    strata = [0] * len(values)
    for rank, (_, index) in enumerate(ranked_values):
        strata[index] = min(rank * n_bins // len(values), n_bins - 1)
    return strata


def build_numeric_strata(
    rows: list[dict[str, str]],
    stratify_col: str,
    n_bins: int,
) -> list[int]:
    """Build equal-frequency strata from a finite numeric column."""
    values: list[float] = []
    for index, row in enumerate(rows):
        value = parse_finite_float(row.get(stratify_col))
        if value is None:
            raise ValueError(
                f"Column '{stratify_col}' is not finite at row {index}."
            )
        values.append(value)
    return build_rank_strata(values, n_bins)


def build_binary_strata(
    rows: list[dict[str, str]],
    stratify_col: str,
) -> list[int]:
    """Build exact 0/1 class strata."""
    strata: list[int] = []
    for index, row in enumerate(rows):
        value = parse_finite_float(row.get(stratify_col))
        if value not in {0.0, 1.0}:
            raise ValueError(
                f"Column '{stratify_col}' is not binary at row {index}."
            )
        strata.append(int(value))
    return strata


def build_residue_strata(
    rows: list[dict[str, str]],
    *,
    sequence_col: str,
    stratify_col: str,
    n_bins: int,
) -> list[int]:
    """Build target-rank strata from residue-positive density per protein."""
    densities: list[float] = []
    for row in rows:
        sequence = clean_sequence(row.get(sequence_col))
        parsed = ast.literal_eval(str(row.get(stratify_col, "")).strip())
        positions = [parsed] if isinstance(parsed, int) else list(parsed)
        densities.append(len({int(position) for position in positions}) / len(sequence))
    return build_rank_strata(densities, n_bins)


def build_ec_label_sets(
    rows: list[dict[str, str]],
    stratify_col: str,
    separator: str,
) -> list[set[str]]:
    """Expand EC annotations into cumulative hierarchical label sets."""
    label_sets: list[set[str]] = []
    for row in rows:
        labels: set[str] = set()
        raw_value = str(row.get(stratify_col, "")).strip()
        for ec_label in raw_value.split(separator):
            ec_label = ec_label.strip()
            if not ec_label:
                continue
            parts = tuple(part.strip() for part in ec_label.split("."))
            if len(parts) != 4:
                raise ValueError(f"Invalid EC label '{ec_label}'. Expected 4 levels.")

            known_parts: list[str] = []
            seen_missing = False
            for depth, part in enumerate(parts, start=1):
                if part == "-":
                    seen_missing = True
                    continue
                if seen_missing:
                    raise ValueError(
                        f"Invalid EC label '{ec_label}'. "
                        "Missing levels must be trailing."
                    )
                known_parts.append(part)
                labels.add(f"level_{depth}:{'.'.join(known_parts)}")
        label_sets.append(labels)
    return label_sets


def assign_stratified_points(
    strata: list[int],
    fractions: dict[str, float],
    seed: int,
) -> dict[int, str]:
    """Assign point indices with a two-stage stratified shuffle split."""
    indices = list(range(len(strata)))
    holdout_fraction = fractions["val"] + fractions["test"]
    if holdout_fraction == 0:
        return {index: "train" for index in indices}

    outer = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout_fraction,
        random_state=seed,
    )
    train_indices, holdout_indices = next(outer.split(indices, strata))

    if fractions["val"] == 0:
        val_indices: list[int] = []
        test_indices = holdout_indices
    elif fractions["test"] == 0:
        val_indices = holdout_indices
        test_indices: list[int] = []
    else:
        holdout_strata = [strata[index] for index in holdout_indices]
        inner = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=fractions["test"] / holdout_fraction,
            random_state=seed + 1,
        )
        val_local, test_local = next(inner.split(holdout_indices, holdout_strata))
        val_indices = holdout_indices[val_local]
        test_indices = holdout_indices[test_local]

    return {
        **{int(index): "train" for index in train_indices},
        **{int(index): "val" for index in val_indices},
        **{int(index): "test" for index in test_indices},
    }


def build_split_capacities(
    n_samples: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    """Return exact split sizes matching two-stage shuffle rounding."""
    n_holdout = math.ceil(n_samples * (fractions["val"] + fractions["test"]))
    if n_holdout == 0:
        return {"train": n_samples, "val": 0, "test": 0}
    if fractions["val"] == 0:
        return {"train": n_samples - n_holdout, "val": 0, "test": n_holdout}
    if fractions["test"] == 0:
        return {"train": n_samples - n_holdout, "val": n_holdout, "test": 0}

    test_ratio = fractions["test"] / (fractions["val"] + fractions["test"])
    n_test = math.ceil(n_holdout * test_ratio)
    return {
        "train": n_samples - n_holdout,
        "val": n_holdout - n_test,
        "test": n_test,
    }


def choose_multilabel_split(
    label: str,
    capacities: dict[str, int],
    split_counts: dict[str, int],
    targets: dict[str, dict[str, float]],
    label_counts: dict[str, dict[str, int]],
    rng: random.Random,
) -> str:
    """Choose the split with the greatest unmet label and sample capacity."""
    candidates = [
        split
        for split in SPLIT_NAMES
        if split_counts[split] < capacities[split]
    ]
    rng.shuffle(candidates)
    return max(
        candidates,
        key=lambda split: (
            targets[label][split] - label_counts[label][split],
            capacities[split] - split_counts[split],
        ),
    )


def assign_multilabel_points(
    label_sets: list[set[str]],
    fractions: dict[str, float],
    seed: int,
) -> dict[int, str]:
    """Iteratively assign rare multilabel samples while preserving marginals."""
    rng = random.Random(seed)
    capacities = build_split_capacities(len(label_sets), fractions)
    split_counts = {split: 0 for split in SPLIT_NAMES}
    remaining_by_label: dict[str, set[int]] = defaultdict(set)
    for index, labels in enumerate(label_sets):
        for label in labels:
            remaining_by_label[label].add(index)

    targets = {
        label: {
            split: len(indices) * fractions[split]
            for split in SPLIT_NAMES
        }
        for label, indices in remaining_by_label.items()
    }
    label_counts = {
        label: {split: 0 for split in SPLIT_NAMES}
        for label in remaining_by_label
    }
    heap = [(len(indices), label) for label, indices in remaining_by_label.items()]
    heapq.heapify(heap)
    unassigned = set(range(len(label_sets)))
    assignments: dict[int, str] = {}

    while heap:
        count, label = heapq.heappop(heap)
        remaining = remaining_by_label[label]
        if not remaining or count != len(remaining):
            continue

        candidates = sorted(remaining)
        rng.shuffle(candidates)
        affected_labels: set[str] = set()
        for index in candidates:
            split = choose_multilabel_split(
                label,
                capacities,
                split_counts,
                targets,
                label_counts,
                rng,
            )
            assignments[index] = split
            unassigned.remove(index)
            split_counts[split] += 1
            for sample_label in label_sets[index]:
                sample_remaining = remaining_by_label[sample_label]
                sample_remaining.discard(index)
                label_counts[sample_label][split] += 1
                affected_labels.add(sample_label)

        for sample_label in sorted(affected_labels - {label}):
            sample_remaining = remaining_by_label[sample_label]
            if sample_remaining:
                heapq.heappush(
                    heap,
                    (len(sample_remaining), sample_label),
                )

    remaining_indices = sorted(unassigned)
    rng.shuffle(remaining_indices)
    for index in remaining_indices:
        split = max(
            SPLIT_NAMES,
            key=lambda name: capacities[name] - split_counts[name],
        )
        assignments[index] = split
        split_counts[split] += 1
    return assignments


def stratification_method(stratify_kind: str) -> str:
    """Return the metadata method name for one stratification kind."""
    match stratify_kind:
        case "regression":
            return "equal_frequency_target_rank"
        case "binary":
            return "exact_binary_class"
        case "multilabel_ec":
            return "iterative_hierarchical_label_marginals"
        case "residue_density":
            return "equal_frequency_positive_residue_density"
        case _:
            raise ValueError(f"Unsupported stratification kind: {stratify_kind}.")


def generate_point_stratified_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    stratify_col: str,
    stratify_kind: str,
    n_bins: int = 10,
    separator: str = ";",
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Generate point-level splits with task-aware label stratification."""
    fractions = validate_split_fractions(train_fraction, val_fraction, test_fraction)
    match stratify_kind:
        case "regression":
            strata = build_numeric_strata(rows, stratify_col, n_bins)
            split_by_index = assign_stratified_points(strata, fractions, seed)
        case "binary":
            strata = build_binary_strata(rows, stratify_col)
            split_by_index = assign_stratified_points(strata, fractions, seed)
        case "residue_density":
            strata = build_residue_strata(
                rows,
                sequence_col=sequence_col,
                stratify_col=stratify_col,
                n_bins=n_bins,
            )
            split_by_index = assign_stratified_points(strata, fractions, seed)
        case "multilabel_ec":
            label_sets = build_ec_label_sets(rows, stratify_col, separator)
            split_by_index = assign_multilabel_points(label_sets, fractions, seed)
        case _:
            raise ValueError(f"Unsupported stratification kind: {stratify_kind}.")

    output_fieldnames = list(fieldnames)
    if "split" not in output_fieldnames:
        output_fieldnames.append("split")
    split_rows = [
        dict(row) | {"split": split_by_index[index]}
        for index, row in enumerate(rows)
    ]
    return output_fieldnames, split_rows, summarize_split_rows(split_rows, sequence_col)


def build_sequence_dedup_index(
    group_to_sequence: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Deduplicate repeated sequences before MMseqs2 clustering."""
    sequence_to_sequence_id: dict[str, str] = {}
    group_to_sequence_id: dict[str, str] = {}

    for index, (group_key, sequence) in enumerate(group_to_sequence.items()):
        sequence_id = sequence_to_sequence_id.get(sequence)
        if sequence_id is None:
            sequence_id = f"seq_{index}"
            sequence_to_sequence_id[sequence] = sequence_id
        group_to_sequence_id[group_key] = sequence_id

    sequence_id_to_sequence = {
        sequence_id: sequence
        for sequence, sequence_id in sequence_to_sequence_id.items()
    }
    return group_to_sequence_id, sequence_id_to_sequence


def write_fasta(sequence_id_to_sequence: dict[str, str], fasta_path: Path) -> None:
    """Write a temporary FASTA file used for MMseqs2 clustering."""
    with fasta_path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in sequence_id_to_sequence.items():
            handle.write(f">{sequence_id}\n{sequence}\n")


def parse_mmseqs_cluster_tsv(
    cluster_tsv_path: Path,
    sequence_ids: set[str],
) -> dict[str, str]:
    """Parse MMseqs2 output into sequence-id to cluster-id assignments."""
    cluster_assignments = {sequence_id: sequence_id for sequence_id in sequence_ids}
    with cluster_tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cluster_id, member_id = parts[0], parts[1]
            cluster_assignments[member_id] = cluster_id
    return cluster_assignments


def build_mmseqs_cluster_command(
    db_path: Path,
    cluster_db_path: Path,
    tmp_work_dir: Path,
    *,
    identity_threshold: float,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> list[str]:
    """Build the ``mmseqs cluster`` command with optional clustering controls."""
    command = [
        "mmseqs",
        "cluster",
        str(db_path),
        str(cluster_db_path),
        str(tmp_work_dir),
        "--min-seq-id",
        str(identity_threshold),
    ]
    if coverage_threshold is not None:
        command.extend(["-c", str(coverage_threshold)])
    if coverage_mode is not None:
        command.extend(["--cov-mode", str(coverage_mode)])
    if cluster_mode is not None:
        command.extend(["--cluster-mode", str(cluster_mode)])
    if similarity_type is not None:
        command.extend(["--similarity-type", str(similarity_type)])
    if sensitivity is not None:
        command.extend(["-s", str(sensitivity)])
    if e_value is not None:
        command.extend(["-e", str(e_value)])
    if alignment_mode is not None:
        command.extend(["--alignment-mode", str(alignment_mode)])
    if seq_id_mode is not None:
        command.extend(["--seq-id-mode", str(seq_id_mode)])
    if max_seqs is not None:
        command.extend(["--max-seqs", str(max_seqs)])
    if single_step_clustering:
        command.extend(["--single-step-clustering", "1"])
    if cluster_reassign:
        command.extend(["--cluster-reassign", "1"])
    return command


def cluster_sequences_with_mmseqs2(
    sequence_id_to_sequence: dict[str, str],
    identity_threshold: float,
    *,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> dict[str, str]:
    """Cluster sequences with MMseqs2 and return a representative per sequence."""
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError("mmseqs is not available in PATH.")

    with tempfile.TemporaryDirectory(prefix="dataset-split-mmseqs-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        fasta_path = tmp_dir / "sequences.fasta"
        db_path = tmp_dir / "db"
        cluster_db_path = tmp_dir / "cluster_db"
        cluster_tsv_path = tmp_dir / "clusters.tsv"
        tmp_work_dir = tmp_dir / "tmp"

        # Keep all MMseqs2 scratch files inside a temporary directory so split
        # generation does not leave intermediate databases in the repo tree.
        write_fasta(sequence_id_to_sequence, fasta_path)
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            build_mmseqs_cluster_command(
                db_path,
                cluster_db_path,
                tmp_work_dir,
                identity_threshold=identity_threshold,
                coverage_threshold=coverage_threshold,
                coverage_mode=coverage_mode,
                cluster_mode=cluster_mode,
                similarity_type=similarity_type,
                sensitivity=sensitivity,
                e_value=e_value,
                alignment_mode=alignment_mode,
                seq_id_mode=seq_id_mode,
                max_seqs=max_seqs,
                single_step_clustering=single_step_clustering,
                cluster_reassign=cluster_reassign,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_db_path),
                str(cluster_tsv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_mmseqs_cluster_tsv(cluster_tsv_path, set(sequence_id_to_sequence))


def build_group_clusters(
    group_to_sequence: dict[str, str],
    *,
    identity_threshold: float,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> dict[str, str]:
    """Map each group key onto the cluster id of its representative sequence."""
    group_to_sequence_id, sequence_id_to_sequence = build_sequence_dedup_index(group_to_sequence)
    sequence_clusters = cluster_sequences_with_mmseqs2(
        sequence_id_to_sequence,
        identity_threshold,
        coverage_threshold=coverage_threshold,
        coverage_mode=coverage_mode,
        cluster_mode=cluster_mode,
        similarity_type=similarity_type,
        sensitivity=sensitivity,
        e_value=e_value,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        max_seqs=max_seqs,
        single_step_clustering=single_step_clustering,
        cluster_reassign=cluster_reassign,
    )

    return {
        group_key: sequence_clusters[sequence_id]
        for group_key, sequence_id in group_to_sequence_id.items()
    }


def generate_similarity_split_from_clusters(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_clusters: dict[str, str],
    group_col: str | None = None,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Assign whole sequence clusters to splits to reduce homology leakage."""
    fractions = validate_split_fractions(train_fraction, val_fraction, test_fraction)
    group_to_indices, _ = build_group_index(rows, sequence_col, group_col=group_col)

    missing_groups = sorted(set(group_to_indices) - set(group_clusters))
    if missing_groups:
        preview = ", ".join(missing_groups[:5])
        raise ValueError(f"Missing cluster assignments for groups: {preview}.")

    cluster_to_weight: dict[str, int] = defaultdict(int)
    cluster_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_key, row_indices in group_to_indices.items():
        cluster_id = group_clusters[group_key]
        cluster_to_groups[cluster_id].append(group_key)
        cluster_to_weight[cluster_id] += len(row_indices)

    cluster_assignments = assign_items_to_splits(
        dict(cluster_to_weight),
        fractions,
        seed=seed,
    )
    group_assignments = {
        group_key: cluster_assignments[cluster_id]
        for cluster_id, group_keys in cluster_to_groups.items()
        for group_key in group_keys
    }
    output_fieldnames, split_rows = apply_group_assignments(
        rows,
        fieldnames,
        group_to_indices,
        group_assignments,
    )
    return output_fieldnames, split_rows, summarize_split_rows(split_rows, sequence_col)


def generate_similarity_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_col: str | None = None,
    identity_threshold: float = 0.3,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Cluster groups first, then derive split rows from those clusters."""
    group_to_indices, group_to_sequence = build_group_index(
        rows,
        sequence_col,
        group_col=group_col,
    )
    if len(group_to_indices) < 3:
        raise ValueError("At least three proteins/groups are required to build train/val/test splits.")

    group_clusters = build_group_clusters(
        group_to_sequence,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        coverage_mode=coverage_mode,
        cluster_mode=cluster_mode,
        similarity_type=similarity_type,
        sensitivity=sensitivity,
        e_value=e_value,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        max_seqs=max_seqs,
        single_step_clustering=single_step_clustering,
        cluster_reassign=cluster_reassign,
    )
    return generate_similarity_split_from_clusters(
        rows,
        fieldnames,
        train_fraction,
        val_fraction,
        test_fraction,
        sequence_col=sequence_col,
        group_clusters=group_clusters,
        group_col=group_col,
        seed=seed,
    )


def write_split_dataset(
    output_csv_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write the full split-labeled dataset (with optional standardized columns)."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_split_metadata(
    metadata_path: Path,
    *,
    mode: str,
    shared_fields: dict[str, object],
    mode_payload: dict[str, object],
) -> Path:
    """Merge mode-specific metadata into the unified per-dataset JSON."""
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}

    for key, value in shared_fields.items():
        existing[key] = value

    modes = existing.setdefault("modes", {})
    if not isinstance(modes, dict):
        modes = {}
        existing["modes"] = modes
    modes[mode] = mode_payload

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def resolve_csv_inputs(input_path: Path) -> list[Path]:
    """Expand a file or directory argument into source CSV datasets."""
    def is_source_dataset(path: Path) -> bool:
        return (
            path.suffix == ".csv"
            and not path.name.endswith("_splits.csv")
            and not path.name.endswith("_standardized.csv")
        )

    if input_path.is_file():
        return [input_path] if is_source_dataset(input_path) else []
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.csv") if is_source_dataset(path))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def format_fraction_label(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> str:
    """Format split fractions into the filename suffix used in the repo."""
    return (
        f"{int(round(train_fraction * 100))}_"
        f"{int(round(val_fraction * 100))}_"
        f"{int(round(test_fraction * 100))}"
    )


def build_output_csv_path(
    input_csv_path: Path,
    *,
    mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    output_root: Path | None = None,
    output_tag: str | None = None,
) -> Path:
    """Build the output path for the wide split-labeled CSV."""
    fraction_label = format_fraction_label(train_fraction, val_fraction, test_fraction)
    resolved_output_tag = sanitize_output_tag(output_tag)
    tag_suffix = f"_{resolved_output_tag}" if resolved_output_tag else ""
    if mode == "random":
        filename = f"{input_csv_path.stem}_random_{fraction_label}{tag_suffix}_standardized.csv"
    elif mode == "similarity":
        filename = (
            f"{input_csv_path.stem}_similarity_mmseqs2_"
            f"{fraction_label}{tag_suffix}_standardized.csv"
        )
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")

    if output_root is None:
        return input_csv_path.parent / filename

    return output_root / filename


def build_metadata_path(
    input_csv_path: Path,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    output_root: Path | None = None,
    output_tag: str | None = None,
) -> Path:
    """Build the unified metadata path shared across split modes for one dataset."""
    fraction_label = format_fraction_label(train_fraction, val_fraction, test_fraction)
    resolved_output_tag = sanitize_output_tag(output_tag)
    tag_suffix = f"_{resolved_output_tag}" if resolved_output_tag else ""
    filename = f"{input_csv_path.stem}_{fraction_label}{tag_suffix}.splits.metadata.json"
    if output_root is None:
        return input_csv_path.parent / filename
    return output_root / filename


def split_dataset_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    *,
    mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    sequence_col: str | None = None,
    group_col: str | None = None,
    random_split_unit: str = "group",
    stratify_col: str | None = None,
    stratify_kind: str | None = None,
    stratify_bins: int = 10,
    stratify_separator: str = ";",
    identity_threshold: float = 0.3,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
    seed: int = 42,
    standardize: bool = True,
    standardization_cols: tuple[str, ...] | None = None,
    metadata_path: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Generate the wide split CSV for one dataset and merge metadata into the shared JSON."""
    fieldnames, rows = read_csv_rows(input_csv_path)
    if not rows:
        raise ValueError(f"{input_csv_path} does not contain any data rows.")

    resolved_sequence_col = detect_sequence_column(fieldnames, sequence_col=sequence_col)
    resolved_group_col = group_col
    resolved_id_col = detect_id_column(fieldnames)

    if mode == "random":
        match random_split_unit:
            case "group":
                output_fieldnames, split_rows, summary = generate_random_split_rows(
                    rows,
                    fieldnames,
                    train_fraction,
                    val_fraction,
                    test_fraction,
                    sequence_col=resolved_sequence_col,
                    group_col=resolved_group_col,
                    seed=seed,
                )
            case "point":
                if stratify_col is None or stratify_kind is None:
                    raise ValueError(
                        "Point-level random splits require a stratification "
                        "kind and column."
                    )
                output_fieldnames, split_rows, summary = (
                    generate_point_stratified_split_rows(
                        rows,
                        fieldnames,
                        train_fraction,
                        val_fraction,
                        test_fraction,
                        sequence_col=resolved_sequence_col,
                        stratify_col=stratify_col,
                        stratify_kind=stratify_kind,
                        n_bins=stratify_bins,
                        separator=stratify_separator,
                        seed=seed,
                    )
                )
            case _:
                raise ValueError(f"Unsupported random split unit: {random_split_unit}.")
    elif mode == "similarity":
        output_fieldnames, split_rows, summary = generate_similarity_split_rows(
            rows,
            fieldnames,
            train_fraction,
            val_fraction,
            test_fraction,
            sequence_col=resolved_sequence_col,
            group_col=resolved_group_col,
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            coverage_mode=coverage_mode,
            cluster_mode=cluster_mode,
            similarity_type=similarity_type,
            sensitivity=sensitivity,
            e_value=e_value,
            alignment_mode=alignment_mode,
            seq_id_mode=seq_id_mode,
            max_seqs=max_seqs,
            single_step_clustering=single_step_clustering,
            cluster_reassign=cluster_reassign,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")

    if standardize:
        (
            resolved_standardization_cols,
            skipped_standardization_cols,
            standardization_source,
        ) = resolve_standardization_columns(
            output_fieldnames,
            split_rows,
            standardization_cols=standardization_cols,
        )
        standardization_metadata: dict[str, object] = {
            "enabled": bool(resolved_standardization_cols),
            "source": standardization_source,
            "target_columns": resolved_standardization_cols,
            "skipped_columns": skipped_standardization_cols,
        }
        if resolved_standardization_cols:
            output_fieldnames, split_rows, standardized_metadata = build_standardized_dataset_rows(
                split_rows,
                output_fieldnames,
                resolved_standardization_cols,
            )
            standardization_metadata.update(standardized_metadata)
        else:
            standardization_metadata["skipped_reason"] = (
                "No scalar numeric target columns were inferred. Use --standardize-col "
                "to opt in explicitly."
            )
    else:
        standardization_metadata = {
            "enabled": False,
            "skipped_reason": "Standardization was disabled for this run.",
        }

    write_split_dataset(output_csv_path, split_rows, output_fieldnames)

    if metadata_path is None:
        metadata_path = build_metadata_path(
            input_csv_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            output_root=output_csv_path.parent,
        )

    shared_fields: dict[str, object] = {
        "input_csv_path": str(input_csv_path),
        "sequence_col": resolved_sequence_col,
        "group_col": resolved_group_col,
        "id_col": resolved_id_col,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    mode_payload: dict[str, object] = {
        "output_csv_path": str(output_csv_path),
        "summary": summary,
        "standardization": standardization_metadata,
    }
    if mode == "random":
        mode_payload["split_unit"] = random_split_unit
        mode_payload["stratification"] = (
            {
                "kind": stratify_kind,
                "column": stratify_col,
                "bins": (
                    stratify_bins
                    if stratify_kind in {"regression", "residue_density"}
                    else None
                ),
                "separator": (
                    stratify_separator
                    if stratify_kind == "multilabel_ec"
                    else None
                ),
                "method": stratification_method(str(stratify_kind)),
            }
            if random_split_unit == "point"
            else None
        )
    if mode == "similarity":
        mode_payload["mmseqs"] = {
            "min_seq_id": identity_threshold,
            "coverage_threshold": coverage_threshold,
            "coverage_mode": coverage_mode,
            "cluster_mode": cluster_mode,
            "similarity_type": similarity_type,
            "sensitivity": sensitivity,
            "e_value": e_value,
            "alignment_mode": alignment_mode,
            "seq_id_mode": seq_id_mode,
            "max_seqs": max_seqs,
            "single_step_clustering": single_step_clustering,
            "cluster_reassign": cluster_reassign,
        }

    write_split_metadata(
        metadata_path,
        mode=mode,
        shared_fields=shared_fields,
        mode_payload=mode_payload,
    )
    return summary


@dataclasses.dataclass(frozen=True)
class RandomSplitSpec:
    """Point-stratification recipe for one random benchmark task."""

    task: str
    source: str
    stratify_kind: str
    stratify_col: str


def build_property_random_specs() -> tuple[RandomSplitSpec, ...]:
    """Return point-stratification recipes for property tasks."""
    return (
        RandomSplitSpec(
            "solubility",
            "properties/solubility/proteinea_solubility.csv",
            "binary",
            "solubility",
        ),
        RandomSplitSpec(
            "ph_opt",
            "properties/ph_opt/brenda_phopt.csv",
            "regression",
            "phopt_median",
        ),
        RandomSplitSpec(
            "ec_number",
            "properties/ec_number/swissprot_ec.csv",
            "multilabel_ec",
            "ec_number",
        ),
        RandomSplitSpec(
            "kcat",
            "properties/kcat/catpred_kcat.csv",
            "regression",
            "kcat",
        ),
        RandomSplitSpec(
            "km",
            "properties/km/catpred_km.csv",
            "regression",
            "km",
        ),
        RandomSplitSpec(
            "binding",
            "properties/binding/biolip_binding.csv",
            "binary",
            "binding",
        ),
        RandomSplitSpec(
            "dtm",
            "properties/dtm/fireprotdb_dtm.csv",
            "regression",
            "dtm",
        ),
        RandomSplitSpec(
            "ddg",
            "properties/ddg_mut/fireprotdb_ddg.csv",
            "regression",
            "ddg",
        ),
        RandomSplitSpec(
            "active_sites",
            "properties/active_sites/swissprot_active_site.csv",
            "residue_density",
            "active_site",
        ),
        RandomSplitSpec(
            "binding_site",
            "properties/binding_site/swissprot_binding_site.csv",
            "residue_density",
            "binding_positions",
        ),
    )


def build_real_case_random_specs() -> tuple[RandomSplitSpec, ...]:
    """Return point-stratification recipes for real-case tasks."""
    return (
        RandomSplitSpec(
            "align_amylase_activity",
            (
                "real_case/align_amylase_activity/"
                "alpha_amylase__specific_activity.csv"
            ),
            "regression",
            "specific_activity",
        ),
        RandomSplitSpec(
            "align_amylase_expression",
            (
                "real_case/align_amylase_expression/"
                "alpha_amylase__expression.csv"
            ),
            "regression",
            "expression",
        ),
        RandomSplitSpec(
            "align_amylase_stability",
            (
                "real_case/align_amylase_stability/"
                "alpha_amylase__thermostability.csv"
            ),
            "regression",
            "thermostability",
        ),
        RandomSplitSpec(
            "align_glucosidase_activity",
            (
                "real_case/align_glucosidase_activity/"
                "b_glucosidase__activity.csv"
            ),
            "regression",
            "activity",
        ),
        RandomSplitSpec(
            "align_glucosidase_expression",
            (
                "real_case/align_glucosidase_expression/"
                "b_glucosidase__expression.csv"
            ),
            "regression",
            "expression",
        ),
        RandomSplitSpec(
            "align_glucosidase_melting",
            (
                "real_case/align_glucosidase_melting/"
                "b_glucosidase__melting_point.csv"
            ),
            "regression",
            "melting_point",
        ),
        RandomSplitSpec(
            "cleeveland_galox",
            "real_case/cleeveland_galox/cleeveland.csv",
            "regression",
            "specific_activity",
        ),
        RandomSplitSpec(
            "erickson_petase",
            "real_case/erickson_petase/erickson.csv",
            "regression",
            "sum_aromatic_products",
        ),
    )


def build_random_split_specs() -> tuple[RandomSplitSpec, ...]:
    """Return recipes for all 18 random benchmark datasets."""
    return build_property_random_specs() + build_real_case_random_specs()


def select_random_split_specs(
    specs: tuple[RandomSplitSpec, ...],
    selected_tasks: list[str] | None,
) -> tuple[RandomSplitSpec, ...]:
    """Filter random split recipes while preserving benchmark order."""
    if selected_tasks is None:
        return specs
    selected = set(selected_tasks)
    return tuple(spec for spec in specs if spec.task in selected)


def generate_random_benchmark_split(
    spec: RandomSplitSpec,
    project_root: Path,
    seed: int,
    stratify_bins: int,
) -> None:
    """Generate one benchmark point-stratified random split."""
    source_path = project_root / "data" / "processed" / spec.source
    output_path = build_output_csv_path(
        source_path,
        mode="random",
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    metadata_path = build_metadata_path(
        source_path,
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    start = time.time()
    summary = split_dataset_csv(
        source_path,
        output_path,
        mode="random",
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
        random_split_unit="point",
        stratify_col=spec.stratify_col,
        stratify_kind=spec.stratify_kind,
        stratify_bins=stratify_bins,
        seed=seed,
        standardization_cols=(
            (spec.stratify_col,)
            if spec.stratify_kind == "regression"
            else None
        ),
        metadata_path=metadata_path,
    )
    print(f"{spec.task}: {output_path} ({time.time() - start:.1f}s)")
    print(json.dumps(summary, indent=2))


def generate_all_random_splits(
    project_root: Path,
    selected_tasks: list[str] | None,
    seed: int,
    stratify_bins: int,
    dry_run: bool,
) -> None:
    """Generate all selected point-stratified random benchmark splits."""
    specs = select_random_split_specs(build_random_split_specs(), selected_tasks)
    if dry_run:
        for spec in specs:
            print(
                f"{spec.task}: {spec.stratify_kind} "
                f"column={spec.stratify_col} source={spec.source}"
            )
        return

    for spec in specs:
        generate_random_benchmark_split(
            spec,
            project_root,
            seed,
            stratify_bins,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for split generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/val/test splits for one CSV or a directory of CSV datasets. "
            "Supports random grouped splits and similarity-aware splits using MMseqs2."
        )
    )
    parser.add_argument(
        "--all-random",
        action="store_true",
        help="Generate point-stratified random splits for all benchmark tasks.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[spec.task for spec in build_random_split_specs()],
        default=None,
        help="With --all-random, limit generation to selected benchmark tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --all-random, print recipes without writing files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV or directory; required unless --all-random is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional directory where generated split files will be written. "
            "If omitted, each split CSV is written next to the source dataset."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "similarity"],
        default=None,
        help="Split strategy; required unless --all-random is used.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction for the train split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction for the validation split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction for the test split.",
    )
    parser.add_argument(
        "--sequence-col",
        type=str,
        default=None,
        help="Sequence column. If omitted, it will be autodetected.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help=(
            "Optional grouping column so all rows for the same protein stay together. "
            "If omitted, grouping is done by the protein sequence."
        ),
    )
    parser.add_argument(
        "--random-split-unit",
        choices=["group", "point"],
        default="group",
        help=(
            "Unit assigned by random mode: whole sequence/group or individual "
            "data point."
        ),
    )
    parser.add_argument(
        "--stratify-col",
        type=str,
        default=None,
        help="Target or residue-position column used for point stratification.",
    )
    parser.add_argument(
        "--stratify-kind",
        choices=["regression", "binary", "multilabel_ec", "residue_density"],
        default=None,
        help="Task-aware strategy used to stratify point-level random splits.",
    )
    parser.add_argument(
        "--stratify-bins",
        type=int,
        default=10,
        help="Number of rank bins for regression and residue-density targets.",
    )
    parser.add_argument(
        "--stratify-separator",
        type=str,
        default=";",
        help="Delimiter for multilabel values such as EC annotations.",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.3,
        help="Minimum sequence identity used to cluster similar proteins.",
    )
    parser.add_argument(
        "--mmseqs-coverage-threshold",
        type=float,
        default=None,
        help="Optional MMseqs2 alignment coverage threshold (-c).",
    )
    parser.add_argument(
        "--mmseqs-coverage-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 coverage mode (--cov-mode).",
    )
    parser.add_argument(
        "--mmseqs-cluster-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 clustering strategy (--cluster-mode).",
    )
    parser.add_argument(
        "--mmseqs-similarity-type",
        type=int,
        default=None,
        help="Optional MMseqs2 similarity metric (--similarity-type).",
    )
    parser.add_argument(
        "--mmseqs-sensitivity",
        type=float,
        default=None,
        help="Optional MMseqs2 sensitivity (-s).",
    )
    parser.add_argument(
        "--mmseqs-evalue",
        type=float,
        default=None,
        help="Optional MMseqs2 E-value threshold (-e).",
    )
    parser.add_argument(
        "--mmseqs-alignment-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 alignment mode (--alignment-mode).",
    )
    parser.add_argument(
        "--mmseqs-seq-id-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 sequence identity mode (--seq-id-mode).",
    )
    parser.add_argument(
        "--mmseqs-max-seqs",
        type=int,
        default=None,
        help="Optional MMseqs2 prefilter cap (--max-seqs).",
    )
    parser.add_argument(
        "--mmseqs-single-step-clustering",
        action="store_true",
        help="Enable MMseqs2 single-step clustering (--single-step-clustering 1).",
    )
    parser.add_argument(
        "--mmseqs-cluster-reassign",
        action="store_true",
        help="Enable MMseqs2 cluster reassignment (--cluster-reassign 1).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional filename tag so exploratory split variants do not overwrite each other.",
    )
    parser.add_argument(
        "--standardize-col",
        action="append",
        default=None,
        help=(
            "Numeric target column to z-score after splitting. Repeat for multiple "
            "columns. If omitted, common scalar regression target columns are "
            "detected automatically."
        ),
    )
    parser.add_argument(
        "--no-standardization",
        action="store_true",
        help="Disable writing the standardized full-data sidecar CSV and scaler metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the split assignment.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and generate split files for one or more datasets."""
    args = build_argument_parser().parse_args()

    if args.all_random:
        generate_all_random_splits(
            project_root=Path(__file__).resolve().parents[1],
            selected_tasks=args.task,
            seed=args.seed,
            stratify_bins=args.stratify_bins,
            dry_run=args.dry_run,
        )
        return

    if args.input is None or args.mode is None:
        raise SystemExit("--input and --mode are required unless --all-random is used.")

    if args.mode == "similarity" and shutil.which("mmseqs") is None:
        raise SystemExit(
            "mmseqs is not available in PATH. Activate the conda environment where "
            "you installed MMseqs2 and retry."
        )

    csv_inputs = resolve_csv_inputs(args.input)
    if not csv_inputs:
        raise SystemExit(f"No CSV files found in {args.input}")

    processed = 0
    skipped: list[tuple[str, str]] = []
    for csv_input in csv_inputs:
        output_csv_path = build_output_csv_path(
            csv_input,
            mode=args.mode,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            output_root=args.output_root,
            output_tag=args.output_tag,
        )
        metadata_path = build_metadata_path(
            csv_input,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            output_root=args.output_root,
            output_tag=args.output_tag,
        )
        try:
            summary = split_dataset_csv(
                csv_input,
                output_csv_path,
                mode=args.mode,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                sequence_col=args.sequence_col,
                group_col=args.group_col,
                random_split_unit=args.random_split_unit,
                stratify_col=args.stratify_col,
                stratify_kind=args.stratify_kind,
                stratify_bins=args.stratify_bins,
                stratify_separator=args.stratify_separator,
                identity_threshold=args.identity_threshold,
                coverage_threshold=args.mmseqs_coverage_threshold,
                coverage_mode=args.mmseqs_coverage_mode,
                cluster_mode=args.mmseqs_cluster_mode,
                similarity_type=args.mmseqs_similarity_type,
                sensitivity=args.mmseqs_sensitivity,
                e_value=args.mmseqs_evalue,
                alignment_mode=args.mmseqs_alignment_mode,
                seq_id_mode=args.mmseqs_seq_id_mode,
                max_seqs=args.mmseqs_max_seqs,
                single_step_clustering=args.mmseqs_single_step_clustering,
                cluster_reassign=args.mmseqs_cluster_reassign,
                seed=args.seed,
                standardize=not args.no_standardization,
                standardization_cols=(
                    tuple(args.standardize_col)
                    if args.standardize_col is not None
                    else None
                ),
                metadata_path=metadata_path,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch directory runs best-effort.
            skipped.append((str(csv_input), str(exc)))
            print(f"SKIPPED {csv_input}: {exc}")
            continue

        processed += 1
        print(f"{csv_input} -> {output_csv_path}")
        print(json.dumps(summary, indent=2))

    print(f"Processed datasets: {processed}")
    if skipped:
        print("Skipped datasets:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")


if __name__ == "__main__":
    main()
