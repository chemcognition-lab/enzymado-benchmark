from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


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


def validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
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
    if sequence is None:
        return ""
    return "".join(str(sequence).split()).upper()


def detect_sequence_column(
    fieldnames: list[str],
    sequence_col: str | None = None,
) -> str:
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
    for alias in ID_COLUMN_ALIASES:
        if alias in fieldnames:
            return alias
    return None


def can_use_id_column(
    rows: list[dict[str, str]],
    id_col: str,
) -> bool:
    assignment_by_id: dict[str, str] = {}
    saw_value = False

    for row in rows:
        protein_id = str(row.get(id_col, "")).strip()
        split_name = str(row.get("split", "")).strip()
        if not protein_id:
            return False

        saw_value = True
        previous_split = assignment_by_id.get(protein_id)
        if previous_split is not None and previous_split != split_name:
            return False
        assignment_by_id[protein_id] = split_name

    return saw_value


def select_output_id_column(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    preferred_id_col: str | None = None,
) -> str:
    candidates: list[str] = []
    if preferred_id_col:
        candidates.append(preferred_id_col)

    for alias in ID_COLUMN_ALIASES:
        if alias in fieldnames and alias not in candidates:
            candidates.append(alias)

    for fieldname in fieldnames:
        if fieldname not in candidates:
            candidates.append(fieldname)

    for candidate in candidates:
        if can_use_id_column(rows, candidate):
            return candidate

    raise ValueError("Unable to find a valid identifier column for the split output.")


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
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
    summary: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        split_rows = [row for row in rows if row.get("split") == split_name]
        proteins = {clean_sequence(row.get(sequence_col)) for row in split_rows}
        summary[split_name] = {
            "rows": len(split_rows),
            "proteins": len(proteins),
        }
    return summary


def build_split_assignment_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    id_col: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    resolved_id_col = select_output_id_column(
        rows,
        fieldnames,
        preferred_id_col=id_col,
    )

    assignment_by_id: dict[str, str] = {}
    for row_index, row in enumerate(rows):
        protein_id = str(row.get(resolved_id_col, "")).strip()
        split_name = str(row.get("split", "")).strip()
        if not protein_id:
            raise ValueError(
                f"Row {row_index} is missing a valid protein identifier in column "
                f"'{resolved_id_col}'."
            )
        if split_name not in SPLIT_NAMES:
            raise ValueError(
                f"Row {row_index} has an invalid split value '{split_name}'."
            )

        previous_split = assignment_by_id.get(protein_id)
        if previous_split is not None and previous_split != split_name:
            raise ValueError(
                f"Protein id '{protein_id}' was assigned to multiple splits: "
                f"'{previous_split}' and '{split_name}'."
            )
        assignment_by_id[protein_id] = split_name

    assignment_rows = [
        {resolved_id_col: protein_id, "split": split_name}
        for protein_id, split_name in assignment_by_id.items()
    ]
    return [resolved_id_col, "split"], assignment_rows


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


def build_sequence_dedup_index(
    group_to_sequence: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
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
    with fasta_path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in sequence_id_to_sequence.items():
            handle.write(f">{sequence_id}\n{sequence}\n")


def parse_mmseqs_cluster_tsv(
    cluster_tsv_path: Path,
    sequence_ids: set[str],
) -> dict[str, str]:
    cluster_assignments = {sequence_id: sequence_id for sequence_id in sequence_ids}
    with cluster_tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cluster_id, member_id = parts[0], parts[1]
            cluster_assignments[member_id] = cluster_id
    return cluster_assignments


def cluster_sequences_with_mmseqs2(
    sequence_id_to_sequence: dict[str, str],
    identity_threshold: float,
) -> dict[str, str]:
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError("mmseqs is not available in PATH.")

    with tempfile.TemporaryDirectory(prefix="dataset-split-mmseqs-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        fasta_path = tmp_dir / "sequences.fasta"
        db_path = tmp_dir / "db"
        cluster_db_path = tmp_dir / "cluster_db"
        cluster_tsv_path = tmp_dir / "clusters.tsv"
        tmp_work_dir = tmp_dir / "tmp"

        write_fasta(sequence_id_to_sequence, fasta_path)
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                str(db_path),
                str(cluster_db_path),
                str(tmp_work_dir),
                "--min-seq-id",
                str(identity_threshold),
            ],
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
) -> dict[str, str]:
    group_to_sequence_id, sequence_id_to_sequence = build_sequence_dedup_index(group_to_sequence)
    sequence_clusters = cluster_sequences_with_mmseqs2(
        sequence_id_to_sequence,
        identity_threshold,
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
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
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


def write_split_outputs(
    output_csv_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    id_col: str | None = None,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    output_fieldnames, assignment_rows = build_split_assignment_rows(
        rows,
        fieldnames,
        id_col=id_col,
    )
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignment_rows)


def resolve_csv_inputs(input_path: Path) -> list[Path]:
    def is_source_dataset(path: Path) -> bool:
        return path.suffix == ".csv" and not path.name.endswith("_splits.csv")

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
) -> Path:
    fraction_label = format_fraction_label(train_fraction, val_fraction, test_fraction)
    if mode == "random":
        filename = f"{input_csv_path.stem}_random_{fraction_label}_splits.csv"
    elif mode == "similarity":
        filename = f"{input_csv_path.stem}_similarity_mmseqs2_{fraction_label}_splits.csv"
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")

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
    identity_threshold: float = 0.3,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    fieldnames, rows = read_csv_rows(input_csv_path)
    if not rows:
        raise ValueError(f"{input_csv_path} does not contain any data rows.")

    resolved_sequence_col = detect_sequence_column(fieldnames, sequence_col=sequence_col)
    resolved_group_col = group_col
    resolved_id_col = detect_id_column(fieldnames)

    if mode == "random":
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
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")

    write_split_outputs(
        output_csv_path,
        split_rows,
        output_fieldnames,
        id_col=resolved_id_col,
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/val/test splits for one CSV or a directory of CSV datasets. "
            "Supports random grouped splits and similarity-aware splits using MMseqs2."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file or a directory containing CSV datasets.",
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
        required=True,
        help="Split strategy to use.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction for the train split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        required=True,
        help="Fraction for the validation split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        required=True,
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
        "--identity-threshold",
        type=float,
        default=0.3,
        help="Minimum sequence identity used to cluster similar proteins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the split assignment.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

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
                identity_threshold=args.identity_threshold,
                seed=args.seed,
            )
        except Exception as exc:
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
