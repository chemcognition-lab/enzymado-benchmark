#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ID_COLUMNS = ("uniprot_id", "dataset_id")
PRIMARY_SEQUENCE_COLUMN = "protein_sequence"
SEQUENCE_COLUMNS = (
    "protein_sequence",
    "wildtype_sequence",
)
DEFAULT_INPUT_DIRS = (
    Path("data/processed/properties"),
    Path("data/processed/real_case"),
)
DEFAULT_OUTPUT = Path("data/processed/protein_sequences_unique_no_fluorescence.csv")


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def normalize_sequence(value: str | None) -> str:
    return "".join(clean_value(value).split()).upper()


def build_sequence_id(base_id: str, sequence_column: str) -> str:
    if sequence_column == PRIMARY_SEQUENCE_COLUMN:
        return base_id
    return f"{base_id}__{sequence_column}"


def should_include_csv(csv_path: Path) -> bool:
    name = csv_path.name
    if name.startswith("protein_sequences_unique"):
        return False
    if "_splits" in csv_path.stem:
        return False
    return True


def load_sequence_records(csv_path: Path) -> list[dict[str, str | bool]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        present_sequence_columns = [
            column
            for column in SEQUENCE_COLUMNS
            if column in fieldnames
        ]
        if not present_sequence_columns:
            raise ValueError(f"{csv_path} doesn't contain any expected sequence column: {SEQUENCE_COLUMNS}.")

        if not any(column in fieldnames for column in ID_COLUMNS):
            raise ValueError(
                f"{csv_path} doesn't contain any of the expected id columns: {ID_COLUMNS}."
            )

        records = []
        for row in reader:
            uniprot_id = clean_value(row.get("uniprot_id"))
            dataset_id = clean_value(row.get("dataset_id"))
            base_sequence_id = uniprot_id or dataset_id

            if not base_sequence_id:
                continue

            for sequence_column in present_sequence_columns:
                sequence = normalize_sequence(row.get(sequence_column))
                if not sequence:
                    continue

                records.append(
                    {
                        "sequence_id": build_sequence_id(base_sequence_id, sequence_column),
                        PRIMARY_SEQUENCE_COLUMN: sequence,
                        "has_uniprot_id": bool(uniprot_id),
                    }
                )

    return records


def build_unique_sequences(input_dirs: list[Path]) -> tuple[list[dict[str, str]], int]:
    csv_paths: list[Path] = []
    for input_dir in input_dirs:
        csv_paths.extend(
            sorted(
                path
                for path in input_dir.rglob("*.csv")
                if should_include_csv(path)
            )
        )
    if not csv_paths:
        joined_dirs = ", ".join(str(path) for path in input_dirs)
        raise FileNotFoundError(f"No CSV files found in: {joined_dirs}.")

    all_records: list[dict[str, str | bool]] = []
    for csv_path in csv_paths:
        all_records.extend(load_sequence_records(csv_path))

    original_count = len(all_records)
    prioritized_records = sorted(
        all_records,
        key=lambda record: (
            not bool(record["has_uniprot_id"]),
            str(record["sequence_id"]),
        ),
    )

    unique_by_sequence: dict[str, dict[str, str]] = {}
    for record in prioritized_records:
        sequence = str(record[PRIMARY_SEQUENCE_COLUMN])
        if sequence in unique_by_sequence:
            continue

        unique_by_sequence[sequence] = {
            "sequence_id": str(record["sequence_id"]),
            PRIMARY_SEQUENCE_COLUMN: sequence,
        }

    return list(unique_by_sequence.values()), original_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate unique protein sequences from processed properties and real-case "
            "datasets, including wildtype sequences from mutation datasets."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root folder to scan for datasets. Repeat the flag to include multiple roots. "
            "Defaults to data/processed/properties and data/processed/real_case."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV file with sequence_id and protein_sequence.",
    )
    args = parser.parse_args()

    input_dirs = args.input_dir or list(DEFAULT_INPUT_DIRS)
    unique_sequences, original_count = build_unique_sequences(input_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", PRIMARY_SEQUENCE_COLUMN])
        writer.writeheader()
        writer.writerows(unique_sequences)

    print("Scanned input directories:")
    for input_dir in input_dirs:
        print(f"  - {input_dir}")
    print(f"Number of original sequences: {original_count}")
    print(f"Number of unique sequences: {len(unique_sequences)}")
    print(f"File saved to: {args.output}")


if __name__ == "__main__":
    main()
