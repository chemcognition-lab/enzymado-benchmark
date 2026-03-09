#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


ID_COLUMNS = ("uniprot_id", "dataset_id")
SEQUENCE_COLUMN = "protein_sequence"


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def load_sequence_records(csv_path: Path) -> list[dict[str, str | bool]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if SEQUENCE_COLUMN not in fieldnames:
            raise ValueError(f"{csv_path} doesn't contain: '{SEQUENCE_COLUMN}'.")

        if not any(column in fieldnames for column in ID_COLUMNS):
            raise ValueError(
                f"{csv_path} doesn't contain any of the expected id columns: {ID_COLUMNS}."
            )

        records = []
        for row in reader:
            uniprot_id = clean_value(row.get("uniprot_id"))
            dataset_id = clean_value(row.get("dataset_id"))
            sequence = clean_value(row.get(SEQUENCE_COLUMN))
            sequence_id = uniprot_id or dataset_id

            if not sequence_id or not sequence:
                continue

            records.append(
                {
                    "sequence_id": sequence_id,
                    SEQUENCE_COLUMN: sequence,
                    "has_uniprot_id": bool(uniprot_id),
                }
            )

    return records


def build_unique_sequences(input_dir: Path) -> tuple[list[dict[str, str]], int]:
    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}.")

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
        sequence = str(record[SEQUENCE_COLUMN])
        if sequence in unique_by_sequence:
            continue

        unique_by_sequence[sequence] = {
            "sequence_id": str(record["sequence_id"]),
            SEQUENCE_COLUMN: sequence,
        }

    return list(unique_by_sequence.values()), original_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenates all sequences from data/processed/properties and removes "
            "duplicates by sequence."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/properties"),
        help="Root folder with the properties datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/protein_sequences_unique.csv"),
        help="Output CSV file with sequence_id and protein_sequence.",
    )
    args = parser.parse_args()

    unique_sequences, original_count = build_unique_sequences(args.input_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", SEQUENCE_COLUMN])
        writer.writeheader()
        writer.writerows(unique_sequences)

    print(f"Number of original sequences: {original_count}")
    print(f"Number of unique sequences: {len(unique_sequences)}")
    print(f"File saved to: {args.output}")


if __name__ == "__main__":
    main()
