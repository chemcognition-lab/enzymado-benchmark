#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
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
KNOWN_DATASET_CODES = {
    "properties/active_sites/swissprot_active_site.csv": "as",
    "properties/binding_site/swissprot_binding_site.csv": "bs",
    "properties/binding/biolip_binding.csv": "bi",
    "properties/ec_number/swissprot_ec.csv": "ec",
    "properties/kcat/catpred_kcat.csv": "kc",
    "properties/km/catpred_km.csv": "km",
    "properties/ph_opt/brenda_phopt.csv": "ph",
    "properties/solubility/proteinea_solubility.csv": "so",
    "properties/dtm/fireprotdb_dtm.csv": "dt",
    "properties/ddg_mut/fireprotdb_ddg.csv": "dg",
}


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


def resolve_dataset_code(csv_path: Path) -> str:
    normalized_path = csv_path.as_posix()
    for suffix, code in KNOWN_DATASET_CODES.items():
        if normalized_path.endswith(suffix):
            return code

    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", csv_path.parent.name.lower())
        if token
    ]
    if not tokens:
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", csv_path.stem.lower())
            if token
        ]
    if not tokens:
        return "ds"
    if len(tokens) == 1:
        return tokens[0][:3]
    return "".join(token[0] for token in tokens[:4])[:4]


def should_include_csv(csv_path: Path) -> bool:
    name = csv_path.name
    if name.startswith("protein_sequences_unique"):
        return False
    if "_splits" in csv_path.stem:
        return False
    return True


def load_sequence_records(csv_path: Path) -> list[dict[str, str | bool]]:
    dataset_code = resolve_dataset_code(csv_path)
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
                        "base_sequence_id": build_sequence_id(base_sequence_id, sequence_column),
                        PRIMARY_SEQUENCE_COLUMN: sequence,
                        "has_uniprot_id": bool(uniprot_id),
                        "dataset_code": dataset_code,
                    }
                )

    return records


def assign_disambiguated_sequence_ids(records: list[dict[str, str | bool]]) -> None:
    sequences_by_base_id: dict[str, set[str]] = defaultdict(set)
    sequences_by_base_id_and_dataset: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        base_sequence_id = str(record["base_sequence_id"])
        dataset_code = str(record["dataset_code"])
        sequence = str(record[PRIMARY_SEQUENCE_COLUMN])
        sequences_by_base_id[base_sequence_id].add(sequence)
        sequences_by_base_id_and_dataset[(base_sequence_id, dataset_code)].add(sequence)

    index_by_base_id_dataset_and_sequence: dict[tuple[str, str, str], int] = {}
    for (base_sequence_id, dataset_code), sequences in sequences_by_base_id_and_dataset.items():
        for index, sequence in enumerate(sorted(sequences), start=1):
            index_by_base_id_dataset_and_sequence[(base_sequence_id, dataset_code, sequence)] = index

    for record in records:
        base_sequence_id = str(record["base_sequence_id"])
        dataset_code = str(record["dataset_code"])
        sequence = str(record[PRIMARY_SEQUENCE_COLUMN])

        if len(sequences_by_base_id[base_sequence_id]) == 1:
            record["sequence_id"] = base_sequence_id
            continue

        dataset_sequences = sequences_by_base_id_and_dataset[(base_sequence_id, dataset_code)]
        if len(dataset_sequences) == 1:
            suffix = dataset_code
        else:
            suffix_index = index_by_base_id_dataset_and_sequence[
                (base_sequence_id, dataset_code, sequence)
            ]
            suffix = f"{dataset_code}_{suffix_index}"
        record["sequence_id"] = f"{base_sequence_id}__{suffix}"


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

    assign_disambiguated_sequence_ids(all_records)

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
