#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths_and_constants import Constants, Paths


ERICKSON_VALUE_COLUMN = "sum_aromatic_products"
ERICKSON_INTERIM_COLUMNS = [
    Constants.dataset_id,
    "enzyme_id",
    Constants.protein_sequence,
    "rxn_temperature_raw",
    "temperature",
    "rxn_condition",
    "measurement_name",
    "measurement_unit",
    ERICKSON_VALUE_COLUMN,
]
ERICKSON_PROCESSED_COLUMNS = [
    Constants.dataset_id,
    Constants.protein_sequence,
    "temperature",
    ERICKSON_VALUE_COLUMN,
]

CLEVELAND_VALUE_COLUMN = "specific_activity"
CLEVELAND_INTERIM_COLUMNS = [
    Constants.dataset_id,
    "protein_name",
    Constants.protein_sequence,
    "substrate_name",
    Constants.ligand_name,
    Constants.smiles,
    "substrate_notes",
    "substrate_concentration",
    "measurement",
    "units",
    CLEVELAND_VALUE_COLUMN,
    "variance",
]
CLEVELAND_PROCESSED_COLUMNS = [
    Constants.dataset_id,
    Constants.protein_sequence,
    Constants.ligand_name,
    Constants.smiles,
    "substrate_concentration",
    CLEVELAND_VALUE_COLUMN,
]
ALIGN_PROCESSED_COLUMNS = [
    Constants.dataset_id,
    Constants.protein_sequence,
    "measurement_name",
    "value",
]
ALIGN_FAMILIES = (
    "alkaline phosphatase",
    "alpha-amylase",
    "b-glucosidase",
)


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def normalize_sequence(value: str | None) -> str:
    return "".join(clean_text(value).split()).upper()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean_text(value).lower()).strip("_")


def parse_temperature(raw_value: str) -> float:
    cleaned = clean_text(raw_value).replace("°C", "").replace("C", "")
    return float(cleaned)


def parse_measurement_label(raw_label: str) -> tuple[str, str]:
    cleaned = clean_text(raw_label)
    match = re.match(r"^(.*?)(?:\s*\((.*?)\))?$", cleaned)
    if match is None:
        return cleaned.lower().replace(" ", "_"), ""

    name = clean_text(match.group(1)).lower().replace(" ", "_")
    name = name.replace("/", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    unit = clean_text(match.group(2))
    return name, unit


def build_erickson_dataset_id(enzyme_id: str, temperature_raw: str, condition: str) -> str:
    return f"{slugify(enzyme_id)}_{slugify(temperature_raw)}_{slugify(condition)}"


def build_cleveland_dataset_id(protein_name: str, substrate_name: str) -> str:
    return f"{slugify(protein_name)}_{slugify(substrate_name)}"


def build_align_dataset_id(family: str, row_index: int) -> str:
    return f"align_{slugify(family)}_{row_index:06d}"


def load_sequences(sequences_path: Path) -> dict[str, str]:
    with sequences_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            clean_text(row["Enzyme ID"]): normalize_sequence(row["Protein Sequence"])
            for row in reader
            if clean_text(row.get("Enzyme ID")) and normalize_sequence(row.get("Protein Sequence"))
        }

    if not records:
        raise ValueError(f"No sequence records were loaded from {sequences_path}.")
    return records


def build_erickson_rows(
    sequences_path: Path,
    values_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sequences_by_id = load_sequences(sequences_path)

    with values_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(f"{values_path} must contain at least two header rows and one data row.")

    enzyme_ids = [clean_text(cell) for cell in rows[0][2:]]
    measurement_labels = [clean_text(cell) for cell in rows[1][2:]]
    if len(enzyme_ids) != len(measurement_labels):
        raise ValueError("Header rows in values.csv do not align.")

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    for raw_row in rows[2:]:
        if not any(clean_text(cell) for cell in raw_row):
            continue

        temperature_raw = clean_text(raw_row[0])
        condition = clean_text(raw_row[1])
        if not temperature_raw or not condition:
            continue

        temperature = parse_temperature(temperature_raw)

        for enzyme_id, measurement_label, raw_value in zip(
            enzyme_ids,
            measurement_labels,
            raw_row[2:],
            strict=True,
        ):
            if not enzyme_id:
                continue

            protein_sequence = sequences_by_id.get(enzyme_id)
            if not protein_sequence:
                raise ValueError(f"Missing protein sequence for enzyme id '{enzyme_id}'.")

            measurement_name, measurement_unit = parse_measurement_label(measurement_label)
            dataset_id = build_erickson_dataset_id(enzyme_id, temperature_raw, condition)
            value = str(float(clean_text(raw_value)))

            interim_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    "enzyme_id": enzyme_id,
                    Constants.protein_sequence: protein_sequence,
                    "rxn_temperature_raw": temperature_raw,
                    "temperature": str(temperature),
                    "rxn_condition": condition,
                    "measurement_name": measurement_name,
                    "measurement_unit": measurement_unit,
                    ERICKSON_VALUE_COLUMN: value,
                }
            )
            processed_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    Constants.protein_sequence: protein_sequence,
                    "temperature": str(temperature),
                    ERICKSON_VALUE_COLUMN: value,
                }
            )

    return interim_rows, processed_rows


def load_cleveland_sequences(proteins_path: Path) -> dict[str, str]:
    with proteins_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            clean_text(row["protein_name"]): normalize_sequence(row["sequence"])
            for row in reader
            if clean_text(row.get("protein_name")) and normalize_sequence(row.get("sequence"))
        }

    if not records:
        raise ValueError(f"No protein records were loaded from {proteins_path}.")
    return records


def load_cleveland_substrates(substrates_path: Path) -> dict[str, str]:
    with substrates_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            clean_text(row["substrate_name"]): clean_text(row["isosmiles"])
            for row in reader
            if clean_text(row.get("substrate_name")) and clean_text(row.get("isosmiles"))
        }

    if not records:
        raise ValueError(f"No substrate records were loaded from {substrates_path}.")
    return records


def build_cleeveland_rows(
    dataset_path: Path,
    proteins_path: Path,
    substrates_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sequences_by_protein = load_cleveland_sequences(proteins_path)
    smiles_by_substrate = load_cleveland_substrates(substrates_path)

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_name = clean_text(row.get("protein_name"))
            substrate_name = clean_text(row.get("substrate_name"))
            if not protein_name or not substrate_name:
                continue

            protein_sequence = sequences_by_protein.get(protein_name)
            if not protein_sequence:
                raise ValueError(f"Missing protein sequence for protein '{protein_name}'.")

            smiles = smiles_by_substrate.get(substrate_name)
            if not smiles:
                raise ValueError(f"Missing SMILES for substrate '{substrate_name}'.")

            substrate_notes = clean_text(row.get("substrate notes"))
            substrate_concentration = (
                str(float(substrate_notes)) if substrate_notes else ""
            )
            measurement = clean_text(row.get("measurement"))
            units = clean_text(row.get("units"))
            value = str(float(clean_text(row.get("value"))))
            variance = str(float(clean_text(row.get("variance"))))
            dataset_id = build_cleveland_dataset_id(protein_name, substrate_name)

            interim_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    "protein_name": protein_name,
                    Constants.protein_sequence: protein_sequence,
                    "substrate_name": substrate_name,
                    Constants.ligand_name: substrate_name,
                    Constants.smiles: smiles,
                    "substrate_notes": substrate_notes,
                    "substrate_concentration": substrate_concentration,
                    "measurement": measurement,
                    "units": units,
                    CLEVELAND_VALUE_COLUMN: value,
                    "variance": variance,
                }
            )
            processed_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    Constants.protein_sequence: protein_sequence,
                    Constants.ligand_name: substrate_name,
                    Constants.smiles: smiles,
                    "substrate_concentration": substrate_concentration,
                    CLEVELAND_VALUE_COLUMN: value,
                }
            )

    return interim_rows, processed_rows


def normalize_align_column_name(column_name: str) -> str:
    return slugify(column_name)


def load_align_rows(
    family: str,
    align_root: Path,
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    family_dir = align_root / family
    if not family_dir.exists():
        raise FileNotFoundError(f"ALIGN family directory not found: {family_dir}")

    raw_rows: list[dict[str, str]] = []
    all_fieldnames: list[str] = []
    for csv_path in sorted(family_dir.glob("*.csv")):
        source_split = "test" if "test" in csv_path.name.lower() else "train"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            for fieldname in fieldnames:
                if fieldname not in all_fieldnames:
                    all_fieldnames.append(fieldname)
            for row in reader:
                normalized_row = {key: clean_text(value) for key, value in row.items()}
                normalized_row["source_split"] = source_split
                raw_rows.append(normalized_row)

    if not raw_rows:
        raise ValueError(f"No ALIGN rows were loaded from {family_dir}.")

    deduped_rows: list[dict[str, str]] = []
    seen_rows: set[tuple[tuple[str, str], ...]] = set()
    for row in raw_rows:
        key = tuple(sorted(row.items()))
        if key in seen_rows:
            continue
        seen_rows.add(key)
        deduped_rows.append(row)

    measurement_columns = [
        fieldname
        for fieldname in all_fieldnames
        if fieldname not in {"mutant", "mutated_sequence", "dataset"}
    ]
    return all_fieldnames, deduped_rows, measurement_columns


def build_align_rows(
    family: str,
    align_root: Path,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    all_fieldnames, deduped_rows, measurement_columns = load_align_rows(family, align_root)

    interim_fieldnames = [
        Constants.dataset_id,
        "source_split",
        "mutant",
        Constants.protein_sequence,
    ]
    if "dataset" in all_fieldnames:
        interim_fieldnames.append("source_dataset")
    interim_fieldnames.extend(normalize_align_column_name(column) for column in measurement_columns)

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    for row_index, row in enumerate(deduped_rows, start=1):
        protein_sequence = normalize_sequence(row.get("mutated_sequence"))
        if not protein_sequence:
            continue

        dataset_id = build_align_dataset_id(family, row_index)
        interim_row = {
            Constants.dataset_id: dataset_id,
            "source_split": clean_text(row.get("source_split")),
            "mutant": clean_text(row.get("mutant")),
            Constants.protein_sequence: protein_sequence,
        }
        if "dataset" in all_fieldnames:
            interim_row["source_dataset"] = clean_text(row.get("dataset"))

        for measurement_column in measurement_columns:
            normalized_name = normalize_align_column_name(measurement_column)
            interim_row[normalized_name] = clean_text(row.get(measurement_column))
        interim_rows.append(interim_row)

        for measurement_column in measurement_columns:
            raw_value = clean_text(row.get(measurement_column))
            if not raw_value:
                continue
            processed_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    Constants.protein_sequence: protein_sequence,
                    "measurement_name": normalize_align_column_name(measurement_column),
                    "value": str(float(raw_value)),
                }
            )

    return interim_fieldnames, interim_rows, processed_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    paths = Paths()
    parser = argparse.ArgumentParser(
        description=(
            "Clean the Erickson, Cleeveland, or ALIGN datasets into interim and processed outputs."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["erickson", "cleeveland", "align"],
        default="erickson",
        help="Dataset cleaner to run. Default: erickson",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=paths.data_raw_dir / "erickson" / "sequences.csv",
        help="Path to the Erickson sequences file.",
    )
    parser.add_argument(
        "--values",
        type=Path,
        default=paths.data_raw_dir / "erickson" / "values.csv",
        help="Path to the Erickson values file.",
    )
    parser.add_argument(
        "--dataset-table",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "og_dataset_clev.csv",
        help="Path to the Cleeveland activity table.",
    )
    parser.add_argument(
        "--align-family",
        choices=ALIGN_FAMILIES,
        default="alkaline phosphatase",
        help="ALIGN family to clean when --dataset align is used.",
    )
    parser.add_argument(
        "--align-root",
        type=Path,
        default=paths.data_raw_dir / "align",
        help="Root directory containing ALIGN datasets.",
    )
    parser.add_argument(
        "--proteins",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "proteins.csv",
        help="Path to the Cleeveland proteins file.",
    )
    parser.add_argument(
        "--substrates",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "substrates.csv",
        help="Path to the Cleeveland substrates file.",
    )
    parser.add_argument(
        "--out-interim",
        type=Path,
        default=paths.data_interim_dir / "erickson" / "erickson_long.csv",
        help="Output path for the interim dataset.",
    )
    parser.add_argument(
        "--out-processed",
        type=Path,
        default=paths.data_processed_properties / "aromatic_products" / "erickson.csv",
        help="Output path for the processed dataset.",
    )
    return parser


def main() -> None:
    paths = Paths()
    args = build_argument_parser().parse_args()

    if args.dataset == "erickson":
        out_interim = args.out_interim
        out_processed = args.out_processed
        interim_rows, processed_rows = build_erickson_rows(args.sequences, args.values)
        write_csv(out_interim, ERICKSON_INTERIM_COLUMNS, interim_rows)
        write_csv(out_processed, ERICKSON_PROCESSED_COLUMNS, processed_rows)
    elif args.dataset == "cleeveland":
        default_erickson_interim = paths.data_interim_dir / "erickson" / "erickson_long.csv"
        default_erickson_processed = (
            paths.data_processed_properties / "aromatic_products" / "erickson.csv"
        )
        default_cleveland_interim = paths.data_interim_dir / "cleeveland" / "cleeveland_long.csv"
        default_cleveland_processed = (
            paths.data_processed_properties / "specific_activity" / "cleeveland.csv"
        )

        out_interim = (
            default_cleveland_interim
            if args.out_interim == default_erickson_interim
            else args.out_interim
        )
        out_processed = (
            default_cleveland_processed
            if args.out_processed == default_erickson_processed
            else args.out_processed
        )
        interim_rows, processed_rows = build_cleeveland_rows(
            args.dataset_table,
            args.proteins,
            args.substrates,
        )
        write_csv(out_interim, CLEVELAND_INTERIM_COLUMNS, interim_rows)
        write_csv(out_processed, CLEVELAND_PROCESSED_COLUMNS, processed_rows)
    else:
        default_erickson_interim = paths.data_interim_dir / "erickson" / "erickson_long.csv"
        default_erickson_processed = (
            paths.data_processed_properties / "aromatic_products" / "erickson.csv"
        )
        family_slug = slugify(args.align_family)
        default_align_interim = paths.data_interim_dir / "align" / f"{family_slug}_wide.csv"
        default_align_processed = paths.data_processed_properties / "align" / f"{family_slug}.csv"

        out_interim = (
            default_align_interim
            if args.out_interim == default_erickson_interim
            else args.out_interim
        )
        out_processed = (
            default_align_processed
            if args.out_processed == default_erickson_processed
            else args.out_processed
        )
        interim_fieldnames, interim_rows, processed_rows = build_align_rows(
            args.align_family,
            args.align_root,
        )
        write_csv(out_interim, interim_fieldnames, interim_rows)
        write_csv(out_processed, ALIGN_PROCESSED_COLUMNS, processed_rows)

    print(f"Saved {len(interim_rows)} interim rows to {out_interim}")
    print(f"Saved {len(processed_rows)} processed rows to {out_processed}")


if __name__ == "__main__":
    main()
