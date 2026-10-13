#!/usr/bin/env python3
"""Normalize the repository's real-case datasets into benchmark-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enzymado.paths_and_constants import Constants, Paths  # noqa: E402

ERICKSON_VALUE_COLUMN = "sum_aromatic_products"
# The products span 0 to ~2,700 mg/L with a fifth of the wells at 0, so the
# target is log10(x + 1); the smallest nonzero value is about 1.
ERICKSON_TARGET_COLUMN = "log10_sum_aromatic_products_plus_one"
# Reaction conditions are a buffer code followed by the pH, e.g. "NP7.5".
ERICKSON_BUFFERS = ("C", "NP", "H", "B", "G")
ERICKSON_BUFFER_COLUMNS = [f"buffer_{code.lower()}" for code in ERICKSON_BUFFERS]
# values.csv labels its seventh data row "30°C, C6" a second time. It sits
# before the 40°C block, which has no C6 row, and its values fall between 30°C
# and 50°C, so it is the 40°C C6 measurement.
ERICKSON_ROW_FIXES = {6: (("30°C", "C6"), ("40°C", "C6"))}
ERICKSON_INTERIM_COLUMNS = [
    Constants.dataset_id,
    "enzyme_id",
    Constants.protein_sequence,
    "rxn_temperature_raw",
    "temperature",
    "rxn_condition",
    "ph",
    "buffer",
    "measurement_name",
    "measurement_unit",
    ERICKSON_VALUE_COLUMN,
]
ERICKSON_PROCESSED_COLUMNS = [
    Constants.dataset_id,
    Constants.protein_sequence,
    "temperature",
    "ph",
    *ERICKSON_BUFFER_COLUMNS,
    ERICKSON_VALUE_COLUMN,
    ERICKSON_TARGET_COLUMN,
]

CLEVELAND_VALUE_COLUMN = "specific_activity"
# Specific activity spans four orders of magnitude (0.01 to 705), so the target
# is its log10.
CLEVELAND_TARGET_COLUMN = "log10_specific_activity"
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
    CLEVELAND_TARGET_COLUMN,
]
ALIGN_PROCESSED_COLUMNS = [
    Constants.dataset_id,
    Constants.protein_sequence,
    "n_measurements",
]
ALIGN_FAMILIES = (
    "alpha-amylase",
    "b-glucosidase",
)
ALIGN_TASK_OUTPUTS = {
    ("alpha-amylase", "specific_activity"): {
        "directory": "align_amylase_activity",
        "section_key": "align_amylase_activity",
        "description": "ALiGN alpha-amylase real-case specific-activity measurements.",
    },
    ("alpha-amylase", "expression"): {
        "directory": "align_amylase_expression",
        "section_key": "align_amylase_expression",
        "description": "ALiGN alpha-amylase real-case expression measurements.",
    },
    ("alpha-amylase", "thermostability"): {
        "directory": "align_amylase_stability",
        "section_key": "align_amylase_stability",
        "description": "ALiGN alpha-amylase real-case thermostability measurements.",
    },
    ("b-glucosidase", "activity"): {
        "directory": "align_glucosidase_activity",
        "section_key": "align_glucosidase_activity",
        "description": "ALiGN b-glucosidase real-case activity measurements.",
    },
    ("b-glucosidase", "expression"): {
        "directory": "align_glucosidase_expression",
        "section_key": "align_glucosidase_expression",
        "description": "ALiGN b-glucosidase real-case expression measurements.",
    },
    ("b-glucosidase", "melting_point"): {
        "directory": "align_glucosidase_melting",
        "section_key": "align_glucosidase_melting",
        "description": "ALiGN b-glucosidase real-case melting-point measurements.",
    },
}


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


def parse_erickson_condition(raw_value: str) -> tuple[str, float]:
    match = re.fullmatch(r"([A-Z]+)(\d+(?:\.\d+)?)", clean_text(raw_value))
    if match is None or match.group(1) not in ERICKSON_BUFFERS:
        raise ValueError(f"Unrecognized Erickson reaction condition '{raw_value}'.")
    return match.group(1), float(match.group(2))


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


def build_erickson_dataset_id(
    enzyme_id: str, temperature_raw: str, condition: str
) -> str:
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
            if clean_text(row.get("Enzyme ID"))
            and normalize_sequence(row.get("Protein Sequence"))
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
        raise ValueError(
            f"{values_path} must contain at least two header rows and one data row."
        )

    enzyme_ids = [clean_text(cell) for cell in rows[0][2:]]
    measurement_labels = [clean_text(cell) for cell in rows[1][2:]]
    if len(enzyme_ids) != len(measurement_labels):
        raise ValueError("Header rows in values.csv do not align.")

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    for row_index, raw_row in enumerate(rows[2:]):
        if not any(clean_text(cell) for cell in raw_row):
            continue

        temperature_raw = clean_text(raw_row[0])
        condition = clean_text(raw_row[1])
        if not temperature_raw or not condition:
            continue
        if row_index in ERICKSON_ROW_FIXES:
            expected, corrected = ERICKSON_ROW_FIXES[row_index]
            if (temperature_raw, condition) != expected:
                raise ValueError(
                    f"values.csv data row {row_index} is {temperature_raw}, "
                    f"{condition}; expected {expected[0]}, {expected[1]} to fix."
                )
            temperature_raw, condition = corrected

        temperature = parse_temperature(temperature_raw)
        buffer, ph = parse_erickson_condition(condition)

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
                raise ValueError(
                    f"Missing protein sequence for enzyme id '{enzyme_id}'."
                )

            measurement_name, measurement_unit = parse_measurement_label(
                measurement_label
            )
            dataset_id = build_erickson_dataset_id(
                enzyme_id, temperature_raw, condition
            )
            value = str(float(clean_text(raw_value)))

            interim_rows.append(
                {
                    Constants.dataset_id: dataset_id,
                    "enzyme_id": enzyme_id,
                    Constants.protein_sequence: protein_sequence,
                    "rxn_temperature_raw": temperature_raw,
                    "temperature": str(temperature),
                    "rxn_condition": condition,
                    "ph": str(ph),
                    "buffer": buffer,
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
                    "ph": str(ph),
                    **{
                        column: "1.0" if code == buffer else "0.0"
                        for code, column in zip(
                            ERICKSON_BUFFERS, ERICKSON_BUFFER_COLUMNS, strict=True
                        )
                    },
                    ERICKSON_VALUE_COLUMN: value,
                    ERICKSON_TARGET_COLUMN: str(math.log10(float(value) + 1.0)),
                }
            )

    dataset_ids = [row[Constants.dataset_id] for row in processed_rows]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Erickson rows do not have unique dataset ids.")
    return interim_rows, processed_rows


def load_cleveland_sequences(proteins_path: Path) -> dict[str, str]:
    with proteins_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            clean_text(row["protein_name"]): normalize_sequence(row["sequence"])
            for row in reader
            if clean_text(row.get("protein_name"))
            and normalize_sequence(row.get("sequence"))
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
            if clean_text(row.get("substrate_name"))
            and clean_text(row.get("isosmiles"))
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
                raise ValueError(
                    f"Missing protein sequence for protein '{protein_name}'."
                )

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
                    CLEVELAND_TARGET_COLUMN: str(math.log10(float(value))),
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

    measurement_columns = [
        fieldname
        for fieldname in all_fieldnames
        if fieldname not in {"mutant", "mutated_sequence", "dataset"}
    ]

    def measurement_key(row: dict[str, str]) -> tuple:
        """Sequence plus every measurement as a number, so 0.600 equals 0.6."""
        values = tuple(
            float(row[c]) if clean_text(row.get(c)) else None
            for c in measurement_columns
        )
        return normalize_sequence(row.get("mutated_sequence")), values

    # Step 1: the tournament's train file already holds its test rows; drop the copies.
    train_keys = {
        measurement_key(row) for row in raw_rows if row["source_split"] == "train"
    }
    kept_rows = [
        row
        for row in raw_rows
        if row["source_split"] == "train" or measurement_key(row) not in train_keys
    ]
    dropped_test = len(raw_rows) - len(kept_rows)

    # Step 2: one row per distinct measurement, whatever the file, round or formatting.
    deduped_rows: list[dict[str, str]] = []
    seen_keys: set[tuple] = set()
    for row in kept_rows:
        key = measurement_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_rows.append(row)

    print(
        f"{family}: {len(raw_rows)} raw rows, {dropped_test} tournament test rows "
        f"already in train, {len(kept_rows) - len(deduped_rows)} duplicate "
        f"measurements, {len(deduped_rows)} kept"
    )
    return all_fieldnames, deduped_rows, measurement_columns


def build_align_rows(
    family: str,
    align_root: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
    list[tuple[str, str, list[str], list[dict[str, str]]]],
]:
    all_fieldnames, deduped_rows, measurement_columns = load_align_rows(
        family, align_root
    )

    interim_fieldnames = [
        Constants.dataset_id,
        "source_split",
        "mutant",
        Constants.protein_sequence,
    ]
    if "dataset" in all_fieldnames:
        interim_fieldnames.append("source_dataset")
    interim_fieldnames.extend(
        normalize_align_column_name(column) for column in measurement_columns
    )

    interim_rows: list[dict[str, str]] = []
    # Step 3: repeated measurements of a variant are averaged into one row.
    values_by_task: dict[str, dict[str, list[float]]] = {
        measurement_column: {} for measurement_column in measurement_columns
    }

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
            if raw_value:
                values_by_task[measurement_column].setdefault(
                    protein_sequence, []
                ).append(float(raw_value))

    processed_outputs: list[tuple[str, str, list[str], list[dict[str, str]]]] = []
    for measurement_column in measurement_columns:
        task_name = normalize_align_column_name(measurement_column)
        by_sequence = values_by_task[measurement_column]
        processed_rows = [
            {
                Constants.dataset_id: build_align_dataset_id(family, index),
                Constants.protein_sequence: sequence,
                "n_measurements": str(len(values)),
                task_name: str(sum(values) / len(values)),
            }
            for index, (sequence, values) in enumerate(by_sequence.items(), start=1)
        ]
        repeated = sum(1 for values in by_sequence.values() if len(values) > 1)
        print(
            f"  {task_name}: {sum(map(len, by_sequence.values()))} measurements, "
            f"{len(processed_rows)} variants, {repeated} averaged over repeats"
        )
        processed_outputs.append(
            (
                measurement_column,
                task_name,
                list(ALIGN_PROCESSED_COLUMNS) + [task_name],
                processed_rows,
            )
        )

    return interim_fieldnames, interim_rows, processed_outputs


def resolve_align_interim_path(
    out_interim_arg: Path,
    default_erickson_interim: Path,
    paths: Paths,
    family_slug: str,
) -> Path:
    if out_interim_arg == default_erickson_interim:
        return paths.data_interim_dir / "align" / f"{family_slug}_wide.csv"
    if out_interim_arg.suffix.lower() == ".csv":
        return out_interim_arg
    return out_interim_arg / f"{family_slug}_wide.csv"


def resolve_align_processed_dir(
    out_processed_arg: Path,
    default_erickson_processed: Path,
    paths: Paths,
) -> Path:
    if out_processed_arg == default_erickson_processed:
        return paths.data_processed_real_case
    if out_processed_arg.suffix.lower() == ".csv":
        raise ValueError(
            "--out-processed must point to a directory for ALIGN because one CSV is "
            "written per task."
        )
    return out_processed_arg


def resolve_align_task_processed_dir(
    processed_dir: Path,
    family: str,
    task_name: str,
) -> Path:
    task_config = ALIGN_TASK_OUTPUTS.get((family, task_name))
    if task_config is None:
        raise ValueError(
            f"No output-directory mapping is configured for ALIGN family '{family}' "
            f"and task '{task_name}'."
        )
    return processed_dir / str(task_config["directory"])


def path_to_card_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def build_align_section_record(
    family: str,
    task_name: str,
    csv_path: Path,
    project_root: Path,
) -> tuple[str, dict[str, object]]:
    task_config = ALIGN_TASK_OUTPUTS[(family, task_name)]
    section_key = str(task_config["section_key"])
    section_description = str(task_config["description"])
    family_slug = slugify(family)
    dataset_filename = f"{family_slug}__{task_name}.csv"
    if csv_path.name != dataset_filename:
        raise ValueError(
            f"ALIGN output file name mismatch for '{family}' / '{task_name}': expected "
            f"'{dataset_filename}', got '{csv_path.name}'."
        )

    dataset_id = csv_path.stem
    record = {
        "task_type": "regression",
        "description": section_description,
        "datasets": [
            {
                "dataset_id": dataset_id,
                "property_scope": "global",
                "label_units": "dataset-provided value",
                "path": path_to_card_path(csv_path, project_root),
                "file_format": "csv",
                "schema": {
                    "protein_id_col": Constants.dataset_id,
                    "sequence_col": Constants.protein_sequence,
                    "target_col": task_name,
                },
                "splits": {
                    "split_type": "predefined_or_none",
                    "train_path": None,
                    "val_path": None,
                    "test_path": None,
                },
                "filters": {
                    "drop_na_in": [
                        Constants.protein_sequence,
                        task_name,
                    ],
                    "deduplicate_by": [
                        Constants.protein_sequence,
                    ],
                    "aggregate_repeats": "mean over distinct measurements",
                },
                "provenance": {
                    "source_name": "ALiGN",
                    "source_version": None,
                    "citation": None,
                },
            }
        ],
    }
    return section_key, record


def collect_align_section_records(
    processed_dir: Path,
    project_root: Path,
) -> list[tuple[str, dict[str, object]]]:
    section_records: list[tuple[str, dict[str, object]]] = []
    for (family, task_name), task_config in ALIGN_TASK_OUTPUTS.items():
        family_slug = slugify(family)
        csv_path = (
            processed_dir
            / str(task_config["directory"])
            / f"{family_slug}__{task_name}.csv"
        )
        if not csv_path.exists():
            continue
        section_records.append(
            build_align_section_record(
                family,
                task_name,
                csv_path,
                project_root,
            )
        )
    return section_records


def sync_align_benchmark_card(
    benchmark_card_path: Path,
    processed_dir: Path,
    project_root: Path,
) -> list[str]:
    with benchmark_card_path.open("r", encoding="utf-8") as handle:
        benchmark_card = json.load(handle)

    real_case = benchmark_card.setdefault("real_case", {})
    preserved_real_case = {
        key: value
        for key, value in real_case.items()
        if key != "align" and not key.startswith("align_")
    }

    align_records = collect_align_section_records(processed_dir, project_root)
    updated_real_case: dict[str, object] = {}
    for section_key, section_record in align_records:
        updated_real_case[section_key] = section_record
    updated_real_case.update(preserved_real_case)
    benchmark_card["real_case"] = updated_real_case

    with benchmark_card_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(benchmark_card, handle, indent=4)
        handle.write("\n")

    return [section_key for section_key, _ in align_records]


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
            "Clean the Erickson, Cleeveland, or ALIGN datasets into interim and "
            "processed outputs."
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
        choices=(*ALIGN_FAMILIES, "all"),
        default="all",
        help="ALIGN family to clean when --dataset align is used. Default: all",
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
        help=(
            "Output path for the processed dataset, or a directory root for ALIGN "
            "outputs."
        ),
    )
    parser.add_argument(
        "--benchmark-card",
        type=Path,
        default=paths.project_root / "benchmark_card.json",
        help="Benchmark card to synchronize after cleaning ALIGN datasets.",
    )
    parser.add_argument(
        "--skip-benchmark-card",
        action="store_true",
        help="Do not update benchmark_card.json after cleaning ALIGN datasets.",
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
        default_erickson_interim = (
            paths.data_interim_dir / "erickson" / "erickson_long.csv"
        )
        default_erickson_processed = (
            paths.data_processed_properties / "aromatic_products" / "erickson.csv"
        )
        default_cleveland_interim = (
            paths.data_interim_dir / "cleeveland" / "cleeveland_long.csv"
        )
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
        default_erickson_interim = (
            paths.data_interim_dir / "erickson" / "erickson_long.csv"
        )
        default_erickson_processed = (
            paths.data_processed_properties / "aromatic_products" / "erickson.csv"
        )
        align_families = (
            ALIGN_FAMILIES if args.align_family == "all" else (args.align_family,)
        )
        align_out_interim_root = (
            paths.data_interim_dir / "align"
            if args.out_interim == default_erickson_interim
            else args.out_interim
        )
        processed_dir = resolve_align_processed_dir(
            args.out_processed,
            default_erickson_processed,
            paths,
        )

        total_interim_rows = 0
        total_processed_rows = 0
        for family in align_families:
            family_slug = slugify(family)
            out_interim = resolve_align_interim_path(
                args.out_interim,
                default_erickson_interim,
                paths,
                family_slug,
            )
            interim_fieldnames, interim_rows, processed_outputs = build_align_rows(
                family,
                args.align_root,
            )
            write_csv(out_interim, interim_fieldnames, interim_rows)
            print(f"Saved {len(interim_rows)} interim rows to {out_interim}")
            total_interim_rows += len(interim_rows)

            for _, task_name, processed_fieldnames, processed_rows in processed_outputs:
                task_processed_dir = resolve_align_task_processed_dir(
                    processed_dir,
                    family,
                    task_name,
                )
                out_processed = task_processed_dir / f"{family_slug}__{task_name}.csv"
                write_csv(out_processed, processed_fieldnames, processed_rows)
                print(f"Saved {len(processed_rows)} processed rows to {out_processed}")
                total_processed_rows += len(processed_rows)

        interim_rows = [{}] * total_interim_rows
        processed_rows = [{}] * total_processed_rows
        out_interim = align_out_interim_root
        out_processed = processed_dir
        if not args.skip_benchmark_card:
            updated_sections = sync_align_benchmark_card(
                args.benchmark_card,
                processed_dir,
                paths.project_root,
            )
            print(
                f"Updated {args.benchmark_card} with {len(updated_sections)} ALIGN "
                f"real-case section(s)"
            )

    print(f"Saved {len(interim_rows)} interim rows to {out_interim}")
    print(f"Saved {len(processed_rows)} processed rows to {out_processed}")


if __name__ == "__main__":
    main()
