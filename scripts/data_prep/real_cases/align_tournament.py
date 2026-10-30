"""Clean Align to Innovate Tournament datasets (Proteins 2025).

Dataset & Processing Rationale:
- Families & endpoints: Processes alpha-amylase (specific_activity, expression,
  thermostability) and b-glucosidase (activity, expression, melting_point)
  combining train and test release splits into unified per-property tables.
- Replicate aggregation: Multiple measurements for the same unique protein
  sequence are averaged arithmetically, recording the replicate count in
  n_measurements.
- Target transformation: Endpoints spanning two or more orders of magnitude
  (alpha-amylase specific_activity and thermostability; b-glucosidase activity
  and expression) are modeled on a log10 scale. When zero values are present,
  an additive offset equal to the smallest positive measurement rounded to one
  significant figure is applied prior to log10 transformation. Melting point
  and alpha-amylase expression remain on a linear scale.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants
_P = paths_and_constants.Paths

ALIGN_PROCESSED_COLUMNS = [
    _C.dataset_id,
    _C.protein_sequence,
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

ALIGN_LOG_TARGETS = {
    ("alpha-amylase", "specific_activity"),
    ("alpha-amylase", "thermostability"),
    ("b-glucosidase", "activity"),
    ("b-glucosidase", "expression"),
}


def log10_offset(values: list[float]) -> float:
    """Return 0, or smallest positive value to one sig fig if any is 0."""
    if any(value < 0.0 for value in values):
        raise ValueError("A log10 target cannot contain negative values.")
    if all(value > 0.0 for value in values):
        return 0.0
    smallest = min(value for value in values if value > 0.0)
    return float(f"{smallest:.0e}")


def build_align_dataset_id(family: str, row_index: int) -> str:
    """Construct a unique dataset_id for an ALIGN observation."""
    return f"align_{utils.slugify(family)}_{row_index:06d}"


def normalize_align_column_name(column_name: str) -> str:
    """Normalize an ALIGN measurement column header."""
    return utils.slugify(column_name)


def _read_align_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one ALIGN family CSV and attach its source_split label."""
    source_split = "test" if "test" in csv_path.name.lower() else "train"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {
                **{key: utils.clean_text(value) for key, value in row.items()},
                "source_split": source_split,
            }
            for row in reader
        ]
    return fieldnames, rows


def _measurement_key(
    row: dict[str, str],
    measurement_columns: list[str],
) -> tuple[str, tuple[float | None, ...]]:
    """Return sequence plus numeric measurement tuple for deduplication."""
    values = tuple(
        float(row[c]) if utils.clean_text(row.get(c)) else None
        for c in measurement_columns
    )
    return utils.normalize_sequence(row.get("mutated_sequence")), values


def load_align_rows(
    family: str,
    align_root: Path,
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Load and deduplicate raw rows for an ALIGN enzyme family."""
    family_dir = align_root / family
    if not family_dir.exists():
        raise FileNotFoundError(f"ALIGN family directory not found: {family_dir}")

    raw_rows: list[dict[str, str]] = []
    all_fieldnames: list[str] = []
    for csv_path in sorted(family_dir.glob("*.csv")):
        fieldnames, rows = _read_align_csv(csv_path)
        all_fieldnames.extend(f for f in fieldnames if f not in all_fieldnames)
        raw_rows.extend(rows)

    if not raw_rows:
        raise ValueError(f"No ALIGN rows were loaded from {family_dir}.")

    measurement_columns = [
        fieldname
        for fieldname in all_fieldnames
        if fieldname not in {"mutant", "mutated_sequence", "dataset"}
    ]

    train_keys = {
        _measurement_key(row, measurement_columns)
        for row in raw_rows
        if row["source_split"] == "train"
    }
    kept_rows = [
        row
        for row in raw_rows
        if row["source_split"] == "train"
        or _measurement_key(row, measurement_columns) not in train_keys
    ]
    dropped_test = len(raw_rows) - len(kept_rows)

    deduped_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, tuple[float | None, ...]]] = set()
    for row in kept_rows:
        key = _measurement_key(row, measurement_columns)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_rows.append(row)

    print(
        f"{family}: {len(raw_rows)} raw rows, {dropped_test} tournament test rows "
        f"already in train, {len(kept_rows) - len(deduped_rows)} duplicate "
        f"measurements, {len(deduped_rows)} kept"
    )
    return all_fieldnames, deduped_rows, measurement_columns


def _build_align_interim_row(
    family: str,
    row_index: int,
    row: dict[str, str],
    protein_sequence: str,
    measurement_columns: list[str],
    *,
    has_dataset_col: bool,
) -> dict[str, str]:
    """Construct a single normalized interim row for an ALIGN variant."""
    interim_row = {
        _C.dataset_id: build_align_dataset_id(family, row_index),
        "source_split": utils.clean_text(row.get("source_split")),
        "mutant": utils.clean_text(row.get("mutant")),
        _C.protein_sequence: protein_sequence,
    }
    if has_dataset_col:
        interim_row["source_dataset"] = utils.clean_text(row.get("dataset"))
    for col in measurement_columns:
        interim_row[normalize_align_column_name(col)] = utils.clean_text(row.get(col))
    return interim_row


def _record_align_measurements(
    row: dict[str, str],
    protein_sequence: str,
    measurement_columns: list[str],
    values_by_task: dict[str, dict[str, list[float]]],
) -> None:
    """Append valid numeric measurements from one row into values_by_task."""
    for col in measurement_columns:
        raw_value = utils.clean_text(row.get(col))
        if raw_value:
            values_by_task[col].setdefault(protein_sequence, []).append(
                float(raw_value)
            )


def _build_align_task_output(
    family: str,
    measurement_column: str,
    by_sequence: dict[str, list[float]],
) -> tuple[str, str, list[str], list[dict[str, str]]]:
    """Build processed rows and fieldnames for one ALIGN family measurement column."""
    task_name = normalize_align_column_name(measurement_column)
    processed_rows = [
        {
            _C.dataset_id: build_align_dataset_id(family, index),
            _C.protein_sequence: sequence,
            "n_measurements": str(len(values)),
            task_name: str(sum(values) / len(values)),
        }
        for index, (sequence, values) in enumerate(by_sequence.items(), start=1)
    ]
    fieldnames = [*ALIGN_PROCESSED_COLUMNS, task_name]
    if (family, task_name) in ALIGN_LOG_TARGETS:
        target = f"log10_{task_name}"
        offset = log10_offset([float(row[task_name]) for row in processed_rows])
        for row in processed_rows:
            row[target] = str(math.log10(float(row[task_name]) + offset))
        fieldnames.append(target)
        print(f"  {task_name}: target {target} with offset {offset:g}")
    repeated = sum(1 for values in by_sequence.values() if len(values) > 1)
    print(
        f"  {task_name}: {sum(map(len, by_sequence.values()))} measurements, "
        f"{len(processed_rows)} variants, {repeated} averaged over repeats"
    )
    return measurement_column, task_name, fieldnames, processed_rows


def build_align_rows(
    family: str,
    align_root: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
    list[tuple[str, str, list[str], list[dict[str, str]]]],
]:
    """Build interim wide rows and per-task processed rows for an ALIGN family."""
    all_fieldnames, deduped_rows, measurement_columns = load_align_rows(
        family, align_root
    )
    has_dataset_col = "dataset" in all_fieldnames
    interim_fieldnames = [
        _C.dataset_id,
        "source_split",
        "mutant",
        _C.protein_sequence,
        *(["source_dataset"] if has_dataset_col else []),
        *(normalize_align_column_name(col) for col in measurement_columns),
    ]

    interim_rows: list[dict[str, str]] = []
    values_by_task: dict[str, dict[str, list[float]]] = {
        col: {} for col in measurement_columns
    }
    for row_index, row in enumerate(deduped_rows, start=1):
        protein_sequence = utils.normalize_sequence(row.get("mutated_sequence"))
        if not protein_sequence:
            continue
        interim_rows.append(
            _build_align_interim_row(
                family,
                row_index,
                row,
                protein_sequence,
                measurement_columns,
                has_dataset_col=has_dataset_col,
            )
        )
        _record_align_measurements(
            row, protein_sequence, measurement_columns, values_by_task
        )

    processed_outputs = [
        _build_align_task_output(family, col, values_by_task[col])
        for col in measurement_columns
    ]
    return interim_fieldnames, interim_rows, processed_outputs


def resolve_align_interim_path(
    out_interim_arg: Path,
    default_erickson_interim: Path,
    paths: _P,
    family_slug: str,
) -> Path:
    """Resolve output interim CSV path for an ALIGN family."""
    if out_interim_arg == default_erickson_interim:
        return paths.data_interim_dir / "align" / f"{family_slug}_wide.csv"
    if out_interim_arg.suffix.lower() == ".csv":
        return out_interim_arg
    return out_interim_arg / f"{family_slug}_wide.csv"


def resolve_align_processed_dir(
    out_processed_arg: Path,
    default_erickson_processed: Path,
    paths: _P,
) -> Path:
    """Resolve base processed directory for ALIGN task outputs."""
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
    """Resolve task-specific processed directory for an ALIGN family and endpoint."""
    task_config = ALIGN_TASK_OUTPUTS.get((family, task_name))
    if task_config is None:
        raise ValueError(
            f"No output-directory mapping is configured for ALIGN family '{family}' "
            f"and task '{task_name}'."
        )
    return processed_dir / str(task_config["directory"])


def path_to_card_path(path: Path, project_root: Path) -> str:
    """Convert path to a POSIX path relative to project_root."""
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
    """Build benchmark_card.json section entry for an ALIGN task."""
    task_config = ALIGN_TASK_OUTPUTS[(family, task_name)]
    section_key = str(task_config["section_key"])
    section_description = str(task_config["description"])
    family_slug = utils.slugify(family)
    dataset_filename = f"{family_slug}__{task_name}.csv"
    if csv_path.name != dataset_filename:
        raise ValueError(
            f"ALIGN output file name mismatch for '{family}' / '{task_name}': expected "
            f"'{dataset_filename}', got '{csv_path.name}'."
        )

    dataset_id = csv_path.stem
    record: dict[str, object] = {
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
                    "protein_id_col": _C.dataset_id,
                    "sequence_col": _C.protein_sequence,
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
                        _C.protein_sequence,
                        task_name,
                    ],
                    "deduplicate_by": [
                        _C.protein_sequence,
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
    """Collect benchmark_card.json section entries for all existing ALIGN CSVs."""
    section_records: list[tuple[str, dict[str, object]]] = []
    for (family, task_name), task_config in ALIGN_TASK_OUTPUTS.items():
        family_slug = utils.slugify(family)
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
    """Synchronize ALIGN real-case task entries inside benchmark_card.json."""
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
