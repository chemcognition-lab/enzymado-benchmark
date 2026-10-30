"""Clean Erickson et al. (2022) PETase thermotolerance dataset.

Dataset & Processing Rationale:
- Target transformation: Sum of aromatic products spans 0 to ~2,700 mg/L with
  roughly one-fifth of reaction wells at 0 and the smallest nonzero value near
  1 mg/L, so the regression target is modeled as log10(sum_aromatic_products + 1).
- Condition encoding: Reaction conditions in values.csv are encoded as a buffer
  code followed by pH (e.g., "NP7.5"). These are parsed into continuous
  temperature (°C), continuous pH, and one-hot buffer columns (C, NP, H, B, G).
- Source table correction: values.csv labels its seventh data row (0-based row
  index 6) "30°C, C6" a second time immediately before the 40°C block, which
  otherwise lacks a C6 row, and its product yields fall between the 30°C and
  50°C series. It is therefore corrected to ("40°C", "C6").
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

ERICKSON_VALUE_COLUMN = "sum_aromatic_products"
ERICKSON_TARGET_COLUMN = "log10_sum_aromatic_products_plus_one"
ERICKSON_BUFFERS = ("C", "NP", "H", "B", "G")
ERICKSON_BUFFER_COLUMNS = [f"buffer_{code.lower()}" for code in ERICKSON_BUFFERS]
ERICKSON_ROW_FIXES = {6: (("30°C", "C6"), ("40°C", "C6"))}
ERICKSON_INTERIM_COLUMNS = [
    _C.dataset_id,
    "enzyme_id",
    _C.protein_sequence,
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
    _C.dataset_id,
    _C.protein_sequence,
    "temperature",
    "ph",
    *ERICKSON_BUFFER_COLUMNS,
    ERICKSON_VALUE_COLUMN,
    ERICKSON_TARGET_COLUMN,
]


def parse_temperature(raw_value: str) -> float:
    """Parse temperature string (e.g. '60°C') into a float."""
    cleaned = utils.clean_text(raw_value).replace("°C", "").replace("C", "")
    return float(cleaned)


def parse_erickson_condition(raw_value: str) -> tuple[str, float]:
    """Parse reaction condition string (e.g. 'NP7.5') into buffer code and pH."""
    match = re.fullmatch(r"([A-Z]+)(\d+(?:\.\d+)?)", utils.clean_text(raw_value))
    if match is None or match.group(1) not in ERICKSON_BUFFERS:
        raise ValueError(f"Unrecognized Erickson reaction condition '{raw_value}'.")
    return match.group(1), float(match.group(2))


def parse_measurement_label(raw_label: str) -> tuple[str, str]:
    """Split a measurement column label into normalized name and unit."""
    cleaned = utils.clean_text(raw_label)
    match = re.match(r"^(.*?)(?:\s*\((.*?)\))?$", cleaned)
    if match is None:
        return cleaned.lower().replace(" ", "_"), ""

    name = utils.clean_text(match.group(1)).lower().replace(" ", "_")
    name = name.replace("/", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    unit = utils.clean_text(match.group(2))
    return name, unit


def build_erickson_dataset_id(
    enzyme_id: str, temperature_raw: str, condition: str
) -> str:
    """Construct a unique dataset_id for an Erickson PETase observation."""
    return (
        f"{utils.slugify(enzyme_id)}_"
        f"{utils.slugify(temperature_raw)}_"
        f"{utils.slugify(condition)}"
    )


def load_sequences(sequences_path: Path) -> dict[str, str]:
    """Load Erickson enzyme sequences keyed by Enzyme ID."""
    with sequences_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            utils.clean_text(row["Enzyme ID"]): utils.normalize_sequence(
                row["Protein Sequence"]
            )
            for row in reader
            if utils.clean_text(row.get("Enzyme ID"))
            and utils.normalize_sequence(row.get("Protein Sequence"))
        }

    if not records:
        raise ValueError(f"No sequence records were loaded from {sequences_path}.")
    return records


def _append_erickson_condition_rows(
    raw_row: list[str],
    enzyme_ids: list[str],
    measurement_labels: list[str],
    sequences_by_id: dict[str, str],
    temperature_raw: str,
    condition: str,
    interim_rows: list[dict[str, str]],
    processed_rows: list[dict[str, str]],
) -> None:
    """Parse one temperature/buffer row across all enzymes and append output rows."""
    temperature = parse_temperature(temperature_raw)
    buffer, ph = parse_erickson_condition(condition)
    buffer_one_hot = {
        col: "1.0" if code == buffer else "0.0"
        for code, col in zip(ERICKSON_BUFFERS, ERICKSON_BUFFER_COLUMNS, strict=True)
    }

    for enzyme_id, measurement_label, raw_value in zip(
        enzyme_ids, measurement_labels, raw_row[2:], strict=True
    ):
        if not enzyme_id:
            continue
        protein_sequence = sequences_by_id.get(enzyme_id)
        if not protein_sequence:
            raise ValueError(f"Missing protein sequence for enzyme id '{enzyme_id}'.")

        measurement_name, measurement_unit = parse_measurement_label(measurement_label)
        dataset_id = build_erickson_dataset_id(enzyme_id, temperature_raw, condition)
        value = str(float(utils.clean_text(raw_value)))

        interim_rows.append(
            {
                _C.dataset_id: dataset_id,
                "enzyme_id": enzyme_id,
                _C.protein_sequence: protein_sequence,
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
                _C.dataset_id: dataset_id,
                _C.protein_sequence: protein_sequence,
                "temperature": str(temperature),
                "ph": str(ph),
                **buffer_one_hot,
                ERICKSON_VALUE_COLUMN: value,
                ERICKSON_TARGET_COLUMN: str(math.log10(float(value) + 1.0)),
            }
        )


def build_erickson_rows(
    sequences_path: Path,
    values_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build interim and processed row dicts from Erickson raw CSV files."""
    sequences_by_id = load_sequences(sequences_path)
    with values_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(
            f"{values_path} must contain at least two header rows and one data row."
        )

    enzyme_ids = [utils.clean_text(cell) for cell in rows[0][2:]]
    measurement_labels = [utils.clean_text(cell) for cell in rows[1][2:]]
    if len(enzyme_ids) != len(measurement_labels):
        raise ValueError("Header rows in values.csv do not align.")

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    for row_index, raw_row in enumerate(rows[2:]):
        if not any(utils.clean_text(cell) for cell in raw_row):
            continue
        temperature_raw = utils.clean_text(raw_row[0])
        condition = utils.clean_text(raw_row[1])
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

        _append_erickson_condition_rows(
            raw_row,
            enzyme_ids,
            measurement_labels,
            sequences_by_id,
            temperature_raw,
            condition,
            interim_rows,
            processed_rows,
        )

    dataset_ids = [row[_C.dataset_id] for row in processed_rows]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Erickson rows do not have unique dataset ids.")
    return interim_rows, processed_rows
