"""Clean Cleveland et al. (2021) GalOx / AA5 CRO dataset.

Dataset & Processing Rationale:
- Target transformation: Galactose oxidase / AA5 copper radical oxidase specific
  activity spans four orders of magnitude (0.01 to 705) with strictly positive
  measurements, so the regression target is modeled on a log10 scale
  (log10_specific_activity).
- Feature retention: Substrate concentration (parsed from the "substrate notes"
  column) and canonical substrate isomeric SMILES are retained alongside the
  full-length variant protein sequence.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

CLEVELAND_VALUE_COLUMN = "specific_activity"
CLEVELAND_TARGET_COLUMN = "log10_specific_activity"
CLEVELAND_INTERIM_COLUMNS = [
    _C.dataset_id,
    "protein_name",
    _C.protein_sequence,
    "substrate_name",
    _C.ligand_name,
    _C.smiles,
    "substrate_notes",
    "substrate_concentration",
    "measurement",
    "units",
    CLEVELAND_VALUE_COLUMN,
    "variance",
]
CLEVELAND_PROCESSED_COLUMNS = [
    _C.dataset_id,
    _C.protein_sequence,
    _C.ligand_name,
    _C.smiles,
    "substrate_concentration",
    CLEVELAND_VALUE_COLUMN,
    CLEVELAND_TARGET_COLUMN,
]


def build_cleveland_dataset_id(protein_name: str, substrate_name: str) -> str:
    """Construct a unique dataset_id for a Cleveland GalOx observation."""
    return f"{utils.slugify(protein_name)}_{utils.slugify(substrate_name)}"


def load_cleveland_sequences(proteins_path: Path) -> dict[str, str]:
    """Load Cleveland protein sequences keyed by protein_name."""
    with proteins_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            utils.clean_text(row["protein_name"]): utils.normalize_sequence(
                row["sequence"]
            )
            for row in reader
            if utils.clean_text(row.get("protein_name"))
            and utils.normalize_sequence(row.get("sequence"))
        }

    if not records:
        raise ValueError(f"No protein records were loaded from {proteins_path}.")
    return records


def load_cleveland_substrates(substrates_path: Path) -> dict[str, str]:
    """Load Cleveland substrate isomeric SMILES keyed by substrate_name."""
    with substrates_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = {
            utils.clean_text(row["substrate_name"]): utils.clean_text(row["isosmiles"])
            for row in reader
            if utils.clean_text(row.get("substrate_name"))
            and utils.clean_text(row.get("isosmiles"))
        }

    if not records:
        raise ValueError(f"No substrate records were loaded from {substrates_path}.")
    return records


def build_cleeveland_rows(
    dataset_path: Path,
    proteins_path: Path,
    substrates_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build interim and processed row dicts from Cleveland raw CSV files."""
    sequences_by_protein = load_cleveland_sequences(proteins_path)
    smiles_by_substrate = load_cleveland_substrates(substrates_path)

    interim_rows: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protein_name = utils.clean_text(row.get("protein_name"))
            substrate_name = utils.clean_text(row.get("substrate_name"))
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

            substrate_notes = utils.clean_text(row.get("substrate notes"))
            substrate_concentration = (
                str(float(substrate_notes)) if substrate_notes else ""
            )
            measurement = utils.clean_text(row.get("measurement"))
            units = utils.clean_text(row.get("units"))
            value = str(float(utils.clean_text(row.get("value"))))
            variance = str(float(utils.clean_text(row.get("variance"))))
            dataset_id = build_cleveland_dataset_id(protein_name, substrate_name)

            interim_rows.append(
                {
                    _C.dataset_id: dataset_id,
                    "protein_name": protein_name,
                    _C.protein_sequence: protein_sequence,
                    "substrate_name": substrate_name,
                    _C.ligand_name: substrate_name,
                    _C.smiles: smiles,
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
                    _C.dataset_id: dataset_id,
                    _C.protein_sequence: protein_sequence,
                    _C.ligand_name: substrate_name,
                    _C.smiles: smiles,
                    "substrate_concentration": substrate_concentration,
                    CLEVELAND_VALUE_COLUMN: value,
                    CLEVELAND_TARGET_COLUMN: str(math.log10(float(value))),
                }
            )

    return interim_rows, processed_rows
