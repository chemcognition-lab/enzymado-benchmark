"""Structural contracts shared by property-preparation sources."""

from pathlib import Path
import typing


class Paths(typing.Protocol):
    """Filesystem locations required by property preparation."""

    data_interim_dir: Path
    data_processed_properties: Path
    data_raw_catpred_dir: Path
    data_raw_swissprot_dir: Path
    results_dir: Path


class Constants(typing.Protocol):
    """Canonical dataset column names required by source cleaners."""

    dataset_id: str
    uniprot_id: str
    protein_sequence: str
    smiles: str
