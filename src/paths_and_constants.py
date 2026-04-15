"""Shared filesystem paths and column-name constants used by data scripts."""

from pathlib import Path


class Paths:
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data"
    data_raw_dir = data_dir / "raw"
    data_interim_dir = data_dir / "interim"
    data_processed_dir = data_dir / "processed"
    data_processed_real_case = data_processed_dir / "real_case"
    data_external_dir = data_dir / "external"

    data_raw_biolip_dir = data_raw_dir / "biolip"
    data_raw_fireprotdb_dir = data_raw_dir / "fireprotdb"
    data_raw_swissprot_dir = data_raw_dir / "swissprot"
    data_raw_catpred_dir = data_raw_dir / "catpred"
    
    data_processed_properties = data_processed_dir / "properties"
    data_processed_datasets = data_processed_dir / "datasets"

    models_dir = project_root / "models"
    notebooks_dir = project_root / "notebooks"
    results_dir = project_root / "results"
    results_figures_dir = results_dir / "figures"
    scripts_dir = project_root / "scripts"
    scripts_data_utils_dir = scripts_dir / "data_utils"
    scripts_enzyme_utils_dir = scripts_dir / "enzyme_utils"
    src_dir = project_root / "src"
    tests_dir = project_root / "tests"

    def __init__(self):
        for name in dir(self.__class__):
            if name.startswith("_"):
                continue
            value = getattr(self.__class__, name)
            if isinstance(value, Path):
                setattr(self, name, value)

class Constants:
    # Constants related to data processing
    UNIPROT_BASE_URL = "https://www.uniprot.org/uniprot/{}.fasta"

    # Constants related to modeling
    dataset_id = "dataset_id"
    uniprot_id = "uniprot_id"
    protein_sequence = "protein_sequence"
    ligand_id = "ligand_id_raw"
    ligand_name = "ligand_name"
    smiles = "smiles"

__all__ = ["Paths", "Constants"]
