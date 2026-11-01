"""Shared filesystem paths and column-name constants used by data scripts."""

from pathlib import Path


class Paths:
    """Absolute project paths shared by data preparation scripts."""

    project_root: Path = Path(__file__).resolve().parents[2]
    benchmark_card: Path = project_root / "benchmark_card.json"

    data_dir: Path = project_root / "data"
    data_raw_dir: Path = data_dir / "raw"
    data_interim_dir: Path = data_dir / "interim"
    data_processed_dir: Path = data_dir / "processed"
    data_processed_real_case: Path = data_processed_dir / "real_case"
    data_external_dir: Path = data_dir / "external"
    data_embeddings_dir: Path = data_dir / "embeddings"
    data_ligand_features_dir: Path = data_dir / "ligand_features"

    data_raw_biolip_dir: Path = data_raw_dir / "biolip"
    data_raw_fireprotdb_dir: Path = data_raw_dir / "fireprotdb"
    data_raw_swissprot_dir: Path = data_raw_dir / "swissprot"
    data_raw_catpred_dir: Path = data_raw_dir / "catpred"
    data_raw_real_case: Path = data_raw_dir / "real_case"

    data_processed_properties: Path = data_processed_dir / "properties"
    data_processed_datasets: Path = data_processed_dir / "datasets"
    protein_pool_csv: Path = data_processed_dir / "protein_pool.csv"
    legacy_seq_index_csv: Path = data_processed_dir / "protein_sequences_unique.csv"

    docs_datasets_dir: Path = project_root / "docs" / "datasets"
    models_dir: Path = project_root / "models"
    notebooks_dir: Path = project_root / "notebooks"
    results_dir: Path = project_root / "results"
    results_figures_dir: Path = results_dir / "figures"
    results_predictions_dir: Path = results_dir / "predictions"
    scripts_dir: Path = project_root / "scripts"
    src_dir: Path = project_root / "src"
    tests_dir: Path = project_root / "tests"


class Constants:
    """Column names and external endpoints shared by data scripts."""

    UNIPROT_BASE_URL: str = "https://www.uniprot.org/uniprot/{}.fasta"
    dataset_id: str = "dataset_id"
    uniprot_id: str = "uniprot_id"
    protein_sequence: str = "protein_sequence"
    ligand_id: str = "ligand_id_raw"
    ligand_name: str = "ligand_name"
    smiles: str = "smiles"
    split: str = "split"


__all__ = ["Paths", "Constants"]
