from __future__ import annotations
"""Central task registry describing dataset paths, labels, and split files."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskConfig:
    """Configuration needed to build datasets and loaders for one task."""
    name: str
    batch_kind: str
    label_kind: str
    dataset_path: Path
    target_column: str | None
    sequence_column: str = "protein_sequence"
    sample_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")
    split_column: str | None = None
    split_files: dict[str, Path] = field(default_factory=dict)
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")
    ligand_column: str | None = None
    wildtype_sequence_column: str | None = None
    residue_positions_column: str | None = None
    ec_separator: str = ";"


TASK_REGISTRY: dict[str, TaskConfig] = {
    "solubility": TaskConfig(
        name="solubility",
        batch_kind="protein_global",
        label_kind="binary",
        dataset_path=PROJECT_ROOT / "data/processed/properties/solubility/proteinea_solubility.csv",
        target_column="solubility",
        split_column="split",
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/solubility/"
                "proteinea_solubility_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/solubility/"
                "proteinea_solubility_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "ph_opt": TaskConfig(
        name="ph_opt",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_path=PROJECT_ROOT / "data/processed/properties/ph_opt/brenda_phopt.csv",
        target_column="phopt_median",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/ph_opt/brenda_phopt_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/ph_opt/"
                "brenda_phopt_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "ec_number": TaskConfig(
        name="ec_number",
        batch_kind="protein_global",
        label_kind="multilabel_ec",
        dataset_path=PROJECT_ROOT / "data/processed/properties/ec_number/swissprot_ec.csv",
        target_column="ec_number",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/ec_number/swissprot_ec_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/ec_number/"
                "swissprot_ec_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "kcat": TaskConfig(
        name="kcat",
        batch_kind="protein_ligand_global",
        label_kind="scalar_regression",
        dataset_path=PROJECT_ROOT / "data/processed/properties/kcat/catpred_kcat.csv",
        target_column="kcat",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/kcat/catpred_kcat_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/kcat/"
                "catpred_kcat_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "km": TaskConfig(
        name="km",
        batch_kind="protein_ligand_global",
        label_kind="scalar_regression",
        dataset_path=PROJECT_ROOT / "data/processed/properties/km/catpred_km.csv",
        target_column="km",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/km/catpred_km_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/km/catpred_km_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "binding": TaskConfig(
        name="binding",
        batch_kind="protein_ligand_global",
        label_kind="binary",
        dataset_path=PROJECT_ROOT / "data/processed/properties/binding/biolip_binding.csv",
        target_column="binding",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/binding/biolip_binding_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/binding/"
                "biolip_binding_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("protein_sequence", "uniprot_id"),
    ),
    "dtm": TaskConfig(
        name="dtm",
        batch_kind="mutation_delta_global",
        label_kind="scalar_regression",
        dataset_path=PROJECT_ROOT / "data/processed/properties/dtm/fireprotdb_dtm.csv",
        target_column="dtm",
        wildtype_sequence_column="wildtype_sequence",
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/dtm/fireprotdb_dtm_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/dtm/"
                "fireprotdb_dtm_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "ddg": TaskConfig(
        name="ddg",
        batch_kind="mutation_delta_global",
        label_kind="scalar_regression",
        dataset_path=PROJECT_ROOT / "data/processed/properties/ddg_mut/fireprotdb_ddg.csv",
        target_column="ddg",
        wildtype_sequence_column="wildtype_sequence",
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/ddg_mut/fireprotdb_ddg_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/ddg_mut/"
                "fireprotdb_ddg_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "active_sites": TaskConfig(
        name="active_sites",
        batch_kind="protein_residue",
        label_kind="residue_binary_vector",
        dataset_path=PROJECT_ROOT / "data/processed/properties/active_sites/swissprot_active_site.csv",
        target_column=None,
        residue_positions_column="active_site",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/active_sites/"
                "swissprot_active_site_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/active_sites/"
                "swissprot_active_site_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "binding_sites": TaskConfig(
        name="binding_sites",
        batch_kind="protein_residue",
        label_kind="residue_binary_vector",
        dataset_path=PROJECT_ROOT / "data/processed/properties/binding_site/swissprot_binding_site.csv",
        target_column=None,
        residue_positions_column="binding_positions",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_files={
            "random": PROJECT_ROOT / Path(
                "data/processed/properties/binding_site/"
                "swissprot_binding_site_random_80_10_10_splits.csv"
            ),
            "similarity": PROJECT_ROOT / Path(
                "data/processed/properties/binding_site/"
                "swissprot_binding_site_similarity_mmseqs2_80_10_10_splits.csv"
            ),
        },
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Return the registered config for a task name."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}") from exc
