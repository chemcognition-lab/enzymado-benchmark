from __future__ import annotations
"""Central task registry describing dataset paths, labels, and split files."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskConfig:
    """Configuration needed to build datasets and loaders for one task."""
    name: str
    batch_kind: str
    label_kind: str
    dataset_paths: dict[str, Path]
    target_column: str | None
    sequence_column: str = "protein_sequence"
    sample_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")
    split_column: str = "split"
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")
    ligand_column: str | None = None
    wildtype_sequence_column: str | None = None
    residue_positions_column: str | None = None
    ec_separator: str = ";"


def _standardized_path(directory: str, stem: str, mode: str) -> Path:
    """Build the per-variant wide CSV path produced by `dataset_splitting.py`."""
    if mode == "random":
        suffix = "random_80_10_10"
    elif mode == "similarity":
        suffix = "similarity_mmseqs2_80_10_10"
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")
    return PROJECT_ROOT / directory / f"{stem}_{suffix}_standardized.csv"


TASK_REGISTRY: dict[str, TaskConfig] = {
    "solubility": TaskConfig(
        name="solubility",
        batch_kind="protein_global",
        label_kind="binary",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/solubility", "proteinea_solubility", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/solubility", "proteinea_solubility", "similarity"
            ),
        },
        target_column="solubility",
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "ph_opt": TaskConfig(
        name="ph_opt",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/ph_opt", "brenda_phopt", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/ph_opt", "brenda_phopt", "similarity"
            ),
        },
        target_column="phopt_median_standardized",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "ec_number": TaskConfig(
        name="ec_number",
        batch_kind="protein_global",
        label_kind="multilabel_ec",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/ec_number", "swissprot_ec", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/ec_number", "swissprot_ec", "similarity"
            ),
        },
        target_column="ec_number",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "kcat": TaskConfig(
        name="kcat",
        batch_kind="protein_ligand_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/kcat", "catpred_kcat", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/kcat", "catpred_kcat", "similarity"
            ),
        },
        target_column="kcat_standardized",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "km": TaskConfig(
        name="km",
        batch_kind="protein_ligand_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/km", "catpred_km", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/km", "catpred_km", "similarity"
            ),
        },
        target_column="km_standardized",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "binding": TaskConfig(
        name="binding",
        batch_kind="protein_ligand_global",
        label_kind="binary",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/binding", "biolip_binding", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/binding", "biolip_binding", "similarity"
            ),
        },
        target_column="binding",
        ligand_column="smiles",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("protein_sequence", "uniprot_id"),
    ),
    "dtm": TaskConfig(
        name="dtm",
        batch_kind="mutation_delta_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/dtm", "fireprotdb_dtm", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/dtm", "fireprotdb_dtm", "similarity"
            ),
        },
        target_column="dtm_standardized",
        wildtype_sequence_column="wildtype_sequence",
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "ddg": TaskConfig(
        name="ddg",
        batch_kind="mutation_delta_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/ddg_mut", "fireprotdb_ddg", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/ddg_mut", "fireprotdb_ddg", "similarity"
            ),
        },
        target_column="ddg_standardized",
        wildtype_sequence_column="wildtype_sequence",
        sample_id_candidates=("dataset_id", "uniprot_id"),
        split_id_candidates=("dataset_id", "uniprot_id"),
    ),
    "active_sites": TaskConfig(
        name="active_sites",
        batch_kind="protein_residue",
        label_kind="residue_binary_vector",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/active_sites", "swissprot_active_site", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/active_sites", "swissprot_active_site", "similarity"
            ),
        },
        target_column=None,
        residue_positions_column="active_site",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "binding_sites": TaskConfig(
        name="binding_sites",
        batch_kind="protein_residue",
        label_kind="residue_binary_vector",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/properties/binding_site", "swissprot_binding_site", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/properties/binding_site", "swissprot_binding_site", "similarity"
            ),
        },
        target_column=None,
        residue_positions_column="binding_positions",
        sample_id_candidates=("uniprot_id", "protein_sequence"),
        split_id_candidates=("uniprot_id", "protein_sequence"),
    ),
    "align_amylase_activity": TaskConfig(
        name="align_amylase_activity",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_amylase_activity/alpha_amylase__specific_activity.csv",
        },
        target_column="specific_activity",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_column="source_split",
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "align_amylase_expression": TaskConfig(
        name="align_amylase_expression",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_amylase_expression/alpha_amylase__expression.csv",
        },
        target_column="expression",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_column="source_split",
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "align_amylase_stability": TaskConfig(
        name="align_amylase_stability",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_amylase_stability/alpha_amylase__thermostability.csv",
        },
        target_column="thermostability",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_column="source_split",
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "align_glucosidase_activity": TaskConfig(
        name="align_glucosidase_activity",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_glucosidase_activity/b_glucosidase__activity.csv",
        },
        target_column="activity",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_column="source_split",
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "align_glucosidase_expression": TaskConfig(
        name="align_glucosidase_expression",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_glucosidase_expression/"
            "b_glucosidase__expression_random_80_10_10_standardized.csv",
        },
        target_column="expression_standardized",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "align_glucosidase_melting": TaskConfig(
        name="align_glucosidase_melting",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": PROJECT_ROOT
            / "data/processed/real_case/align_glucosidase_melting/"
            "b_glucosidase__melting_point_random_80_10_10_standardized.csv",
        },
        target_column="melting_point_standardized",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "cleeveland_galox": TaskConfig(
        name="cleeveland_galox",
        batch_kind="protein_ligand_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/real_case/cleeveland_galox", "cleeveland", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/real_case/cleeveland_galox", "cleeveland", "similarity"
            ),
        },
        target_column="specific_activity_standardized",
        ligand_column="smiles",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
    "erickson_petase": TaskConfig(
        name="erickson_petase",
        batch_kind="protein_global",
        label_kind="scalar_regression",
        dataset_paths={
            "random": _standardized_path(
                "data/processed/real_case/erickson_petase", "erickson", "random"
            ),
            "similarity": _standardized_path(
                "data/processed/real_case/erickson_petase", "erickson", "similarity"
            ),
        },
        target_column="sum_aromatic_products_standardized",
        sample_id_candidates=("dataset_id", "protein_sequence"),
        split_id_candidates=("dataset_id", "protein_sequence"),
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Return the registered config for a task name."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}") from exc
