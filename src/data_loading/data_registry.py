from __future__ import annotations
"""Central task registry describing dataset paths, labels, and split files."""

import enum
import sys
from dataclasses import dataclass
from pathlib import Path

# enum.StrEnum was added in Python 3.11; provide a shim for 3.10
if sys.version_info < (3, 11) and not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LabelType(enum.StrEnum):
    BINARY = "binary"
    SCALAR_REGRESSION = "scalar_regression"
    MULTILABEL_EC = "multilabel_ec"
    RESIDUE_BINARY_VECTOR = "residue_binary_vector"


class TaskType(enum.StrEnum):
    PROTEIN_GLOBAL = "protein_global"
    PROTEIN_LIGAND_GLOBAL = "protein_ligand_global"
    MUTATION_DELTA_GLOBAL = "mutation_delta_global"
    PROTEIN_RESIDUE = "protein_residue"


class SplitType(enum.StrEnum):
    RANDOM = "random"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class TaskConfig:
    """Configuration needed to build datasets and loaders for one task."""
    name: str
    batch_kind: TaskType
    label_kind: LabelType
    dataset_paths: dict[str, Path]
    target_column: str | None
    task_scope: str = "properties"
    sequence_column: str = "protein_sequence"
    split_column: str = "split"
    ligand_column: str | None = None
    index_columns: tuple[str, ...] = ("dataset_id", "uniprot_id")
    wildtype_sequence_column: str | None = None
    residue_positions_column: str | None = None
    ec_separator: str = ";"


_DATA = PROJECT_ROOT / "data/processed"


TASK_REGISTRY: dict[str, TaskConfig] = {
    "solubility": TaskConfig(
        name="solubility",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.BINARY,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/solubility/proteinea_solubility_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/solubility/proteinea_solubility_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="solubility",
        index_columns=("dataset_id", "uniprot_id"),
    ),
    "ph_opt": TaskConfig(
        name="ph_opt",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/ph_opt/brenda_phopt_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/ph_opt/brenda_phopt_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="phopt_median_standardized",
        index_columns=("uniprot_id",),
    ),
    "ec_number": TaskConfig(
        name="ec_number",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.MULTILABEL_EC,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/ec_number/swissprot_ec_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/ec_number/swissprot_ec_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="ec_number",
        index_columns=("uniprot_id",),
    ),
    "kcat": TaskConfig(
        name="kcat",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/kcat/catpred_kcat_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/kcat/catpred_kcat_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="kcat_standardized",
        ligand_column="smiles",
        index_columns=("uniprot_id",),
    ),
    "km": TaskConfig(
        name="km",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/km/catpred_km_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/km/catpred_km_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="km_standardized",
        ligand_column="smiles",
        index_columns=("uniprot_id",),
    ),
    "binding": TaskConfig(
        name="binding",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.BINARY,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/binding/biolip_binding_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/binding/biolip_binding_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="binding",
        ligand_column="smiles",
        index_columns=("uniprot_id",),
    ),
    "dtm": TaskConfig(
        name="dtm",
        batch_kind=TaskType.MUTATION_DELTA_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/dtm/fireprotdb_dtm_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/dtm/fireprotdb_dtm_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="dtm_standardized",
        wildtype_sequence_column="wildtype_sequence",
        index_columns=("dataset_id", "uniprot_id"),
    ),
    "ddg": TaskConfig(
        name="ddg",
        batch_kind=TaskType.MUTATION_DELTA_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/ddg_mut/fireprotdb_ddg_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/ddg_mut/fireprotdb_ddg_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="ddg_standardized",
        wildtype_sequence_column="wildtype_sequence",
        index_columns=("dataset_id", "uniprot_id"),
    ),
    "active_sites": TaskConfig(
        name="active_sites",
        batch_kind=TaskType.PROTEIN_RESIDUE,
        label_kind=LabelType.RESIDUE_BINARY_VECTOR,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/active_sites/swissprot_active_site_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/active_sites/swissprot_active_site_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column=None,
        residue_positions_column="active_site",
        index_columns=("uniprot_id",),
    ),
    "binding_site": TaskConfig(
        name="binding_site",
        batch_kind=TaskType.PROTEIN_RESIDUE,
        label_kind=LabelType.RESIDUE_BINARY_VECTOR,
        dataset_paths={
            SplitType.RANDOM: _DATA / "properties/binding_site/swissprot_binding_site_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA / "properties/binding_site/swissprot_binding_site_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column=None,
        residue_positions_column="binding_positions",
        index_columns=("uniprot_id",),
    ),
    "align_amylase_activity": TaskConfig(
        name="align_amylase_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_amylase_activity/alpha_amylase__specific_activity_random_80_10_10_standardized.csv",
        },
        target_column="specific_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "align_amylase_expression": TaskConfig(
        name="align_amylase_expression",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_amylase_expression/alpha_amylase__expression_random_80_10_10_standardized.csv",
        },
        target_column="expression_standardized",
        index_columns=("dataset_id",),
    ),
    "align_amylase_stability": TaskConfig(
        name="align_amylase_stability",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_amylase_stability/alpha_amylase__thermostability_random_80_10_10_standardized.csv",
        },
        target_column="thermostability_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_activity": TaskConfig(
        name="align_glucosidase_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_glucosidase_activity/b_glucosidase__activity_random_80_10_10_standardized.csv",
        },
        target_column="activity_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_expression": TaskConfig(
        name="align_glucosidase_expression",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_glucosidase_expression/b_glucosidase__expression_random_80_10_10_standardized.csv",
        },
        target_column="expression_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_melting": TaskConfig(
        name="align_glucosidase_melting",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/align_glucosidase_melting/b_glucosidase__melting_point_random_80_10_10_standardized.csv",
        },
        target_column="melting_point_standardized",
        index_columns=("dataset_id",),
    ),
    "cleeveland_galox": TaskConfig(
        name="cleeveland_galox",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/cleeveland_galox/cleeveland_random_80_10_10_standardized.csv",
        },
        target_column="specific_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "erickson_petase": TaskConfig(
        name="erickson_petase",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/erickson_petase/erickson_random_80_10_10_standardized.csv",
        },
        target_column="sum_aromatic_products_standardized",
        index_columns=("dataset_id",),
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Return the registered config for a task name."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}") from exc
