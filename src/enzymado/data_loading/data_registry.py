"""Central task registry describing dataset paths, labels, and split files."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
import sys

from enzymado import paths_and_constants

# enum.StrEnum was added in Python 3.11; provide a shim for 3.10
if sys.version_info < (3, 11) and not hasattr(enum, "StrEnum"):

    class _StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]


_C = paths_and_constants.Constants
_P = paths_and_constants.Paths
PROJECT_ROOT = _P.project_root


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


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Configuration needed to build datasets and loaders for one task."""

    name: str
    batch_kind: TaskType
    label_kind: LabelType
    dataset_paths: dict[str, Path]
    # Source column the label is built from. Every label_kind reads it: a scalar
    # for regression, a class for binary, an EC string for multilabel_ec, and the
    # annotated residue positions for residue_binary_vector.
    target_column: str
    task_scope: str = "properties"
    sequence_column: str = _C.protein_sequence
    split_column: str = "split"
    ligand_column: str | None = None
    numeric_feature_columns: tuple[str, ...] = ()
    index_columns: tuple[str, ...] = (_C.dataset_id, "uniprot_id")
    wildtype_sequence_column: str | None = None
    ec_separator: str = ";"


_DATA = _P.data_processed_dir


TASK_REGISTRY: dict[str, TaskConfig] = {
    "solubility": TaskConfig(
        name="solubility",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.BINARY,
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "properties/solubility"
            / "proteinea_solubility_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/solubility"
            / "proteinea_solubility_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="solubility",
        index_columns=("dataset_id", "uniprot_id"),
    ),
    "ph_opt": TaskConfig(
        name="ph_opt",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "properties/ph_opt/brenda_phopt_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/ph_opt"
            / "brenda_phopt_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="phopt_median_standardized",
        index_columns=("uniprot_id",),
    ),
    "ec_number": TaskConfig(
        name="ec_number",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.MULTILABEL_EC,
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "properties/ec_number/swissprot_ec_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/ec_number"
            / "swissprot_ec_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="ec_number",
        index_columns=("uniprot_id",),
    ),
    "kcat": TaskConfig(
        name="kcat",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "properties/kcat/catpred_kcat_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/kcat"
            / "catpred_kcat_similarity_mmseqs2_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "properties/km/catpred_km_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/km/catpred_km_similarity_mmseqs2_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "properties/binding/biolip_binding_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/binding"
            / "biolip_binding_similarity_mmseqs2_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "properties/dtm/fireprotdb_dtm_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/dtm"
            / "fireprotdb_dtm_similarity_mmseqs2_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "properties/ddg_mut/fireprotdb_ddg_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/ddg_mut"
            / "fireprotdb_ddg_similarity_mmseqs2_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "properties/active_sites"
            / "swissprot_active_site_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/active_sites"
            / "swissprot_active_site_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="active_site",
        index_columns=("uniprot_id",),
    ),
    "binding_site": TaskConfig(
        name="binding_site",
        batch_kind=TaskType.PROTEIN_RESIDUE,
        label_kind=LabelType.RESIDUE_BINARY_VECTOR,
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "properties/binding_site"
            / "swissprot_binding_site_random_80_10_10_standardized.csv",
            SplitType.SIMILARITY: _DATA
            / "properties/binding_site"
            / "swissprot_binding_site_similarity_mmseqs2_80_10_10_standardized.csv",
        },
        target_column="binding_positions",
        index_columns=("uniprot_id",),
    ),
    "align_amylase_activity": TaskConfig(
        name="align_amylase_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/align_amylase_activity"
            / "alpha_amylase__specific_activity_random_80_10_10_standardized.csv",
        },
        target_column="log10_specific_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "align_amylase_expression": TaskConfig(
        name="align_amylase_expression",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/align_amylase_expression"
            / "alpha_amylase__expression_random_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "real_case/align_amylase_stability"
            / "alpha_amylase__thermostability_random_80_10_10_standardized.csv",
        },
        target_column="log10_thermostability_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_activity": TaskConfig(
        name="align_glucosidase_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/align_glucosidase_activity"
            / "b_glucosidase__activity_random_80_10_10_standardized.csv",
        },
        target_column="log10_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_expression": TaskConfig(
        name="align_glucosidase_expression",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/align_glucosidase_expression"
            / "b_glucosidase__expression_random_80_10_10_standardized.csv",
        },
        target_column="log10_expression_standardized",
        index_columns=("dataset_id",),
    ),
    "align_glucosidase_melting": TaskConfig(
        name="align_glucosidase_melting",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/align_glucosidase_melting"
            / "b_glucosidase__melting_point_random_80_10_10_standardized.csv",
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
            SplitType.RANDOM: _DATA
            / "real_case/cleeveland_galox/cleeveland_random_80_10_10_standardized.csv",
        },
        target_column="log10_specific_activity_standardized",
        ligand_column="smiles",
        numeric_feature_columns=("substrate_concentration",),
        index_columns=("dataset_id",),
    ),
    "erickson_petase": TaskConfig(
        name="erickson_petase",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/erickson_petase/erickson_random_80_10_10_standardized.csv",
        },
        target_column="log10_sum_aromatic_products_plus_one_standardized",
        numeric_feature_columns=(
            "temperature",
            "ph",
            "buffer_c",
            "buffer_np",
            "buffer_h",
            "buffer_b",
            "buffer_g",
        ),
        index_columns=("dataset_id",),
    ),
    "welo5_soraphen_activity": TaskConfig(
        name="welo5_soraphen_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/welo5_soraphen_activity"
            / "welo5_soraphen_activity_random_80_10_10_standardized.csv",
        },
        target_column="log10_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "pafa_htmek_kcat": TaskConfig(
        name="pafa_htmek_kcat",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/pafa_htmek_kcat"
            / "pafa_htmek_kcat_random_80_10_10_standardized.csv",
        },
        target_column="log10_kcat_standardized",
        index_columns=("dataset_id",),
    ),
    "pafa_htmek_km": TaskConfig(
        name="pafa_htmek_km",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/pafa_htmek_km"
            / "pafa_htmek_km_random_80_10_10_standardized.csv",
        },
        target_column="log10_km_standardized",
        index_columns=("dataset_id",),
    ),
    "pafa_htmek_kcat_km": TaskConfig(
        name="pafa_htmek_kcat_km",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/pafa_htmek_kcat_km"
            / "pafa_htmek_kcat_km_random_80_10_10_standardized.csv",
        },
        target_column="log10_kcat_km_standardized",
        index_columns=("dataset_id",),
    ),
    "daox_epseq_activity": TaskConfig(
        name="daox_epseq_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/daox_epseq_activity"
            / "daox_epseq_activity_random_80_10_10_standardized.csv",
        },
        target_column="log_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "daox_epseq_stability": TaskConfig(
        name="daox_epseq_stability",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/daox_epseq_stability"
            / "daox_epseq_stability_random_80_10_10_standardized.csv",
        },
        target_column="log_stability_standardized",
        index_columns=("dataset_id",),
    ),
    "cyp2c9_activity": TaskConfig(
        name="cyp2c9_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/cyp2c9_activity"
            / "cyp2c9_activity_random_80_10_10_standardized.csv",
        },
        target_column="activity_score_standardized",
        index_columns=("dataset_id",),
    ),
    "cyp2c9_abundance": TaskConfig(
        name="cyp2c9_abundance",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/cyp2c9_abundance"
            / "cyp2c9_abundance_random_80_10_10_standardized.csv",
        },
        target_column="abundance_score_standardized",
        index_columns=("dataset_id",),
    ),
    "mpro_fret_cleavage": TaskConfig(
        name="mpro_fret_cleavage",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/mpro_fret_cleavage"
            / "mpro_fret_cleavage_random_80_10_10_standardized.csv",
        },
        target_column="functional_score_standardized",
        index_columns=("dataset_id",),
    ),
    "plpro_activity": TaskConfig(
        name="plpro_activity",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/plpro_activity"
            / "plpro_activity_random_80_10_10_standardized.csv",
        },
        target_column="fitness_activity_standardized",
        index_columns=("dataset_id",),
    ),
    "plpro_abundance": TaskConfig(
        name="plpro_abundance",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/plpro_abundance"
            / "plpro_abundance_random_80_10_10_standardized.csv",
        },
        target_column="fitness_abundance_standardized",
        index_columns=("dataset_id",),
    ),
    "amie_acetamide_dms": TaskConfig(
        name="amie_acetamide_dms",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/amie_acetamide_dms"
            / "amie_acetamide_dms_random_80_10_10_standardized.csv",
        },
        target_column="normalized_fitness_standardized",
        index_columns=("dataset_id",),
    ),
    "amie_propionamide_dms": TaskConfig(
        name="amie_propionamide_dms",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/amie_propionamide_dms"
            / "amie_propionamide_dms_random_80_10_10_standardized.csv",
        },
        target_column="normalized_fitness_standardized",
        index_columns=("dataset_id",),
    ),
    "amie_isobutyramide_dms": TaskConfig(
        name="amie_isobutyramide_dms",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/amie_isobutyramide_dms"
            / "amie_isobutyramide_dms_random_80_10_10_standardized.csv",
        },
        target_column="normalized_fitness_standardized",
        index_columns=("dataset_id",),
    ),
    "pyks_solubility": TaskConfig(
        name="pyks_solubility",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/pyks_solubility"
            / "pyks_solubility_random_80_10_10_standardized.csv",
        },
        target_column="normalized_er_standardized",
        index_columns=("dataset_id",),
    ),
    "trpb3_epistasis": TaskConfig(
        name="trpb3_epistasis",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/trpb3_epistasis"
            / "trpb3_epistasis_random_80_10_10_standardized.csv",
        },
        target_column="fitness_standardized",
        index_columns=("dataset_id",),
    ),
    "goase_variant_substrate": TaskConfig(
        name="goase_variant_substrate",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/goase_variant_substrate"
            / "goase_variant_substrate_random_80_10_10_standardized.csv",
        },
        target_column="log10_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "goldman_olea_thiolase": TaskConfig(
        name="goldman_olea_thiolase",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/goldman_olea_thiolase"
            / "goldman_olea_thiolase_random_80_10_10_standardized.csv",
        },
        target_column="log_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "goldman_aminotransferase": TaskConfig(
        name="goldman_aminotransferase",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/goldman_aminotransferase"
            / "goldman_aminotransferase_random_80_10_10_standardized.csv",
        },
        target_column="log10_specific_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "fisher_fdh_nacl": TaskConfig(
        name="fisher_fdh_nacl",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/fisher_fdh_nacl"
            / "fisher_fdh_nacl_random_80_10_10_standardized.csv",
        },
        target_column="log10_conversion_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "fisher_fdh_nabr": TaskConfig(
        name="fisher_fdh_nabr",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/fisher_fdh_nabr"
            / "fisher_fdh_nabr_random_80_10_10_standardized.csv",
        },
        target_column="log10_conversion_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "martinez_esterase_panel": TaskConfig(
        name="martinez_esterase_panel",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/martinez_esterase_panel"
            / "martinez_esterase_panel_random_80_10_10_standardized.csv",
        },
        target_column="log10_specific_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "huang_had_phosphatase_c0": TaskConfig(
        name="huang_had_phosphatase_c0",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/huang_had_phosphatase_c0"
            / "huang_had_phosphatase_c0_random_80_10_10_standardized.csv",
        },
        target_column="log10_conversion_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "huang_had_phosphatase_c1": TaskConfig(
        name="huang_had_phosphatase_c1",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/huang_had_phosphatase_c1"
            / "huang_had_phosphatase_c1_random_80_10_10_standardized.csv",
        },
        target_column="log10_conversion_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "huang_had_phosphatase_c2": TaskConfig(
        name="huang_had_phosphatase_c2",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/huang_had_phosphatase_c2"
            / "huang_had_phosphatase_c2_random_80_10_10_standardized.csv",
        },
        target_column="log10_conversion_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "mcba_reaction_activity": TaskConfig(
        name="mcba_reaction_activity",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/mcba_reaction_activity"
            / "mcba_reaction_activity_random_80_10_10_standardized.csv",
        },
        target_column="log10_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "ao_3fcr_transaminase": TaskConfig(
        name="ao_3fcr_transaminase",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/ao_3fcr_transaminase"
            / "ao_3fcr_transaminase_random_80_10_10_standardized.csv",
        },
        target_column="log10_specific_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
    "koudelakova_hld_panel": TaskConfig(
        name="koudelakova_hld_panel",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA
            / "real_case/koudelakova_hld_panel"
            / "koudelakova_hld_panel_random_80_10_10_standardized.csv",
        },
        target_column="log10_activity_standardized",
        ligand_column="smiles",
        index_columns=("dataset_id",),
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Return the registered config for a task name."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(
            f"Unknown task '{task_name}'. Available tasks: {available}"
        ) from exc
