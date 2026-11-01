"""Central task registry describing dataset paths, labels, and split files."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path

from enzymado import paths_and_constants

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
    split_column: str = _C.split
    ligand_column: str | None = None
    numeric_feature_columns: tuple[str, ...] = ()
    index_columns: tuple[str, ...] = (_C.dataset_id, _C.uniprot_id)
    wildtype_sequence_column: str | None = None
    ec_separator: str = ";"


def _property_paths(subdir: str, stem: str) -> dict[str, Path]:
    base = _P.data_processed_properties / subdir
    return {
        SplitType.RANDOM: base / f"{stem}_random_80_10_10_standardized.csv",
        SplitType.SIMILARITY: (
            base / f"{stem}_similarity_mmseqs2_80_10_10_standardized.csv"
        ),
    }


def _real_case_task(
    name: str,
    target_column: str,
    *,
    stem: str | None = None,
    has_ligand: bool = False,
    numeric_feature_columns: tuple[str, ...] = (),
) -> tuple[str, TaskConfig]:
    file_stem = stem or name
    csv_path = (
        _P.data_processed_real_case
        / name
        / f"{file_stem}_random_80_10_10_standardized.csv"
    )
    return name, TaskConfig(
        name=name,
        batch_kind=(
            TaskType.PROTEIN_LIGAND_GLOBAL if has_ligand else TaskType.PROTEIN_GLOBAL
        ),
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={SplitType.RANDOM: csv_path},
        target_column=f"{target_column}_standardized",
        ligand_column=_C.smiles if has_ligand else None,
        numeric_feature_columns=numeric_feature_columns,
        index_columns=(_C.dataset_id,),
    )


_PROPERTY_TASKS: dict[str, TaskConfig] = {
    "solubility": TaskConfig(
        name="solubility",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.BINARY,
        dataset_paths=_property_paths("solubility", "proteinea_solubility"),
        target_column="solubility",
        index_columns=(_C.dataset_id, _C.uniprot_id),
    ),
    "ph_opt": TaskConfig(
        name="ph_opt",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths=_property_paths("ph_opt", "brenda_phopt"),
        target_column="phopt_median_standardized",
        index_columns=(_C.uniprot_id,),
    ),
    "ec_number": TaskConfig(
        name="ec_number",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.MULTILABEL_EC,
        dataset_paths=_property_paths("ec_number", "swissprot_ec"),
        target_column="ec_number",
        index_columns=(_C.uniprot_id,),
    ),
    "kcat": TaskConfig(
        name="kcat",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths=_property_paths("kcat", "catpred_kcat"),
        target_column="kcat_standardized",
        ligand_column=_C.smiles,
        index_columns=(_C.uniprot_id,),
    ),
    "km": TaskConfig(
        name="km",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths=_property_paths("km", "catpred_km"),
        target_column="km_standardized",
        ligand_column=_C.smiles,
        index_columns=(_C.uniprot_id,),
    ),
    "binding": TaskConfig(
        name="binding",
        batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
        label_kind=LabelType.BINARY,
        dataset_paths=_property_paths("binding", "biolip_binding"),
        target_column="binding",
        ligand_column=_C.smiles,
        index_columns=(_C.uniprot_id,),
    ),
    "dtm": TaskConfig(
        name="dtm",
        batch_kind=TaskType.MUTATION_DELTA_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths=_property_paths("dtm", "fireprotdb_dtm"),
        target_column="dtm_standardized",
        wildtype_sequence_column="wildtype_sequence",
        index_columns=(_C.dataset_id, _C.uniprot_id),
    ),
    "ddg": TaskConfig(
        name="ddg",
        batch_kind=TaskType.MUTATION_DELTA_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        dataset_paths=_property_paths("ddg_mut", "fireprotdb_ddg"),
        target_column="ddg_standardized",
        wildtype_sequence_column="wildtype_sequence",
        index_columns=(_C.dataset_id, _C.uniprot_id),
    ),
    "active_sites": TaskConfig(
        name="active_sites",
        batch_kind=TaskType.PROTEIN_RESIDUE,
        label_kind=LabelType.RESIDUE_BINARY_VECTOR,
        dataset_paths=_property_paths("active_sites", "swissprot_active_site"),
        target_column="active_site",
        index_columns=(_C.uniprot_id,),
    ),
    "binding_site": TaskConfig(
        name="binding_site",
        batch_kind=TaskType.PROTEIN_RESIDUE,
        label_kind=LabelType.RESIDUE_BINARY_VECTOR,
        dataset_paths=_property_paths("binding_site", "swissprot_binding_site"),
        target_column="binding_positions",
        index_columns=(_C.uniprot_id,),
    ),
}

_REAL_CASE_TASKS: dict[str, TaskConfig] = dict(
    (
        _real_case_task(
            "align_amylase_activity",
            "log10_specific_activity",
            stem="alpha_amylase__specific_activity",
        ),
        _real_case_task(
            "align_amylase_expression",
            "expression",
            stem="alpha_amylase__expression",
        ),
        _real_case_task(
            "align_amylase_stability",
            "log10_thermostability",
            stem="alpha_amylase__thermostability",
        ),
        _real_case_task(
            "align_glucosidase_activity",
            "log10_activity",
            stem="b_glucosidase__activity",
        ),
        _real_case_task(
            "align_glucosidase_expression",
            "log10_expression",
            stem="b_glucosidase__expression",
        ),
        _real_case_task(
            "align_glucosidase_melting",
            "melting_point",
            stem="b_glucosidase__melting_point",
        ),
        _real_case_task(
            "cleeveland_galox",
            "log10_specific_activity",
            stem="cleeveland",
            has_ligand=True,
            numeric_feature_columns=("substrate_concentration",),
        ),
        _real_case_task(
            "erickson_petase",
            "log10_sum_aromatic_products_plus_one",
            stem="erickson",
            numeric_feature_columns=(
                "temperature",
                "ph",
                "buffer_c",
                "buffer_np",
                "buffer_h",
                "buffer_b",
                "buffer_g",
            ),
        ),
        _real_case_task("welo5_soraphen_activity", "log10_activity"),
        _real_case_task("pafa_htmek_kcat", "log10_kcat"),
        _real_case_task("pafa_htmek_km", "log10_km"),
        _real_case_task("pafa_htmek_kcat_km", "log10_kcat_km"),
        _real_case_task("daox_epseq_activity", "log_activity"),
        _real_case_task("daox_epseq_stability", "log_stability"),
        _real_case_task("cyp2c9_activity", "activity_score"),
        _real_case_task("cyp2c9_abundance", "abundance_score"),
        _real_case_task("mpro_fret_cleavage", "functional_score"),
        _real_case_task("plpro_activity", "fitness_activity"),
        _real_case_task("plpro_abundance", "fitness_abundance"),
        _real_case_task("amie_acetamide_dms", "normalized_fitness"),
        _real_case_task("amie_propionamide_dms", "normalized_fitness"),
        _real_case_task("amie_isobutyramide_dms", "normalized_fitness"),
        _real_case_task("pyks_solubility", "normalized_er"),
        _real_case_task("trpb3_epistasis", "fitness"),
        _real_case_task("goase_variant_substrate", "log10_activity", has_ligand=True),
        _real_case_task("goldman_olea_thiolase", "log_activity", has_ligand=True),
        _real_case_task(
            "goldman_aminotransferase", "log10_specific_activity", has_ligand=True
        ),
        _real_case_task("fisher_fdh_nacl", "log10_conversion", has_ligand=True),
        _real_case_task("fisher_fdh_nabr", "log10_conversion", has_ligand=True),
        _real_case_task(
            "martinez_esterase_panel", "log10_specific_activity", has_ligand=True
        ),
        _real_case_task(
            "huang_had_phosphatase_c0", "log10_conversion", has_ligand=True
        ),
        _real_case_task(
            "huang_had_phosphatase_c1", "log10_conversion", has_ligand=True
        ),
        _real_case_task(
            "huang_had_phosphatase_c2", "log10_conversion", has_ligand=True
        ),
        _real_case_task("mcba_reaction_activity", "log10_activity", has_ligand=True),
        _real_case_task(
            "ao_3fcr_transaminase", "log10_specific_activity", has_ligand=True
        ),
        _real_case_task("koudelakova_hld_panel", "log10_activity", has_ligand=True),
    )
)

TASK_REGISTRY: dict[str, TaskConfig] = {**_PROPERTY_TASKS, **_REAL_CASE_TASKS}


def get_task_config(task_name: str) -> TaskConfig:
    """Return the registered config for a task name."""
    return TASK_REGISTRY[task_name]
