#!/usr/bin/env python3
"""Regenerate point-stratified random splits for all real-case datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from enzymado import dataset_splitting, paths_and_constants

_C = paths_and_constants.Constants
_P = paths_and_constants.Paths
_REAL_CASE_DIR = Path("data/processed/real_case")


def _dataset_spec(spec: str, target: str) -> tuple[Path, str]:
    rel_path = spec if "/" in spec else f"{spec}/{spec}.csv"
    return _REAL_CASE_DIR / rel_path, target


_RAW_SPECS: tuple[tuple[str, str], ...] = (
    (
        "align_amylase_activity/alpha_amylase__specific_activity.csv",
        "log10_specific_activity",
    ),
    ("align_amylase_expression/alpha_amylase__expression.csv", "expression"),
    (
        "align_amylase_stability/alpha_amylase__thermostability.csv",
        "log10_thermostability",
    ),
    ("align_glucosidase_activity/b_glucosidase__activity.csv", "log10_activity"),
    ("align_glucosidase_expression/b_glucosidase__expression.csv", "log10_expression"),
    ("align_glucosidase_melting/b_glucosidase__melting_point.csv", "melting_point"),
    ("cleeveland_galox/cleeveland.csv", "log10_specific_activity"),
    ("erickson_petase/erickson.csv", "log10_sum_aromatic_products_plus_one"),
    ("welo5_soraphen_activity", "log10_activity"),
    ("pafa_htmek_kcat", "log10_kcat"),
    ("pafa_htmek_km", "log10_km"),
    ("pafa_htmek_kcat_km", "log10_kcat_km"),
    ("daox_epseq_activity", "log_activity"),
    ("daox_epseq_stability", "log_stability"),
    ("cyp2c9_activity", "activity_score"),
    ("cyp2c9_abundance", "abundance_score"),
    ("mpro_fret_cleavage", "functional_score"),
    ("plpro_activity", "fitness_activity"),
    ("plpro_abundance", "fitness_abundance"),
    ("amie_acetamide_dms", "normalized_fitness"),
    ("amie_propionamide_dms", "normalized_fitness"),
    ("amie_isobutyramide_dms", "normalized_fitness"),
    ("pyks_solubility", "normalized_er"),
    ("trpb3_epistasis", "fitness"),
    ("goase_variant_substrate", "log10_activity"),
    ("goldman_olea_thiolase", "log_activity"),
    ("goldman_aminotransferase", "log10_specific_activity"),
    ("fisher_fdh_nacl", "log10_conversion"),
    ("fisher_fdh_nabr", "log10_conversion"),
    ("martinez_esterase_panel", "log10_specific_activity"),
    ("huang_had_phosphatase_c0", "log10_conversion"),
    ("huang_had_phosphatase_c1", "log10_conversion"),
    ("huang_had_phosphatase_c2", "log10_conversion"),
    ("mcba_reaction_activity", "log10_activity"),
    ("ao_3fcr_transaminase", "log10_specific_activity"),
    ("koudelakova_hld_panel", "log10_activity"),
)
REAL_CASE_DATASETS: tuple[tuple[Path, str], ...] = tuple(
    _dataset_spec(spec, target) for spec, target in _RAW_SPECS
)


def _metadata_paths(source_path: Path) -> tuple[Path, ...]:
    """Return existing or canonical split metadata sidecar paths for a source CSV."""
    canonical_path = dataset_splitting.build_metadata_path(
        source_path,
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    legacy_path = source_path.with_name(
        f"{source_path.stem}_random_80_10_10_splits.metadata.json"
    )
    existing_paths = tuple(
        path for path in (canonical_path, legacy_path) if path.exists()
    )
    return existing_paths or (canonical_path,)


def main() -> None:
    """Regenerate standardized files and all tracked provenance sidecars."""
    os.chdir(_P.project_root)
    for relative_source_path, target_column in REAL_CASE_DATASETS:
        source_path = relative_source_path
        output_path = dataset_splitting.build_output_csv_path(
            source_path,
            mode="random",
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
        )
        summaries = [
            dataset_splitting.split_dataset_csv(
                source_path,
                output_path,
                mode="random",
                train_fraction=0.8,
                val_fraction=0.1,
                test_fraction=0.1,
                sequence_col=_C.protein_sequence,
                random_split_unit="point",
                point_id_col=_C.dataset_id,
                stratify_col=target_column,
                stratify_bins=10,
                seed=42,
                standardization_cols=(target_column,),
                metadata_path=metadata_path,
            )
            for metadata_path in _metadata_paths(source_path)
        ]
        print(f"{relative_source_path} -> {output_path}")
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
