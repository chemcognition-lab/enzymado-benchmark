#!/usr/bin/env python3
"""Regenerate point-stratified random splits for all real-case datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enzymado.dataset_splitting import (  # noqa: E402
    build_metadata_path,
    build_output_csv_path,
    split_dataset_csv,
)

REAL_CASE_DATASETS = (
    (
        Path(
            "data/processed/real_case/align_amylase_activity/"
            "alpha_amylase__specific_activity.csv"
        ),
        "log10_specific_activity",
    ),
    (
        Path(
            "data/processed/real_case/align_amylase_expression/"
            "alpha_amylase__expression.csv"
        ),
        "expression",
    ),
    (
        Path(
            "data/processed/real_case/align_amylase_stability/"
            "alpha_amylase__thermostability.csv"
        ),
        "log10_thermostability",
    ),
    (
        Path(
            "data/processed/real_case/align_glucosidase_activity/"
            "b_glucosidase__activity.csv"
        ),
        "log10_activity",
    ),
    (
        Path(
            "data/processed/real_case/align_glucosidase_expression/"
            "b_glucosidase__expression.csv"
        ),
        "log10_expression",
    ),
    (
        Path(
            "data/processed/real_case/align_glucosidase_melting/"
            "b_glucosidase__melting_point.csv"
        ),
        "melting_point",
    ),
    (
        Path("data/processed/real_case/cleeveland_galox/cleeveland.csv"),
        "log10_specific_activity",
    ),
    (
        Path("data/processed/real_case/erickson_petase/erickson.csv"),
        "log10_sum_aromatic_products_plus_one",
    ),
    (
        Path(
            "data/processed/real_case/welo5_soraphen_activity/"
            "welo5_soraphen_activity.csv"
        ),
        "log10_activity",
    ),
    (
        Path("data/processed/real_case/pafa_htmek_kcat/pafa_htmek_kcat.csv"),
        "log10_kcat",
    ),
    (
        Path("data/processed/real_case/pafa_htmek_km/pafa_htmek_km.csv"),
        "log10_km",
    ),
    (
        Path("data/processed/real_case/pafa_htmek_kcat_km/pafa_htmek_kcat_km.csv"),
        "log10_kcat_km",
    ),
    (
        Path("data/processed/real_case/daox_epseq_activity/daox_epseq_activity.csv"),
        "log_activity",
    ),
    (
        Path("data/processed/real_case/daox_epseq_stability/daox_epseq_stability.csv"),
        "log_stability",
    ),
    (
        Path("data/processed/real_case/cyp2c9_activity/cyp2c9_activity.csv"),
        "activity_score",
    ),
    (
        Path("data/processed/real_case/cyp2c9_abundance/cyp2c9_abundance.csv"),
        "abundance_score",
    ),
    (
        Path("data/processed/real_case/mpro_fret_cleavage/mpro_fret_cleavage.csv"),
        "functional_score",
    ),
    (
        Path("data/processed/real_case/plpro_activity/plpro_activity.csv"),
        "fitness_activity",
    ),
    (
        Path("data/processed/real_case/plpro_abundance/plpro_abundance.csv"),
        "fitness_abundance",
    ),
    (
        Path("data/processed/real_case/amie_acetamide_dms/amie_acetamide_dms.csv"),
        "normalized_fitness",
    ),
    (
        Path(
            "data/processed/real_case/amie_propionamide_dms/amie_propionamide_dms.csv"
        ),
        "normalized_fitness",
    ),
    (
        Path(
            "data/processed/real_case/amie_isobutyramide_dms/amie_isobutyramide_dms.csv"
        ),
        "normalized_fitness",
    ),
    (
        Path("data/processed/real_case/pyks_solubility/pyks_solubility.csv"),
        "normalized_er",
    ),
    (
        Path("data/processed/real_case/trpb3_epistasis/trpb3_epistasis.csv"),
        "fitness",
    ),
    (
        Path(
            "data/processed/real_case/goase_variant_substrate/"
            "goase_variant_substrate.csv"
        ),
        "log10_activity",
    ),
    (
        Path(
            "data/processed/real_case/goldman_olea_thiolase/goldman_olea_thiolase.csv"
        ),
        "log_activity",
    ),
    (
        Path(
            "data/processed/real_case/goldman_aminotransferase/"
            "goldman_aminotransferase.csv"
        ),
        "log10_specific_activity",
    ),
    (
        Path("data/processed/real_case/fisher_fdh_nacl/fisher_fdh_nacl.csv"),
        "log10_conversion",
    ),
    (
        Path("data/processed/real_case/fisher_fdh_nabr/fisher_fdh_nabr.csv"),
        "log10_conversion",
    ),
    (
        Path(
            "data/processed/real_case/martinez_esterase_panel/"
            "martinez_esterase_panel.csv"
        ),
        "log10_specific_activity",
    ),
    (
        Path(
            "data/processed/real_case/huang_had_phosphatase_c0/"
            "huang_had_phosphatase_c0.csv"
        ),
        "log10_conversion",
    ),
    (
        Path(
            "data/processed/real_case/huang_had_phosphatase_c1/"
            "huang_had_phosphatase_c1.csv"
        ),
        "log10_conversion",
    ),
    (
        Path(
            "data/processed/real_case/huang_had_phosphatase_c2/"
            "huang_had_phosphatase_c2.csv"
        ),
        "log10_conversion",
    ),
    (
        Path(
            "data/processed/real_case/mcba_reaction_activity/mcba_reaction_activity.csv"
        ),
        "log10_activity",
    ),
    (
        Path("data/processed/real_case/ao_3fcr_transaminase/ao_3fcr_transaminase.csv"),
        "log10_specific_activity",
    ),
    (
        Path(
            "data/processed/real_case/koudelakova_hld_panel/koudelakova_hld_panel.csv"
        ),
        "log10_activity",
    ),
)


def _metadata_paths(source_path: Path) -> tuple[Path, ...]:
    canonical_path = build_metadata_path(
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
    os.chdir(PROJECT_ROOT)
    for relative_source_path, target_column in REAL_CASE_DATASETS:
        source_path = relative_source_path
        output_path = build_output_csv_path(
            source_path,
            mode="random",
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
        )
        summaries = []
        for metadata_path in _metadata_paths(source_path):
            summary = split_dataset_csv(
                source_path,
                output_path,
                mode="random",
                train_fraction=0.8,
                val_fraction=0.1,
                test_fraction=0.1,
                sequence_col="protein_sequence",
                random_split_unit="point",
                point_id_col="dataset_id",
                stratify_col=target_column,
                stratify_bins=10,
                seed=42,
                standardization_cols=(target_column,),
                metadata_path=metadata_path,
            )
            summaries.append(summary)
        if any(summary != summaries[0] for summary in summaries[1:]):
            raise RuntimeError(f"Inconsistent split summaries for {source_path}.")
        print(f"{relative_source_path} -> {output_path}")
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
