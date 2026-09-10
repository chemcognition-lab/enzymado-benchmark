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
        "specific_activity",
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
        "thermostability",
    ),
    (
        Path(
            "data/processed/real_case/align_glucosidase_activity/"
            "b_glucosidase__activity.csv"
        ),
        "activity",
    ),
    (
        Path(
            "data/processed/real_case/align_glucosidase_expression/"
            "b_glucosidase__expression.csv"
        ),
        "expression",
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
        "specific_activity",
    ),
    (
        Path("data/processed/real_case/erickson_petase/erickson.csv"),
        "sum_aromatic_products",
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
