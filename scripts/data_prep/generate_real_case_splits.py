#!/usr/bin/env python3
"""Regenerate point-stratified random splits for all real-case datasets.

Real case = case studies (the code says real_case, the docs case studies).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from enzymado import dataset_splitting, paths_and_constants
from enzymado.data_loading import data_registry

_C = paths_and_constants.Constants
_P = paths_and_constants.Paths
_SPLIT_SUFFIX = "_random_80_10_10_standardized.csv"


def _real_case_spec(config: data_registry.TaskConfig) -> tuple[Path, str]:
    split_path = config.dataset_paths[data_registry.SplitType.RANDOM]
    source_name = f"{split_path.name.removesuffix(_SPLIT_SUFFIX)}.csv"
    rel_source = split_path.with_name(source_name).relative_to(_P.project_root)
    target_col = config.target_column.removesuffix("_standardized")
    return rel_source, target_col


REAL_CASE_DATASETS: tuple[tuple[Path, str], ...] = tuple(
    _real_case_spec(cfg)
    for cfg in data_registry.TASK_REGISTRY.values()
    if cfg.task_scope == "real_case"
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
    for source_path, target_column in REAL_CASE_DATASETS:
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
        print(f"{source_path} -> {output_path}")
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
