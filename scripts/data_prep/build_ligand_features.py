#!/usr/bin/env python
"""Precompute ligand features for every SMILES in the benchmark.

Each SMILES gets descriptastorus RDKit2DNormalized (200 normalized 2D descriptors).
These are 2D, so stereoisomers share a vector. The result is cached once so
training only looks features up. Non-finite descriptor values are stored as 0
and counted in the metadata.

Usage:
    python scripts/data_prep/build_ligand_features.py --workers 32
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado.data_loading import data_registry  # noqa: E402

OUT = _REPO / "data" / "processed" / "ligand_features" / "descriptastorus.npz"
GENERATORS = ("RDKit2DNormalized",)


def _generators():
    from descriptastorus.descriptors import rdNormalizedDescriptors
    return (rdNormalizedDescriptors.RDKit2DNormalized(),)


def featurize(smiles: str) -> tuple[str, np.ndarray | None, int]:
    """Concatenated features for one SMILES, or None if RDKit cannot parse it."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    parts, non_finite = [], 0
    for generator in GENERATORS_CACHE:
        result = generator.process(smiles)
        if result is None or not result[0]:
            return smiles, None, 0
        values = np.asarray(result[1:], dtype=np.float64)
        non_finite += int((~np.isfinite(values)).sum())
        parts.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    return smiles, np.concatenate(parts).astype(np.float32), non_finite


def _init() -> None:
    global GENERATORS_CACHE
    GENERATORS_CACHE = _generators()


def benchmark_smiles() -> list[str]:
    smiles = set()
    for config in data_registry.TASK_REGISTRY.values():
        if not getattr(config, "ligand_column", None):
            continue
        for path in config.dataset_paths.values():
            with path.open(encoding="utf-8") as handle:
                smiles.update(r[config.ligand_column].strip() for r in csv.DictReader(handle))
    return sorted(smiles - {""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    smiles = benchmark_smiles()
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(featurize, smiles, chunksize=64)
    failed = [s for s, v, _ in results if v is None]
    if failed:
        raise SystemExit(f"RDKit could not featurize {len(failed)} SMILES, e.g. {failed[:3]}")
    matrix = np.stack([v for _, v, _ in results])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_suffix(".tmp.npz")
    np.savez_compressed(staging, smiles=np.array(smiles), features=matrix)
    staging.replace(args.out)
    meta = {
        "generators": list(GENERATORS),
        "dims": int(matrix.shape[1]),
        "n_smiles": len(smiles),
        "smiles_with_non_finite_values": sum(1 for _, _, n in results if n),
        "non_finite_policy": "stored as 0",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(smiles):,} SMILES x {matrix.shape[1]} features -> {args.out}")


if __name__ == "__main__":
    main()
