#!/usr/bin/env python
"""Build the no-PLM baseline: ProtParam descriptors and a residue one-hot.

Every protein in the pool gets two representations, stored like the PLM ones:

* pooled (36 values, float32): amino acid composition (20), secondary structure
  fractions (3), molecular weight, pI, GRAVY, instability index, aromaticity,
  reduced and oxidized extinction coefficients, net charge at pH 7.4 (8), and
  mean, SD, max, min and median of the Vihinen flexibility profile (5). Float32
  keeps the small mutant - wild type differences that bf16 would round away.
* residue (L x 20, bf16): one-hot amino acid identity, used by the residue
  tasks and the attention route. Its mean over residues is the composition.

ProtParam only knows the 20 standard residues, so U, O, B, Z and X are removed
before the descriptors. In the one-hot, U counts as C, O as K, B as half D and
half N, Z as half E and half Q, and X is a zero row. Biopython's flexibility
needs more than 9 residues; shorter proteins use the per-residue Vihinen values.

Usage:
    python scripts/embeddings/build_noplm_features.py --workers 32
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch

_REPO = Path(__file__).resolve().parents[2]
POOL = _REPO / "data" / "processed" / "protein_pool.csv"
POOLED_OUT = (
    _REPO / "data" / "embeddings_mean_pooled" / "protein_sequences_unique" / "noplm.safetensors"
)
RESIDUE_OUT = Path(
    "/scratch/chemist/enzymado/data/embeddings/protein_sequences_unique/noplm_shards"
)
STANDARD = "ACDEFGHIKLMNPQRSTVWY"
ONE_HOT_SHARES = {
    **{aa: {aa: 1.0} for aa in STANDARD},
    "U": {"C": 1.0},
    "O": {"K": 1.0},
    "B": {"D": 0.5, "N": 0.5},
    "Z": {"E": 0.5, "Q": 0.5},
}
FEATURES = (
    [f"AAC_{aa}" for aa in STANDARD]
    + ["Frac_Helix", "Frac_Turn", "Frac_Sheet"]
    + [
        "Molecular_Weight",
        "Isoelectric_Point",
        "GRAVY",
        "Instability_Index",
        "Aromaticity",
        "Extinction_Coef_Reduced",
        "Extinction_Coef_Oxidized",
        "Net_Charge_7_4",
    ]
    + ["Flex_Mean", "Flex_Std", "Flex_Max", "Flex_Min", "Flex_Median"]
)


def descriptors(sequence: str) -> tuple[np.ndarray, bool, bool]:
    """Return the 36 descriptors, whether residues were removed, and the flex fallback."""
    from Bio.SeqUtils import ProtParamData
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    clean = "".join(aa for aa in sequence if aa in STANDARD)
    analysis = ProteinAnalysis(clean)
    composition = [clean.count(aa) / len(clean) for aa in STANDARD]
    reduced, oxidized = analysis.molar_extinction_coefficient()
    flex = analysis.flexibility()
    fallback = not flex
    if fallback:
        flex = [ProtParamData.Flex[aa] for aa in clean]
    flex = np.asarray(flex, dtype=np.float64)
    values = [
        *composition,
        *analysis.secondary_structure_fraction(),
        analysis.molecular_weight(),
        analysis.isoelectric_point(),
        analysis.gravy(),
        analysis.instability_index() if len(clean) > 1 else 0.0,
        analysis.aromaticity(),
        reduced,
        oxidized,
        analysis.charge_at_pH(7.4),
        flex.mean(),
        flex.std(),
        flex.max(),
        flex.min(),
        float(np.median(flex)),
    ]
    return np.asarray(values, dtype=np.float32), clean != sequence, fallback


def one_hot(sequence: str) -> torch.Tensor:
    """Return the L x 20 residue one-hot, with fractional rows for ambiguous codes."""
    matrix = np.zeros((len(sequence), len(STANDARD)), dtype=np.float32)
    for position, aa in enumerate(sequence):
        for target, share in ONE_HOT_SHARES.get(aa, {}).items():
            matrix[position, STANDARD.index(target)] = share
    return torch.from_numpy(matrix).to(torch.bfloat16)


def build_shard(args: tuple[int, list[tuple[str, str]], str]) -> dict:
    """Compute one residue shard and the descriptors of its proteins.

    Descriptors come back as numpy arrays: torch tensors would travel between
    processes as shared-memory file descriptors and exhaust them.
    """
    index, rows, residue_dir = args
    pooled, residue = {}, {}
    removed = fallback = 0
    for sequence_id, sequence in rows:
        vector, was_cleaned, used_fallback = descriptors(sequence)
        pooled[sequence_id] = vector
        residue[sequence_id] = one_hot(sequence)
        removed += was_cleaned
        fallback += used_fallback
    save_file(residue, str(Path(residue_dir) / f"shard_{index:05d}.safetensors"))
    return {"pooled": pooled, "removed": removed, "fallback": fallback}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--pooled-out", type=Path, default=POOLED_OUT)
    parser.add_argument("--residue-out", type=Path, default=RESIDUE_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=5000)
    args = parser.parse_args()

    with args.pool.open(newline="") as handle:
        rows = [
            (row["sequence_id"], row["protein_sequence"].strip().upper())
            for row in csv.DictReader(handle)
        ]
    args.residue_out.mkdir(parents=True, exist_ok=True)
    chunks = [
        (index, rows[start : start + args.shard_size], str(args.residue_out))
        for index, start in enumerate(range(0, len(rows), args.shard_size))
    ]
    pooled: dict[str, np.ndarray] = {}
    removed = fallback = 0
    with Pool(args.workers) as pool:
        for done, result in enumerate(pool.imap_unordered(build_shard, chunks), 1):
            pooled.update(result["pooled"])
            removed += result["removed"]
            fallback += result["fallback"]
            print(f"  {done}/{len(chunks)} shards", flush=True)

    if not np.isfinite(np.stack(list(pooled.values()))).all():
        raise ValueError("Non-finite descriptor values.")
    args.pooled_out.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: torch.from_numpy(value) for key, value in pooled.items()}, str(args.pooled_out))
    metadata = {
        "features": FEATURES,
        "proteins": len(pooled),
        "residue_shards": len(chunks),
        "proteins_with_nonstandard_removed": removed,
        "proteins_with_flexibility_fallback": fallback,
    }
    args.pooled_out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata | {"features": len(FEATURES)}))


if __name__ == "__main__":
    main()
