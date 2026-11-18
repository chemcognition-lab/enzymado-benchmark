"""Shared utilities for MLIP-based embedding scripts.

Provides structure loading, atom-to-residue averaging, the main embedding
loop with OOM handling and resume support, and a common argument parser.
Free of mace and ase imports so it loads in every per-model virtualenv.
"""

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time

from jaxtyping import Float, Int
import numpy as np
import shard_writer
import torch
from torch import Tensor


@dataclass(frozen=True)
class Structure:
    """Prepared protein structure loaded from an NPZ file."""

    id: str
    numbers: np.ndarray           # int32 [N]
    positions: np.ndarray         # float32 [N x 3], Angstrom
    residue_index: np.ndarray     # int32 [N], 0-based
    plddt: np.ndarray             # float32 [L]
    charge: int
    sequence: str


def load_structure(path: Path) -> Structure:
    """Load a prepared NPZ structure file into a Structure dataclass."""
    data = np.load(path, allow_pickle=False)
    sequence = str(data["sequence"].item())
    return Structure(
        id=path.stem,
        numbers=data["numbers"],
        positions=data["positions"],
        residue_index=data["residue_index"],
        plddt=data["plddt"],
        charge=int(data["charge"]),
        sequence=sequence,
    )


def to_residues(
    x: Float[Tensor, "N d"],  # noqa: F821
    residue_index: Int[Tensor, "N"],  # noqa: F821
    n_residues: int,
) -> Float[Tensor, "L d"]:  # noqa: F821
    """Average per-atom features to per-residue features.

    Implements the matrix A from ARQUITECTURAS.md section 2: each row of A
    sums to 1 and assigns equal weight to every atom in the residue. Raises
    ValueError if any residue has no atoms.
    """
    n_atoms, d = x.shape
    if n_atoms == 0:
        raise ValueError("x has zero atoms")

    result = torch.zeros(n_residues, d, dtype=x.dtype, device=x.device)
    counts = torch.zeros(n_residues, dtype=torch.long, device=x.device)
    idx_expanded = residue_index.unsqueeze(1).expand(n_atoms, d)
    result.scatter_add_(0, idx_expanded, x)
    counts.scatter_add_(0, residue_index, torch.ones_like(residue_index))

    missing = (counts == 0).nonzero(as_tuple=True)[0]
    if missing.numel() > 0:
        raise ValueError(f"Residues with no atoms: {missing.tolist()}")

    return result / counts.unsqueeze(1).to(x.dtype)


def _npz_paths(structure_dir: Path, ids_file: Path | None) -> tuple[list[Path], int]:
    """Return the NPZ paths to embed and how many listed IDs have no file.

    Without ids_file, every NPZ in structure_dir is used. With ids_file (one
    sequence_id per line), only those IDs are used, in file order.
    """
    if ids_file is None:
        return sorted(structure_dir.glob("*.npz")), 0
    with ids_file.open(encoding="utf-8") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    paths = [structure_dir / f"{seq_id}.npz" for seq_id in ids]
    present = [p for p in paths if p.exists()]
    return present, len(paths) - len(present)


def spatial_chunks(
    positions: np.ndarray, max_atoms: int, halo: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split atoms into (core, context) index pairs for a strictly local model.

    The cores partition all atoms. Each context holds its core plus every
    atom inside the core's bounding box widened by halo on every side, which
    includes every atom within halo (Euclidean) of any core atom. For a model
    whose per-atom output depends only on atoms within halo (message passing:
    layers x cutoff), the core rows computed on the context alone equal the
    rows computed on the whole protein. Cores are cut by recursive bisection
    along the longest box axis until each context has at most max_atoms
    atoms. Index arrays are sorted. Raises ValueError if a one-atom core
    already needs more than max_atoms atoms of context.
    """
    n_atoms = len(positions)
    pending = [np.arange(n_atoms)]
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    while pending:
        core = pending.pop()
        lo = positions[core].min(axis=0) - halo
        hi = positions[core].max(axis=0) + halo
        inside = np.all((positions >= lo) & (positions <= hi), axis=1)
        context = np.flatnonzero(inside)
        if len(context) <= max_atoms:
            chunks.append((core, context))
            continue
        if len(core) == 1:
            raise ValueError(
                f"one atom needs {len(context)} context atoms > max_atoms={max_atoms}"
            )
        axis = int(np.argmax(hi - lo))
        order = core[np.argsort(positions[core, axis], kind="stable")]
        half = len(order) // 2
        pending.append(np.sort(order[:half]))
        pending.append(np.sort(order[half:]))
    return chunks


def embed_structures(
    structure_dir: Path,
    output: Path,
    mean_pooled_output: Path | None,
    embed_fn: Callable[["Structure"], Float[Tensor, "L d"]],  # noqa: F821
    max_atoms: int = 0,
    ids_file: Path | None = None,
    shard_prefix: str = "shard_",
    pooled_dtype: torch.dtype = torch.float32,
) -> None:
    """Iterate over prepared NPZ files and embed each structure.

    ids_file restricts the run to a list of IDs, and shard_prefix names this
    run's shards, so parallel jobs can share one output directory. The pooled
    vector is stored in pooled_dtype (float32 by default; residue shards stay
    bf16). Skips IDs already present in the shard writer (resume support). Catches
    RuntimeErrors whose message contains "out of memory" per protein (covers
    torch.cuda.OutOfMemoryError and TorchScript OOM, both arrive as
    RuntimeError in practice). Empties the cache, counts, and continues.
    Any other RuntimeError propagates and stops the run. Proteins with more
    than max_atoms atoms are skipped when max_atoms > 0.

    Prints a summary: newly embedded, resumed, skipped by size, failed by
    OOM, seconds per newly embedded protein, and peak GPU memory.
    """
    npz_paths, n_missing = _npz_paths(structure_dir, ids_file)
    if n_missing:
        print(f"{n_missing} listed IDs have no .npz in {structure_dir}")
    if not npz_paths:
        print(f"No .npz files found in {structure_dir}")
        return

    n_embedded = 0  # newly embedded this run
    n_resumed = 0   # already present, skipped
    n_skipped_size = 0
    n_oom = 0
    t_start = time.time()

    with ExitStack() as stack:
        hf = stack.enter_context(
            shard_writer.SafetensorsShardWriter(output, prefix=shard_prefix)
        )
        hf_pooled = (
            stack.enter_context(
                shard_writer.SafetensorsPooledWriter(
                    mean_pooled_output, dtype=pooled_dtype
                )
            )
            if mean_pooled_output is not None
            else None
        )

        for npz_path in npz_paths:
            seq_id = npz_path.stem
            if seq_id in hf:
                n_resumed += 1
                continue

            structure = load_structure(npz_path)

            if max_atoms > 0 and len(structure.numbers) > max_atoms:
                n_skipped_size += 1
                print(
                    f"SKIP {seq_id}: {len(structure.numbers)} atoms"
                    f" > max_atoms={max_atoms}"
                )
                continue

            try:
                residue_emb = embed_fn(structure)  # [L x d]
                protein_emb = residue_emb.mean(dim=0)  # [d]
                hf.add(seq_id, residue_emb)
                if hf_pooled is not None:
                    hf_pooled.add(seq_id, protein_emb)
                n_embedded += 1
            except RuntimeError as err:
                # TorchScript OOM arrives as RuntimeError, not the subclass.
                if "out of memory" not in str(err).lower():
                    raise
                torch.cuda.empty_cache()
                n_oom += 1
                print(f"OOM {seq_id}: skipped, cache cleared")

    elapsed = time.time() - t_start
    per_prot = elapsed / max(n_embedded, 1)

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\n--- MLIP embedding summary ---")
    print(f"Embedded (new):    {n_embedded}")
    print(f"Resumed (skipped): {n_resumed}")
    print(f"Skipped (size):    {n_skipped_size}")
    print(f"Missing (no npz):  {n_missing}")
    print(f"Failed (OOM):      {n_oom}")
    print(f"Time per protein:  {per_prot:.4f} s")
    print(f"Peak GPU memory:   {peak_mb:.1f} MB")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared input/output arguments to an argparse parser."""
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Directory containing prepared .npz structure files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Directory for per-residue safetensors shards (created if absent).",
    )
    parser.add_argument(
        "--mean_pooled_output",
        type=str,
        default=None,
        help="Path for mean-pooled per-protein safetensors file (optional).",
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=0,
        help="Skip structures with more than this many atoms; 0 = no limit.",
    )
    parser.add_argument(
        "--ids_file",
        type=str,
        default=None,
        help="Text file with one sequence_id per line; embed only these.",
    )
    parser.add_argument(
        "--shard_prefix",
        type=str,
        default="shard_",
        help="File name prefix for this run's shards (default 'shard_').",
    )
    parser.add_argument(
        "--pooled_dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="dtype of the mean-pooled file (default float32).",
    )


def common_kwargs(args: argparse.Namespace) -> dict:
    """embed_structures keyword arguments taken from add_common_args flags."""
    return {
        "structure_dir": Path(args.input),
        "output": Path(args.output),
        "mean_pooled_output": (
            Path(args.mean_pooled_output) if args.mean_pooled_output else None
        ),
        "max_atoms": args.max_atoms,
        "ids_file": Path(args.ids_file) if args.ids_file else None,
        "shard_prefix": args.shard_prefix,
        "pooled_dtype": getattr(torch, args.pooled_dtype),
    }
