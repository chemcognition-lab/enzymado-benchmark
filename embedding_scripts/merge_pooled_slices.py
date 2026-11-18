"""Merge per-slice mean-pooled files into one safetensors file.

Parallel embedding jobs each write their own pooled file. This script joins
them, refuses duplicate IDs or mixed dtypes and dimensions, and with
--shard_dir checks that the pooled IDs are exactly the IDs in the residue
shards. The output is written under a temporary name and renamed.

Example:
    python embedding_scripts/merge_pooled_slices.py \\
        --input_dir /scratch/.../embeddings/mlip/pooled_slices/so3lr_tauri_m_2606 \\
        --shard_dir /scratch/.../embeddings/mlip/so3lr_tauri_m_2606_shards \\
        --output data/embeddings_mean_pooled/mlip/so3lr_tauri_m_2606.safetensors
"""

import argparse
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def merge_pooled(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Return all tensors from paths; raise ValueError on any inconsistency."""
    merged: dict[str, torch.Tensor] = {}
    origin: dict[str, str] = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in merged:
                    raise ValueError(f"{key} is in {origin[key]} and {path.name}")
                merged[key] = handle.get_tensor(key)
                origin[key] = path.name
    if not merged:
        raise ValueError("no tensors found")
    first = next(iter(merged.values()))
    for key, tensor in merged.items():
        if tensor.dtype != first.dtype or tensor.shape != first.shape:
            raise ValueError(
                f"{key}: {tensor.dtype} {tuple(tensor.shape)} differs from "
                f"{first.dtype} {tuple(first.shape)}"
            )
    return merged


def shard_keys(shard_dir: Path) -> set[str]:
    """Return every key in the residue shards; raise ValueError on duplicates."""
    keys: set[str] = set()
    for path in sorted(shard_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in keys:
                    raise ValueError(f"{key} appears in two shards ({path.name})")
                keys.add(key)
    return keys


def run(input_dir: Path, output: Path, shard_dir: Path | None) -> None:
    paths = sorted(input_dir.glob("*.safetensors"))
    merged = merge_pooled(paths)
    if shard_dir is not None:
        residue = shard_keys(shard_dir)
        if residue != set(merged):
            only_pooled = len(set(merged) - residue)
            only_shards = len(residue - set(merged))
            raise ValueError(
                f"pooled and shard IDs differ: {only_pooled} only pooled, "
                f"{only_shards} only in shards"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    save_file(dict(sorted(merged.items())), str(tmp))
    os.replace(tmp, output)
    first = next(iter(merged.values()))
    print(
        f"{len(merged)} vectors from {len(paths)} files, {first.dtype}, "
        f"dim {first.shape[-1]} -> {output}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge per-slice mean-pooled safetensors files."
    )
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Directory of per-slice pooled .safetensors files.")
    parser.add_argument("--output", required=True, type=str,
                        help="Merged pooled .safetensors path.")
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Residue shard directory to cross-check IDs against.")
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(
        Path(_args.input_dir),
        Path(_args.output),
        Path(_args.shard_dir) if _args.shard_dir else None,
    )
