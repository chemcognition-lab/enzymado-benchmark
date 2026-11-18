"""Safetensors writers shared by the embedding-generation scripts.

Per-residue embeddings are too large for a single safetensors file, so the
main output is a directory of numbered shards (globbed later by
src/data_loading/stores.py). Mean-pooled embeddings stay small enough to
buffer in memory and write once, mirroring
data/embeddings_mean_pooled/protein_sequences_unique/convert.py.

Both writers save to a temporary name and rename, so a job killed mid-write
never leaves a truncated .safetensors file that the loader would glob.
Several jobs can share one shard directory if each uses its own prefix.
"""

from __future__ import annotations

import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def _save_atomic(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Write a safetensors file under a temporary name, then rename it."""
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


class SafetensorsShardWriter:
    """Buffers bf16 residue tensors and flushes them to numbered .safetensors shards.

    Shards are named <prefix><index>.safetensors. Resume only reads shards
    with this writer's prefix, so parallel jobs with distinct prefixes never
    read each other's files while those are being written.
    """

    def __init__(self, shard_dir: Path, shard_size: int = 1000, prefix: str = "shard_"):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix

        existing_shards = sorted(self.shard_dir.glob(f"{prefix}[0-9]*.safetensors"))
        self._existing_keys: set[str] = set()
        for shard_path in existing_shards:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                self._existing_keys.update(handle.keys())
        self._next_index = (
            int(existing_shards[-1].stem.rsplit("_", 1)[-1]) + 1
            if existing_shards
            else 0
        )
        self._buffer: dict[str, torch.Tensor] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._existing_keys or key in self._buffer

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self._buffer[key] = tensor.detach().to(torch.bfloat16).contiguous().cpu()
        self._existing_keys.add(key)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        shard_path = self.shard_dir / f"{self.prefix}{self._next_index:05d}.safetensors"
        _save_atomic(self._buffer, shard_path)
        self._next_index += 1
        self._buffer = {}

    def __enter__(self) -> "SafetensorsShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()


class SafetensorsPooledWriter:
    """Buffers mean-pooled tensors in memory and writes one .safetensors file.

    dtype defaults to bf16, the format of the PLM pooled files. Resumed
    tensors are cast to dtype as well, so one file never mixes dtypes.
    """

    def __init__(
        self,
        path: Path,
        flush_every: int = 5000,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.path = Path(path)
        self.flush_every = flush_every
        self.dtype = dtype
        self._tensors: dict[str, torch.Tensor] = {}
        self._since_flush = 0
        if self.path.exists():
            with safe_open(str(self.path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self._tensors[key] = handle.get_tensor(key).to(dtype)

    def __contains__(self, key: str) -> bool:
        return key in self._tensors

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self._tensors[key] = tensor.detach().to(self.dtype).contiguous().cpu()
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._tensors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _save_atomic(self._tensors, self.path)
        self._since_flush = 0

    def __enter__(self) -> "SafetensorsPooledWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
