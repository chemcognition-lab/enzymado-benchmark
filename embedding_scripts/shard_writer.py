"""Safetensors writers shared by the embedding-generation scripts.

Per-residue embeddings are too large for a single safetensors file, so the
main output is a directory of numbered shards (globbed later by
src/data_loading/stores.py). Mean-pooled embeddings stay small enough to
buffer in memory and write once, mirroring
data/embeddings_mean_pooled/protein_sequences_unique/convert.py.
"""

from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


class SafetensorsShardWriter:
    """Buffers bf16 residue tensors and flushes them to numbered .safetensors shards."""

    def __init__(self, shard_dir: Path, shard_size: int = 1000):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size

        existing_shards = sorted(self.shard_dir.glob("shard_*.safetensors"))
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
        shard_path = self.shard_dir / f"shard_{self._next_index:05d}.safetensors"
        save_file(self._buffer, str(shard_path))
        self._next_index += 1
        self._buffer = {}

    def __enter__(self) -> "SafetensorsShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()


class SafetensorsPooledWriter:
    """Buffers mean-pooled bf16 tensors in memory and writes one .safetensors file."""

    def __init__(self, path: Path, flush_every: int = 5000):
        self.path = Path(path)
        self.flush_every = flush_every
        self._tensors: dict[str, torch.Tensor] = {}
        self._since_flush = 0
        if self.path.exists():
            with safe_open(str(self.path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self._tensors[key] = handle.get_tensor(key)

    def __contains__(self, key: str) -> bool:
        return key in self._tensors

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self._tensors[key] = tensor.detach().to(torch.bfloat16).contiguous().cpu()
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._tensors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self._tensors, str(self.path))
        self._since_flush = 0

    def __enter__(self) -> "SafetensorsPooledWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
