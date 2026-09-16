"""Fingerprint a residue-embedding shard directory by content, not by layout.

``run_manifest.json`` records a ``tree_inventory_sha256`` built from file names,
sizes and mtimes, so the same embeddings repacked into a different number of
shards, or copied to another machine, no longer match. This script hashes the
sequence keys and a few tensors instead, which survives both. Run it on two
machines and compare: identical key and probe hashes mean the stores hold the
same embeddings however they are packed.

    python scripts/embeddings/fingerprint_embedding_store.py <model>_shards
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import safetensors
import torch

PROBE_COUNT = 3


def _shard_keys(shards: list[Path]) -> tuple[list[str], str | None]:
    """Collect every sequence key across the shards, plus the stored dtype."""
    keys: list[str] = []
    dtype: str | None = None
    for shard in shards:
        with safetensors.safe_open(shard, framework="pt") as handle:
            names = list(handle.keys())
            keys.extend(names)
            if dtype is None and names:
                dtype = str(handle.get_slice(names[0]).get_dtype())
    return sorted(keys), dtype


def _probe_digest(shards: list[Path], key: str) -> tuple[torch.Size, str] | None:
    """Hash the raw bytes of one sequence's embedding, whichever shard holds it.

    ``view(torch.uint8)`` is what makes this work for BF16, which numpy cannot
    represent and therefore cannot serialize directly.
    """
    for shard in shards:
        with safetensors.safe_open(shard, framework="pt") as handle:
            if key not in handle.keys():
                continue
            tensor = handle.get_tensor(key)
            raw = tensor.view(torch.uint8).numpy().tobytes()
            return tensor.shape, hashlib.sha256(raw).hexdigest()[:32]
    return None


def main() -> None:
    """Print one content fingerprint for the given shard directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{args.shard_dir} holds no .safetensors shards.")

    keys, dtype = _shard_keys(shards)
    total_bytes = sum(shard.stat().st_size for shard in shards)
    key_digest = hashlib.sha256("\0".join(keys).encode()).hexdigest()

    print(f"dir          : {args.shard_dir}")
    print(f"shards       : {len(shards)}")
    print(f"total bytes  : {total_bytes:,}")
    print(f"sequences    : {len(keys):,} ({len(set(keys)):,} unique)")
    print(f"dtype        : {dtype}")
    print(f"sha256(keys) : {key_digest}")

    # Fixed positions in the sorted key list, so both machines probe the same
    # sequences without needing to agree on a name up front.
    positions = [0, len(keys) // 2, len(keys) - 1][:PROBE_COUNT]
    for position in positions:
        key = keys[position]
        probe = _probe_digest(shards, key)
        if probe is None:
            print(f"  {key[:40]:<42} MISSING")
            continue
        shape, digest = probe
        print(f"  {key[:40]:<42} shape={str(tuple(shape)):<16} sha256={digest}")


if __name__ == "__main__":
    main()
