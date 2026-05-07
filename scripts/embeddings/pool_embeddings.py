"""Pre-compute mean-pooled embeddings from per-residue H5 files.

Reads each (seq_len, dim) entry, mean-pools to (dim,), and writes to a new H5.
This shrinks files from ~hundreds of GB to ~1-4 GB per model.

Usage:
    python pool_embeddings.py <input.h5> <output.h5>
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pool_h5(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        log(f"Output already exists at {output_path}, skipping.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Input:  {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)")

    with h5py.File(str(input_path), "r") as src:
        keys = list(src.keys())
        n_total = len(keys)
        log(f"Sequences: {n_total:,}")

        # Infer dim from first entry
        first = src[keys[0]][:]
        if first.ndim == 1:
            dim = first.shape[0]
        else:
            dim = first.shape[1]
        log(f"Embedding dim: {dim}")

        log_interval = max(1, n_total // 50)
        t_start = time.time()

        n_dupes = 0
        n_missing = 0
        with h5py.File(str(output_path), "w") as dst:
            for i, key in enumerate(keys, 1):
                if key in dst:
                    n_dupes += 1
                    continue
                try:
                    arr = src[key][:]
                except KeyError:
                    n_missing += 1
                    log(f"  WARNING: key '{key}' listed but not readable, skipping.")
                    continue
                if arr.ndim == 2:
                    pooled = arr.mean(axis=0).astype(np.float32)
                else:
                    pooled = arr.astype(np.float32)  # already pooled
                dst.create_dataset(key, data=pooled)

                if i % log_interval == 0 or i == n_total:
                    elapsed = time.time() - t_start
                    rate = i / elapsed
                    eta = (n_total - i) / rate if rate > 0 else 0
                    log(f"  {i:,}/{n_total:,} ({100*i/n_total:.0f}%)  "
                        f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    if n_dupes:
        log(f"Skipped {n_dupes:,} duplicate keys in source H5.")
    if n_missing:
        log(f"Skipped {n_missing:,} unreadable keys (corrupt entries) in source H5.")
    out_size = output_path.stat().st_size
    log(f"Done. Output: {output_path} ({out_size / 1e9:.2f} GB)")
    log(f"Compression ratio: {input_path.stat().st_size / out_size:.0f}x")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.h5> <output.h5>")
        sys.exit(1)

    pool_h5(Path(sys.argv[1]), Path(sys.argv[2]))
