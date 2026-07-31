"""Persist reusable evaluation sufficient statistics.

Example:
    ``save_counts(path, counts, prediction_path)`` writes a validated cache
    consumed later by the bootstrap pipeline.
"""

from __future__ import annotations

import hashlib
import pathlib
import tempfile

import numpy as np
import numpy.typing as npt

from src.evaluation import binary_statistics, loaders


def path_for(
    statistics_root: pathlib.Path,
    predictions_root: pathlib.Path,
    run_dir: pathlib.Path,
    split_name: str,
) -> pathlib.Path:
    """Return the mirrored cache path for one prediction split."""
    relative_run = run_dir.relative_to(predictions_root)
    return statistics_root / relative_run / f"{split_name}_binary_counts.npz"


def _sha256(path: pathlib.Path) -> str:
    """Hash one prediction artifact for cache validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_counts(
    cache_path: pathlib.Path,
    counts: npt.NDArray[np.int64],
    prediction_path: pathlib.Path,
) -> None:
    """Write counts and source provenance to a compressed NumPy archive."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent,
        prefix=f".{cache_path.stem}-",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary_path = pathlib.Path(handle.name)
        np.savez_compressed(
            handle,
            counts=counts.astype(np.int64, copy=False),
            source_name=np.asarray(prediction_path.name),
            source_sha256=np.asarray(_sha256(prediction_path)),
        )
    temporary_path.replace(cache_path)


def load_counts(
    cache_path: pathlib.Path,
    prediction_path: pathlib.Path,
) -> npt.NDArray[np.int64] | None:
    """Load counts only when they match the current prediction artifact."""
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as archive:
        source_name = str(archive["source_name"].item())
        source_sha256 = str(archive["source_sha256"].item())
        if source_name != prediction_path.name:
            return None
        if source_sha256 != _sha256(prediction_path):
            return None
        counts = archive["counts"].astype(np.int64, copy=False)

    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError(f"Invalid cached count shape {counts.shape} in {cache_path}.")
    return counts


def load_or_create_counts(
    statistics_root: pathlib.Path,
    predictions_root: pathlib.Path,
    run_dir: pathlib.Path,
    split_name: str,
) -> binary_statistics.CountArray | None:
    """Load valid counts or derive them from prediction blocks once."""
    prediction_path = loaders.find_split_prediction_path(run_dir, split_name)
    if prediction_path is None:
        return None

    cache_path = path_for(
        statistics_root,
        predictions_root,
        run_dir,
        split_name,
    )
    counts = load_counts(cache_path, prediction_path)
    if counts is not None:
        return counts

    blocks = loaders.load_split_blocks(run_dir, split_name)
    if blocks is None:
        return None
    counts = binary_statistics.count_blocks(blocks)
    save_counts(cache_path, counts, prediction_path)
    return counts
