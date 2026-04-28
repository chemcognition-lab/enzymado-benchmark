"""Paired bootstrap of test-set metrics for a single experiment.

The same `(n_replicates, n_units)` index matrix is reused across every experiment of
the same task so embedding models are scored on identical resamples; the caller is
responsible for generating and supplying that matrix.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.loaders import PredictionPayload
from src.evaluation.metrics import MetricFn


def n_resample_units(payload: PredictionPayload) -> int:
    """Number of units that bootstrap resamples (proteins for per-residue, rows otherwise)."""
    return int(payload.y_true.shape[0])


def _row_slice(payload: PredictionPayload, idx: np.ndarray):
    y = payload.y_true[idx]
    p = payload.predictions[idx]
    m = payload.target_mask[idx] if payload.target_mask is not None else None
    return y, p, m


def bootstrap_replicates(
    payload: PredictionPayload,
    indices: np.ndarray,
    metric_set: dict[str, MetricFn],
) -> dict[str, np.ndarray]:
    """Apply the supplied resample indices and evaluate every metric per replicate.

    `indices` has shape (n_replicates, n_units). Returns one float64 array per metric.
    """
    n_rep, n_units = indices.shape
    if n_resample_units(payload) != n_units:
        raise ValueError(
            f"Index matrix expects n_units={n_units} but predictions have "
            f"{n_resample_units(payload)} rows."
        )

    out = {name: np.full(n_rep, np.nan, dtype=np.float64) for name in metric_set}
    for r in range(n_rep):
        y, p, m = _row_slice(payload, indices[r])
        for name, fn in metric_set.items():
            out[name][r] = fn(y, p, m)
    return out


def summarise_replicates(
    replicates: dict[str, np.ndarray],
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Mean / std / percentile CI per metric, ignoring NaN replicates."""
    alpha = 1.0 - confidence_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    summary: dict[str, dict[str, float]] = {}
    for name, arr in replicates.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            summary[name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_finite": 0,
            }
            continue
        summary[name] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=0)),
            "ci_lo": float(np.percentile(finite, lo_q)),
            "ci_hi": float(np.percentile(finite, hi_q)),
            "n_finite": int(finite.size),
        }
    return summary


def make_resample_indices(
    n_units: int,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Deterministic (n_replicates, n_units) integer index matrix."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=n_units, size=(n_replicates, n_units), dtype=np.int64)
