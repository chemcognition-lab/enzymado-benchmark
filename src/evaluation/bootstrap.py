"""Paired bootstrap of test-set metrics for a single experiment.

The same `(n_replicates, n_units)` index matrix is reused across every experiment of
the same task so embedding models are scored on identical resamples; the caller is
responsible for generating and supplying that matrix.

For binary classification tasks (solubility, binding, active_sites, binding_site,
ec_number, etc.) the fast path is used automatically: TP/FP/FN/TN counts are
pre-aggregated once per payload, making each replicate O(N) integer arithmetic
instead of a full sklearn call. The result is mathematically identical to the
generic path but ~100x faster on large 2D arrays.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.loaders import PredictionPayload
from src.evaluation.metrics_registry import MetricFn, metric_set_for


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def n_resample_units(payload: PredictionPayload) -> int:
    """Number of units that bootstrap resamples (proteins for per-residue, rows otherwise)."""
    return int(payload.y_true.shape[0])


def make_resample_indices(
    n_units: int,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Deterministic (n_replicates, n_units) integer index matrix."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=n_units, size=(n_replicates, n_units), dtype=np.int64)


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


# ---------------------------------------------------------------------------
# Fast path — binary classification (solubility, binding, active_sites, etc.)
# ---------------------------------------------------------------------------

def _is_binary_classification(payload: PredictionPayload) -> bool:
    """Return True when labels are binary {0,1} and the fast path can be used."""
    y = payload.y_true
    p = payload.predictions
    if y.size == 0 or p.size == 0:
        return False
    if np.issubdtype(y.dtype, np.floating):
        y_max, y_min = float(np.nanmax(y)), float(np.nanmin(y))
    else:
        y_max, y_min = float(y.max()), float(y.min())
    if np.issubdtype(p.dtype, np.floating):
        p_max, p_min = float(np.nanmax(p)), float(np.nanmin(p))
    else:
        p_max, p_min = float(p.max()), float(p.min())
    if not all(np.isfinite(v) for v in (y_max, y_min, p_max, p_min)):
        return False
    return y_max <= 1.5 and p_max <= 1.5 and y_min >= -0.5 and p_min >= -0.5


def _precompute_binary_counts(payload: PredictionPayload) -> np.ndarray:
    """Aggregate per-row [TP, FP, FN, TN] counts — computed once before the loop.

    Returns int64 array of shape (n_units, 4). Each bootstrap replicate sums
    selected rows to get global totals in O(N) with no sklearn calls.
    """
    y = payload.y_true
    p = payload.predictions
    m = payload.target_mask

    valid = np.asarray(m, dtype=bool) if m is not None else np.ones(y.shape, dtype=bool)

    with np.errstate(invalid="ignore"):
        y_int = np.rint(y).astype(np.int8, copy=False)
        p_int = np.rint(p).astype(np.int8, copy=False)

    tp_mask = (y_int == 1) & (p_int == 1) & valid
    fp_mask = (y_int == 0) & (p_int == 1) & valid
    fn_mask = (y_int == 1) & (p_int == 0) & valid
    tn_mask = (y_int == 0) & (p_int == 0) & valid

    if y_int.ndim == 1:
        tp = tp_mask.astype(np.int64)
        fp = fp_mask.astype(np.int64)
        fn = fn_mask.astype(np.int64)
        tn = tn_mask.astype(np.int64)
    else:
        tp = tp_mask.sum(axis=1, dtype=np.int64)
        fp = fp_mask.sum(axis=1, dtype=np.int64)
        fn = fn_mask.sum(axis=1, dtype=np.int64)
        tn = tn_mask.sum(axis=1, dtype=np.int64)

    return np.stack([tp, fp, fn, tn], axis=1)


def _bootstrap_binary_fast(
    counts: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute classification metrics per replicate from pre-aggregated counts."""
    totals = counts[indices].sum(axis=1, dtype=np.int64)
    tp = totals[:, 0].astype(np.float64)
    fp = totals[:, 1].astype(np.float64)
    fn = totals[:, 2].astype(np.float64)
    tn = totals[:, 3].astype(np.float64)

    pred_pos = tp + fp
    actual_pos = tp + fn
    total = tp + fp + fn + tn

    precision = np.where(pred_pos > 0, tp / np.where(pred_pos > 0, pred_pos, 1.0), 0.0)
    recall = np.where(actual_pos > 0, tp / np.where(actual_pos > 0, actual_pos, 1.0), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    accuracy = np.where(total > 0, (tp + tn) / np.where(total > 0, total, 1.0), 0.0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Generic path — regression and multi-class classification
# ---------------------------------------------------------------------------

def _bootstrap_generic(
    payload: PredictionPayload,
    indices: np.ndarray,
    metric_set: dict[str, MetricFn],
) -> dict[str, np.ndarray]:
    """Evaluate every metric function per replicate via row slicing."""
    n_rep, n_units = indices.shape
    if n_resample_units(payload) != n_units:
        raise ValueError(
            f"Index matrix expects n_units={n_units} but predictions have "
            f"{n_resample_units(payload)} rows."
        )
    out = {name: np.full(n_rep, np.nan, dtype=np.float64) for name in metric_set}
    for r in range(n_rep):
        idx = indices[r]
        y = payload.y_true[idx]
        p = payload.predictions[idx]
        m = payload.target_mask[idx] if payload.target_mask is not None else None
        for name, fn in metric_set.items():
            out[name][r] = fn(y, p, m)
    return out


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def bootstrap_replicates(
    payload: PredictionPayload,
    indices: np.ndarray,
    metric_set: dict[str, MetricFn],
) -> dict[str, np.ndarray]:
    """Compute bootstrap replicates, auto-selecting fast or generic path.

    Binary classification payloads (labels in {0,1}) use pre-aggregated
    TP/FP/FN/TN counts for ~100x speedup. All other tasks use the generic path.
    Returns one float64 array of length n_replicates per metric.
    """
    if _is_binary_classification(payload):
        counts = _precompute_binary_counts(payload)
        return _bootstrap_binary_fast(counts, indices)
    return _bootstrap_generic(payload, indices, metric_set)
