"""Paired bootstrap of test-set metrics for a single experiment.

The same `(n_replicates, n_units)` index matrix is reused across every experiment of
the same task so embedding models are scored on identical resamples; the caller is
responsible for generating and supplying that matrix.

Metrics that reduce to a sum of precomputed per-row values (classification's
accuracy/precision/recall/f1 via TP/FP/FN/TN counts; regression's mae/rmse via
per-row error sums) are evaluated for all replicates in one vectorized NumPy call
instead of a Python loop calling the metric function once per replicate — the
per-row values are summed once per replicate via `stats[indices].sum(axis=1)`,
then combined into the final metric. r2 and kendall_tau cannot be reduced this
way: r2's sum-of-squares-total depends on the resample's own mean (not a fixed
global value), and kendall_tau depends on the pairwise rank order of whichever
rows end up in a given resample — neither is a sum of independent per-row
contributions, so both are recomputed directly on each replicate's resampled
values. The caller passes `task_type` explicitly; it is never inferred from the
numeric range of the data (a normalized regression target living in [0, 1] would
be indistinguishable from binary labels by value alone).
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
# Additive path — classification (accuracy/precision/recall/f1 from TP/FP/FN/TN)
# ---------------------------------------------------------------------------

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


def _bootstrap_classification_counts(
    counts: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute accuracy/precision/recall/f1 per replicate from pre-aggregated counts."""
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
# Additive path — regression (mae/rmse from per-row error sums)
# ---------------------------------------------------------------------------

_ADDITIVE_REGRESSION_METRICS = {"mae", "rmse"}


def _precompute_regression_errors(payload: PredictionPayload) -> np.ndarray:
    """Aggregate per-row [n_valid, sum_abs_error, sum_sq_error] — computed once before the loop.

    mae/rmse are masked, flattened means (see metrics_registry._flatten_known):
    summing these three per-row totals across a resampled set of rows and dividing
    reproduces that same masked mean exactly, for any set of resampled rows.
    """
    y = np.asarray(payload.y_true, dtype=np.float64)
    p = np.asarray(payload.predictions, dtype=np.float64)
    m = payload.target_mask

    valid = np.asarray(m, dtype=bool) if m is not None else np.ones(y.shape, dtype=bool)
    valid &= np.isfinite(y) & np.isfinite(p)

    abs_err = np.where(valid, np.abs(p - y), 0.0)
    sq_err = np.where(valid, (p - y) ** 2, 0.0)

    if y.ndim == 1:
        n_valid = valid.astype(np.float64)
        sum_abs = abs_err
        sum_sq = sq_err
    else:
        axes = tuple(range(1, y.ndim))
        n_valid = valid.sum(axis=axes, dtype=np.float64)
        sum_abs = abs_err.sum(axis=axes)
        sum_sq = sq_err.sum(axis=axes)

    return np.stack([n_valid, sum_abs, sum_sq], axis=1)


def _bootstrap_additive_regression(
    stats: np.ndarray,
    indices: np.ndarray,
    metric_names: set[str],
) -> dict[str, np.ndarray]:
    """Compute mae/rmse per replicate from pre-aggregated per-row error sums."""
    totals = stats[indices].sum(axis=1)
    n_valid = totals[:, 0]
    sum_abs = totals[:, 1]
    sum_sq = totals[:, 2]
    safe_n = np.where(n_valid > 0, n_valid, 1.0)

    out: dict[str, np.ndarray] = {}
    if "mae" in metric_names:
        out["mae"] = np.where(n_valid > 0, sum_abs / safe_n, np.nan)
    if "rmse" in metric_names:
        out["rmse"] = np.where(n_valid > 0, np.sqrt(sum_sq / safe_n), np.nan)
    return out


# ---------------------------------------------------------------------------
# Per-replicate path — r2 and kendall_tau, recomputed on each replicate's
# actual resampled values (not reducible to summed per-row statistics)
# ---------------------------------------------------------------------------

def _bootstrap_per_replicate(
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
    task_type: str,
) -> dict[str, np.ndarray]:
    """Compute bootstrap replicates for every metric in metric_set.

    task_type selects the path — it is the caller's already-known task type
    (e.g. from hparams.json), never guessed from the data itself.
    """
    if task_type == "classification":
        counts = _precompute_binary_counts(payload)
        return _bootstrap_classification_counts(counts, indices)

    out: dict[str, np.ndarray] = {}
    additive_names = _ADDITIVE_REGRESSION_METRICS & metric_set.keys()
    if additive_names:
        stats = _precompute_regression_errors(payload)
        out.update(_bootstrap_additive_regression(stats, indices, additive_names))

    remaining = {name: fn for name, fn in metric_set.items() if name not in additive_names}
    if remaining:
        out.update(_bootstrap_per_replicate(payload, indices, remaining))
    return out
