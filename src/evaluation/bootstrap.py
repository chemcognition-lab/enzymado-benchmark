"""Compute paired bootstrap metrics for a single experiment.

The same `(n_replicates, n_units)` index matrix is reused across every experiment of
the same task so embedding models are scored on identical resamples; the caller is
responsible for generating and supplying that matrix.

Threshold classification metrics reduce to per-row TP/FP/FN/TN counts, while
regression MAE/RMSE reduce to per-row error sums. Ranking metrics AUROC/AUPRC
retain continuous scores and are recomputed per replicate. Additive values are
evaluated for all replicates in one vectorized NumPy call via
`stats[indices].sum(axis=1)`. r2 and kendall_tau cannot be reduced this way:
r2's sum-of-squares-total depends on the resample's own mean (not a fixed
global value), and kendall_tau depends on the rank order of resampled rows.
The caller passes `task_type` explicitly; it is never inferred from the numeric
range of the data (a normalized regression target living in [0, 1] would
be indistinguishable from binary labels by value alone).

Example:
    Call ``bootstrap_replicates(payload, indices, metric_set, task_type)``.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import binary_statistics, loaders, metrics_registry


def n_resample_units(payload: loaders.PredictionPayload) -> int:
    """Return resampling units: proteins for residue tasks, rows otherwise."""
    return int(payload.y_true.shape[0])


def make_resample_indices(
    n_units: int,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Deterministic (n_replicates, n_units) integer index matrix."""
    rng = np.random.default_rng(seed)
    return rng.integers(
        low=0,
        high=n_units,
        size=(n_replicates, n_units),
        dtype=np.int64,
    )


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


def _precompute_binary_counts(
    payload: loaders.PredictionPayload,
) -> np.ndarray:
    """Aggregate per-row TP, FP, FN, and TN counts."""
    return binary_statistics.count_payload(payload)


def _bootstrap_classification_counts(
    counts: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute classification metrics from pre-aggregated counts."""
    return binary_statistics.bootstrap(counts, indices)


_COUNT_CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}
_ADDITIVE_REGRESSION_METRICS = {"mae", "rmse"}


def _precompute_regression_errors(
    payload: loaders.PredictionPayload,
) -> np.ndarray:
    """Aggregate row-level valid counts and absolute/squared error sums.

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


def _bootstrap_per_replicate(
    payload: loaders.PredictionPayload,
    indices: np.ndarray,
    metric_set: dict[str, metrics_registry.MetricFn],
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


def bootstrap_replicates(
    payload: loaders.PredictionPayload,
    indices: np.ndarray,
    metric_set: dict[str, metrics_registry.MetricFn],
    task_type: str,
) -> dict[str, np.ndarray]:
    """Compute bootstrap replicates for every metric in metric_set.

    task_type selects the path — it is the caller's already-known task type
    (e.g. from hparams.json), never guessed from the data itself.
    """
    out: dict[str, np.ndarray] = {}
    if task_type == "classification":
        additive_names = _COUNT_CLASSIFICATION_METRICS & metric_set.keys()
        if additive_names:
            counts = _precompute_binary_counts(payload)
            additive = _bootstrap_classification_counts(counts, indices)
            out.update(
                {
                    name: values
                    for name, values in additive.items()
                    if name in additive_names
                }
            )
        remaining = {
            name: function
            for name, function in metric_set.items()
            if name not in additive_names
        }
        if remaining:
            out.update(_bootstrap_per_replicate(payload, indices, remaining))
        return out

    additive_names = _ADDITIVE_REGRESSION_METRICS & metric_set.keys()
    if additive_names:
        stats = _precompute_regression_errors(payload)
        out.update(_bootstrap_additive_regression(stats, indices, additive_names))

    remaining = {
        name: function
        for name, function in metric_set.items()
        if name not in additive_names
    }
    if remaining:
        out.update(_bootstrap_per_replicate(payload, indices, remaining))
    return out
