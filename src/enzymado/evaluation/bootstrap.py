"""Compute paired bootstrap metrics using their registered strategies."""

from __future__ import annotations

import numpy as np

from enzymado.evaluation import binary_statistics, loaders, metrics_registry
import enzymado.types


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
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute classification metrics from pre-aggregated counts."""
    return metrics_registry.bootstrap(counts, indices, specs)


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
    specs: tuple[enzymado.types.MetricSpec, ...],
) -> dict[str, np.ndarray]:
    """Compute registered metrics from aggregated row-level error sums."""
    totals = stats[indices].sum(axis=1)
    return metrics_registry.aggregate_metrics(totals, specs)


def _bootstrap_per_replicate(
    payload: loaders.PredictionPayload,
    indices: np.ndarray,
    specs: tuple[enzymado.types.MetricSpec, ...],
) -> dict[str, np.ndarray]:
    """Evaluate every metric function per replicate via row slicing."""
    n_rep, n_units = indices.shape
    if n_resample_units(payload) != n_units:
        raise ValueError(
            f"Index matrix expects n_units={n_units} but predictions have "
            f"{n_resample_units(payload)} rows."
        )
    out = {spec.name: np.full(n_rep, np.nan, dtype=np.float64) for spec in specs}
    for r in range(n_rep):
        idx = indices[r]
        y = payload.y_true[idx]
        p = payload.predictions[idx]
        m = payload.target_mask[idx] if payload.target_mask is not None else None
        for spec in specs:
            out[spec.name][r] = spec.fn(y, p, m)
    return out


def bootstrap_replicates(
    payload: loaders.PredictionPayload,
    indices: np.ndarray,
    specs: tuple[enzymado.types.MetricSpec, ...],
) -> dict[str, np.ndarray]:
    """Compute every requested metric using its declared statistic strategy."""
    out: dict[str, np.ndarray] = {}
    confusion_specs = tuple(
        spec
        for spec in specs
        if spec.statistic is enzymado.types.MetricStatistic.CONFUSION
    )
    if confusion_specs:
        counts = _precompute_binary_counts(payload)
        out.update(_bootstrap_classification_counts(counts, indices, confusion_specs))
        score_specs = tuple(
            spec
            for spec in confusion_specs
            if spec.input_kind
            is enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS
        )
        if payload.proba is not None and score_specs:
            histograms = binary_statistics.ranking_histograms(payload)
            out.update(
                metrics_registry.bootstrap_ranking(
                    histograms,
                    indices,
                    specs=score_specs,
                )
            )

    error_specs = tuple(
        spec
        for spec in specs
        if spec.statistic is enzymado.types.MetricStatistic.ERROR_SUMS
    )
    if error_specs:
        stats = _precompute_regression_errors(payload)
        out.update(_bootstrap_additive_regression(stats, indices, error_specs))

    raw_specs = tuple(
        spec for spec in specs if spec.statistic is enzymado.types.MetricStatistic.RAW
    )
    if raw_specs:
        out.update(_bootstrap_per_replicate(payload, indices, raw_specs))
    return out
