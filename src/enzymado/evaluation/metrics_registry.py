"""Define and evaluate the benchmark's canonical metric catalogue."""

from __future__ import annotations

import numpy as np
import scipy.stats

import enzymado.types

type Array = enzymado.types.NDArray

RANKING_BINS = 16384


def metric_specs(task_type: str | None = None) -> tuple[enzymado.types.MetricSpec, ...]:
    """Return the canonical metric declarations, optionally filtered by task."""
    metrics = tuple(
        spec for spec in METRICS if task_type is None or spec.task_type == task_type
    )
    if not metrics:
        raise ValueError(f"No metrics registered for task_type '{task_type}'.")
    return metrics


def metric_specs_for_statistic(
    statistic: enzymado.types.MetricStatistic,
    task_type: str | None = None,
) -> tuple[enzymado.types.MetricSpec, ...]:
    """Return metrics backed by one sufficient-statistic representation."""
    return tuple(
        spec
        for spec in metric_specs(task_type)
        if spec.statistic is statistic
    )


def higher_is_better_for(task_type: str) -> dict[str, int]:
    """Return sign convention (+1 / -1) per metric for the given task type."""
    return {
        spec.name: (1 if spec.higher_is_better else -1)
        for spec in metric_specs(task_type)
    }


def rating_weights_for(task_type: str) -> dict[str, float]:
    """Return the unnormalized leaderboard weight for every metric."""
    return {spec.name: spec.rating_weight for spec in metric_specs(task_type)}


def primary_metric_for(task_type: str) -> str | None:
    """Return the name of the primary metric for the given task type."""
    for spec in metric_specs(task_type):
        if spec.is_primary:
            return spec.name
    return None


def _flatten_known(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Flatten and optionally mask arrays to 1D float64."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    if target_mask is not None:
        keep = np.asarray(target_mask, dtype=bool).ravel()
        if keep.shape == y.shape:
            y, p = y[keep], p[keep]
    finite = np.isfinite(y) & np.isfinite(p)
    return y[finite], p[finite]


def regression_mae(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    return float("nan") if y.size == 0 else float(np.mean(np.abs(p - y)))


def regression_rmse(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    return float("nan") if y.size == 0 else float(np.sqrt(np.mean((p - y) ** 2)))


def regression_r2(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2:
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0.0 else float(1.0 - np.sum((p - y) ** 2) / ss_tot)


def regression_kendall_tau(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    """Kendall tau-b, scored 0 when predictions or labels are all tied.

    A full tie carries no ordering, so it is scored as no skill instead of
    undefined; a bootstrap replicate of a near-constant model stays ratable.
    Model selection rejects constant predictions before calling this.
    """
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2:
        return float("nan")
    if np.unique(y).size < 2 or np.unique(p).size < 2:
        return 0.0
    tau = scipy.stats.kendalltau(y, p).statistic
    return float("nan") if tau is None or not np.isfinite(tau) else float(tau)


def _flatten_for_eval(
    y_true: object,
    pred: object,
    target_mask: object = None,
) -> tuple[Array, Array]:
    """Flatten potentially nested (EC, residue) payloads into 1D arrays."""
    if isinstance(y_true, dict):
        y_chunks, p_chunks = [], []
        for key, y_val in y_true.items():
            if not isinstance(pred, dict) or key not in pred:
                continue
            mask_val = target_mask.get(key) if isinstance(target_mask, dict) else None
            y_f, p_f = _flatten_for_eval(y_val, pred[key], mask_val)
            if y_f.size:
                y_chunks.append(y_f)
                p_chunks.append(p_f)
        if not y_chunks:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return np.concatenate(y_chunks), np.concatenate(p_chunks)

    y_np = np.asarray(y_true, dtype=np.float32)
    p_np = np.asarray(pred, dtype=np.float32)
    if y_np.shape != p_np.shape:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    known = np.ones(y_np.shape, dtype=bool)
    if target_mask is not None:
        mask_np = np.asarray(target_mask, dtype=bool)
        if mask_np.shape == y_np.shape:
            known &= mask_np
        elif mask_np.ndim == 1 and y_np.ndim > 1 and mask_np.shape[0] == y_np.shape[0]:
            reshape = (mask_np.shape[0],) + (1,) * (y_np.ndim - 1)
            known &= np.broadcast_to(mask_np.reshape(reshape), y_np.shape)

    y_f = y_np[known].ravel()
    p_f = p_np[known].ravel()
    finite = np.isfinite(y_f.astype(np.float64)) & np.isfinite(p_f.astype(np.float64))
    return y_f[finite], p_f[finite]


def evaluate_predictions(
    y_true: object,
    pred: object,
    task_type: str,
    target_mask: object = None,
    *,
    split: str,
    proba: object = None,
) -> dict[str, object]:
    """Recompute metrics for one split directly from raw arrays.

    Only metrics that declare ``fn`` are computed here. Classification metrics
    deliberately do not: they are derived from per-row confusion counts and
    score histograms (``evaluate`` and ``evaluate_ranking``), which is what
    makes bootstrapping the largest tasks tractable. Callers are
    ``compute_metrics`` and ``compute_bootstrap_metrics``, both on their
    regression branch; training does not call this.
    """
    y_flat, pred_flat = _flatten_for_eval(y_true, pred, target_mask)
    result: dict[str, object] = {
        "split": split,
        "task_type": task_type,
        "n": int(y_flat.size),
    }
    if y_flat.size == 0:
        return result
    score_flat = None
    if proba is not None:
        _, score_flat = _flatten_for_eval(y_true, proba, target_mask)
    result["ranking_source"] = "proba" if score_flat is not None else "hard_labels"
    for spec in metric_specs(task_type):
        if spec.fn is None:
            continue
        accepts_scores = (
            spec.input_kind
            is enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS
        )
        use_scores = score_flat is not None and accepts_scores
        values = score_flat if use_scores else pred_flat
        result[spec.name] = spec.fn(y_flat, values, None)
    return result


def _confusion_columns(
    totals: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return floating-point TP, FP, FN, and TN columns."""
    if totals.ndim != 2 or totals.shape[1] != 4:
        raise ValueError(f"Expected counts with shape (N, 4), got {totals.shape}.")
    return tuple(
        totals[:, column].astype(np.float64) for column in range(4)
    )


def _ratio(
    numerator: Array,
    denominator: Array,
    *,
    missing: float = 0.0,
) -> Array:
    """Divide arrays with the benchmark's zero-denominator convention."""
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, missing),
        where=denominator > 0,
    )


def _count_accuracy(totals: Array) -> Array:
    true_positive, false_positive, false_negative, true_negative = (
        _confusion_columns(totals)
    )
    total = true_positive + false_positive + false_negative + true_negative
    return _ratio(true_positive + true_negative, total)


def _count_precision(totals: Array) -> Array:
    true_positive, false_positive, _, _ = _confusion_columns(totals)
    return _ratio(true_positive, true_positive + false_positive)


def _count_recall(totals: Array) -> Array:
    true_positive, _, false_negative, _ = _confusion_columns(totals)
    return _ratio(true_positive, true_positive + false_negative)


def _count_f1(totals: Array) -> Array:
    precision = _count_precision(totals)
    recall = _count_recall(totals)
    denominator = precision + recall
    return _ratio(2.0 * precision * recall, denominator)


def _count_auroc(totals: Array) -> Array:
    true_positive, false_positive, false_negative, true_negative = (
        _confusion_columns(totals)
    )
    true_positive_rate = _ratio(
        true_positive,
        true_positive + false_negative,
        missing=np.nan,
    )
    true_negative_rate = _ratio(
        true_negative,
        false_positive + true_negative,
        missing=np.nan,
    )
    return 0.5 * (true_positive_rate + true_negative_rate)


def _count_auprc(totals: Array) -> Array:
    true_positive, false_positive, false_negative, true_negative = (
        _confusion_columns(totals)
    )
    precision = _count_precision(totals)
    recall = _count_recall(totals)
    total = true_positive + false_positive + false_negative + true_negative
    prevalence = _ratio(true_positive + false_negative, total)
    return precision * recall + prevalence * (1.0 - recall)


def _error_mae(totals: Array) -> Array:
    n_valid = totals[:, 0]
    safe_n = np.where(n_valid > 0, n_valid, 1.0)
    return np.where(n_valid > 0, totals[:, 1] / safe_n, np.nan)


def _error_rmse(totals: Array) -> Array:
    n_valid = totals[:, 0]
    safe_n = np.where(n_valid > 0, n_valid, 1.0)
    return np.where(n_valid > 0, np.sqrt(totals[:, 2] / safe_n), np.nan)


def aggregate_metrics(
    totals: Array,
    specs: tuple[enzymado.types.MetricSpec, ...],
    *,
    use_scores: bool = False,
) -> dict[str, Array]:
    """Apply each metric's declared reducer to aggregated statistics."""
    metrics: dict[str, Array] = {}
    for spec in specs:
        fn = spec.score_aggregate_fn if use_scores else spec.aggregate_fn
        if fn is None:
            raise ValueError(f"Metric '{spec.name}' has no aggregate function.")
        metrics[spec.name] = fn(totals)
    return metrics


def evaluate(
    counts: Array,
    *,
    split: str,
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, float | int | str]:
    """Calculate point metrics from per-row sufficient statistics."""
    specs = specs or metric_specs_for_statistic(
        enzymado.types.MetricStatistic.CONFUSION,
        "classification",
    )
    totals = counts.sum(axis=0, dtype=np.int64, keepdims=True)
    n_known = int(totals.sum())
    result: dict[str, float | int | str] = {
        "split": split,
        "task_type": "classification",
        "n": n_known,
    }
    if not n_known:
        return result
    metrics = aggregate_metrics(totals, specs)
    return result | {name: float(values[0]) for name, values in metrics.items()}


def bootstrap(
    counts: Array,
    indices: Array,
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, Array]:
    """Calculate bootstrap metrics without materializing a 3D count tensor."""
    specs = specs or metric_specs_for_statistic(
        enzymado.types.MetricStatistic.CONFUSION,
        "classification",
    )
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D bootstrap indices, got {indices.shape}.")
    if not counts.size or indices.shape[1] != counts.shape[0]:
        raise ValueError(
            f"Indices expect {indices.shape[1]} rows but counts have {counts.shape[0]}."
        )

    totals = np.empty((indices.shape[0], 4), dtype=np.int64)
    for column_index in range(4):
        totals[:, column_index] = counts[:, column_index][indices].sum(axis=1)
    return aggregate_metrics(totals, specs)


def _auroc_from_totals(negative: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Mann-Whitney U with mid-rank tie handling, per replicate."""
    n_negative = negative.sum(axis=1)
    n_positive = positive.sum(axis=1)
    below = np.cumsum(negative, axis=1) - negative
    wins = (positive * below).sum(axis=1) + 0.5 * (positive * negative).sum(axis=1)
    denominator = n_positive * n_negative
    return np.divide(
        wins,
        denominator,
        out=np.full(wins.shape, np.nan),
        where=denominator > 0,
    )


def _auprc_from_totals(negative: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Average precision accumulated from the highest score bin downwards."""
    positive_desc = positive[:, ::-1]
    negative_desc = negative[:, ::-1]
    true_positive = np.cumsum(positive_desc, axis=1)
    predicted = true_positive + np.cumsum(negative_desc, axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    n_positive = positive.sum(axis=1)
    weighted = (precision * positive_desc).sum(axis=1)
    return np.divide(
        weighted,
        n_positive,
        out=np.zeros_like(weighted),
        where=n_positive > 0,
    )


def _histogram_auroc(totals: Array) -> Array:
    n_bins = totals.shape[1] // 2
    return _auroc_from_totals(totals[:, :n_bins], totals[:, n_bins:])


def _histogram_auprc(totals: Array) -> Array:
    n_bins = totals.shape[1] // 2
    return _auprc_from_totals(totals[:, :n_bins], totals[:, n_bins:])


def auprc_from_scores(
    y_true: Array,
    scores: Array,
    target_mask: Array | None = None,
) -> float:
    """Average precision from raw probabilities, via the benchmark's estimator.

    Checkpoint selection has to score a model the same way the leaderboard
    will, so this reuses the histogram AUPRC rather than a second definition.
    Invalid probabilities and splits without both classes fail explicitly.
    """
    scores_flat = np.asarray(scores, dtype=np.float64).ravel()
    labels_flat = np.asarray(y_true, dtype=np.float64).ravel()
    if target_mask is not None:
        keep = np.asarray(target_mask, dtype=bool).ravel()
        scores_flat, labels_flat = scores_flat[keep], labels_flat[keep]
    if not np.isfinite(scores_flat).all() or not np.isfinite(labels_flat).all():
        raise FloatingPointError("Non-finite validation labels or probabilities.")
    if scores_flat.size == 0 or np.unique(labels_flat).size < 2:
        raise ValueError("AUPRC selection requires both validation classes.")
    if not np.isin(labels_flat, [0, 1]).all():
        raise ValueError("AUPRC selection requires binary labels.")
    if np.any((scores_flat < 0) | (scores_flat > 1)):
        raise ValueError("AUPRC probabilities must lie in [0, 1].")

    levels = np.clip(
        np.rint(scores_flat * (RANKING_BINS - 1)).astype(np.int64),
        0,
        RANKING_BINS - 1,
    )
    totals = np.zeros((1, 2 * RANKING_BINS), dtype=np.int64)
    offsets = (labels_flat == 1).astype(np.int64) * RANKING_BINS + levels
    np.add.at(totals, (np.zeros(offsets.size, dtype=np.int64), offsets), 1)
    return auprc_from_histogram(totals[0])


def auprc_from_histogram(totals: Array) -> float:
    """AUPRC of one score histogram: RANKING_BINS negative then positive counts."""
    totals = np.asarray(totals, dtype=np.float64).reshape(1, -1)
    if totals.shape[1] != 2 * RANKING_BINS:
        raise ValueError(
            f"Expected {2 * RANKING_BINS} histogram bins, got {totals.shape[1]}."
        )
    if totals[0, :RANKING_BINS].sum() == 0 or totals[0, RANKING_BINS:].sum() == 0:
        raise ValueError("AUPRC selection requires both validation classes.")
    return float(_histogram_auprc(totals)[0])


def _score_specs() -> tuple[enzymado.types.MetricSpec, ...]:
    """Return metrics that can consume probability histograms."""
    return tuple(
        spec
        for spec in metric_specs("classification")
        if spec.input_kind
        is enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS
    )


def ranking_metrics(
    totals: Array,
    *,
    n_bins: int = RANKING_BINS,
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, Array]:
    """Derive registered score metrics from aggregated histograms."""
    totals = np.asarray(totals, dtype=np.float64)
    if totals.ndim != 2 or totals.shape[1] != 2 * n_bins:
        raise ValueError(
            f"Expected histograms with shape (N, {2 * n_bins}), got "
            f"{totals.shape}."
        )
    return aggregate_metrics(totals, specs or _score_specs(), use_scores=True)


def bootstrap_ranking(
    histograms: Array,
    indices: Array,
    *,
    n_bins: int = RANKING_BINS,
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, Array]:
    """Resample row histograms and derive ranking metrics for every replicate."""
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D bootstrap indices, got {indices.shape}.")
    if indices.shape[1] != histograms.shape[0]:
        raise ValueError(
            f"Indices expect {indices.shape[1]} rows but histograms have "
            f"{histograms.shape[0]}."
        )
    n_rows = histograms.shape[0]
    counts = np.zeros((indices.shape[0], n_rows), dtype=np.float64)
    for replicate, row_indices in enumerate(indices):
        counts[replicate] = np.bincount(row_indices, minlength=n_rows)
    # One matmul replaces per-replicate gathering: resampling is a weighted sum
    # of additive row histograms.
    totals = counts @ histograms.astype(np.float64)
    return ranking_metrics(totals, n_bins=n_bins, specs=specs)


def evaluate_ranking(
    histograms: Array,
    *,
    n_bins: int = RANKING_BINS,
    specs: tuple[enzymado.types.MetricSpec, ...] | None = None,
) -> dict[str, float]:
    """Return registered score metrics from aggregated row histograms."""
    totals = histograms.sum(axis=0, dtype=np.int64, keepdims=True)
    metrics = ranking_metrics(totals, n_bins=n_bins, specs=specs)
    return {name: float(values[0]) for name, values in metrics.items()}


METRICS = (
    enzymado.types.MetricSpec(
        name="mae",
        task_type="regression",
        fn=regression_mae,
        higher_is_better=False,
        statistic=enzymado.types.MetricStatistic.ERROR_SUMS,
        aggregate_fn=_error_mae,
        is_primary=True,
    ),
    enzymado.types.MetricSpec(
        name="rmse",
        task_type="regression",
        fn=regression_rmse,
        higher_is_better=False,
        statistic=enzymado.types.MetricStatistic.ERROR_SUMS,
        aggregate_fn=_error_rmse,
        # Not rated, for the same reason as r2 below: it orders models exactly as
        # r2 does, and MAE already carries the size of the error.
        rating_weight=0.0,
    ),
    enzymado.types.MetricSpec(
        name="r2",
        task_type="regression",
        fn=regression_r2,
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.RAW,
        # Reported per model but given no leaderboard weight: within a bootstrap
        # replicate SS_tot is fixed, so r2 is a strictly monotone function of
        # rmse and casts the same pairwise battles. Neither is rated; MAE scores
        # the error and Kendall's tau the ordering.
        rating_weight=0.0,
    ),
    enzymado.types.MetricSpec(
        name="kendall_tau",
        task_type="regression",
        fn=regression_kendall_tau,
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.RAW,
        rating_weight=3.0,
    ),
    enzymado.types.MetricSpec(
        name="accuracy",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        aggregate_fn=_count_accuracy,
        # Reported but not rated: on rare labels the all-negative classifier
        # scores near 1, and on balanced ones precision, recall and F1 already
        # judge the same threshold.
        rating_weight=0.0,
    ),
    enzymado.types.MetricSpec(
        name="precision",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        aggregate_fn=_count_precision,
    ),
    enzymado.types.MetricSpec(
        name="recall",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        aggregate_fn=_count_recall,
    ),
    enzymado.types.MetricSpec(
        name="f1",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        aggregate_fn=_count_f1,
        is_primary=True,
    ),
    enzymado.types.MetricSpec(
        name="auroc",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        input_kind=enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS,
        aggregate_fn=_count_auroc,
        score_aggregate_fn=_histogram_auroc,
    ),
    enzymado.types.MetricSpec(
        name="auprc",
        task_type="classification",
        higher_is_better=True,
        statistic=enzymado.types.MetricStatistic.CONFUSION,
        input_kind=enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS,
        aggregate_fn=_count_auprc,
        score_aggregate_fn=_histogram_auprc,
    ),
)


def _validate_catalog() -> None:
    """Validate invariants that keep all metric consumers synchronized."""
    keys = [(spec.task_type, spec.name) for spec in METRICS]
    if len(keys) != len(set(keys)):
        raise ValueError("Metric names must be unique within each task type.")
    for task_type in {spec.task_type for spec in METRICS}:
        primary = [
            spec
            for spec in METRICS
            if spec.task_type == task_type and spec.is_primary
        ]
        if len(primary) != 1:
            raise ValueError(f"Task type '{task_type}' must have one primary metric.")
    for spec in METRICS:
        needs_aggregate = spec.statistic is not enzymado.types.MetricStatistic.RAW
        if needs_aggregate and spec.aggregate_fn is None:
            raise ValueError(f"Metric '{spec.name}' requires an aggregate function.")
        accepts_scores = (
            spec.input_kind
            is enzymado.types.MetricInput.PROBABILITIES_OR_PREDICTIONS
        )
        if accepts_scores and spec.score_aggregate_fn is None:
            raise ValueError(f"Metric '{spec.name}' requires a score reducer.")
        # The RAW bootstrap path recomputes the whole function per replicate,
        # so those metrics are the ones that genuinely need `fn`. Checking it
        # here keeps the declaration honest: a metric may only advertise an
        # implementation that production actually calls.
        if spec.statistic is enzymado.types.MetricStatistic.RAW and spec.fn is None:
            raise ValueError(f"Metric '{spec.name}' is RAW and requires a function.")


_validate_catalog()


__all__ = [
    "METRICS",
    "RANKING_BINS",
    "auprc_from_scores",
    "aggregate_metrics",
    "bootstrap",
    "bootstrap_ranking",
    "evaluate",
    "evaluate_predictions",
    "evaluate_ranking",
    "higher_is_better_for",
    "metric_specs",
    "metric_specs_for_statistic",
    "primary_metric_for",
    "ranking_metrics",
    "rating_weights_for",
    "regression_kendall_tau",
    "regression_mae",
    "regression_r2",
    "regression_rmse",
]
