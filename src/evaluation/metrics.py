"""Pure metric functions consumed by the bootstrap loop.

Each function accepts (y_true, pred, target_mask=None) where target_mask filters
unknown entries (used for multilabel EC and per-residue tasks). Functions return a
single float so they can be passed straight into resampling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


def _flatten_known(y_true: np.ndarray, pred: np.ndarray, target_mask=None):
    if target_mask is not None:
        keep = np.asarray(target_mask, dtype=bool)
        return y_true[keep].ravel(), pred[keep].ravel()
    return y_true.ravel(), pred.ravel()


def regression_mae(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p.astype(np.float64) - y.astype(np.float64))))


def regression_rmse(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    diff = p.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def regression_r2(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2:
        return float("nan")
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - np.sum((p - y) ** 2) / ss_tot)


def regression_kendall_tau(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2:
        return float("nan")
    tau = kendalltau(y, p).statistic
    if tau is None or not np.isfinite(tau):
        return float("nan")
    return float(tau)


def _binary_or_macro(y: np.ndarray, p: np.ndarray):
    y_label = np.rint(y).astype(np.int8, copy=False)
    p_label = np.rint(p).astype(np.int8, copy=False)
    labels = np.unique(np.concatenate([y_label, p_label]))
    average = "binary" if labels.size <= 2 and set(labels.tolist()) <= {0, 1} else "macro"
    return y_label, p_label, average


def classification_accuracy(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, _ = _binary_or_macro(y, p)
    return float(accuracy_score(y_label, p_label))


def classification_precision(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(precision_score(y_label, p_label, average=average, zero_division=0))


def classification_recall(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(recall_score(y_label, p_label, average=average, zero_division=0))


def classification_f1(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(f1_score(y_label, p_label, average=average, zero_division=0))


REGRESSION_METRICS: dict[str, MetricFn] = {
    "mae": regression_mae,
    "rmse": regression_rmse,
    "r2": regression_r2,
    "kendall_tau": regression_kendall_tau,
}

CLASSIFICATION_METRICS: dict[str, MetricFn] = {
    "accuracy": classification_accuracy,
    "precision": classification_precision,
    "recall": classification_recall,
    "f1": classification_f1,
}

PRIMARY_METRIC: dict[str, str] = {
    "regression": "mae",
    "classification": "f1",
}

# Sign convention used by the rating step to flip lower-is-better metrics so that
# "higher" always means "better" before computing pairwise outcomes.
HIGHER_IS_BETTER: dict[str, int] = {
    "mae": -1,
    "rmse": -1,
    "r2": +1,
    "kendall_tau": +1,
    "accuracy": +1,
    "precision": +1,
    "recall": +1,
    "f1": +1,
}


def metric_set_for(task_type: str) -> dict[str, MetricFn]:
    if task_type == "regression":
        return REGRESSION_METRICS
    if task_type == "classification":
        return CLASSIFICATION_METRICS
    raise ValueError(f"Unknown task_type '{task_type}'; expected 'regression' or 'classification'.")
