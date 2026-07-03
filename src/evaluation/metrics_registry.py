"""Central registry of evaluation metrics.

To add a new metric, create a function with the signature:

    def my_metric(y_true: np.ndarray, pred: np.ndarray, target_mask=None) -> float: ...

Then call register_metric() with the appropriate task_type, higher_is_better, and
is_primary flags. The metric will automatically appear in training metrics.json and
in the bootstrap post-training evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


@dataclass
class MetricEntry:
    name: str
    task_type: str
    fn: MetricFn
    higher_is_better: bool
    is_primary: bool = False


_REGISTRY: list[MetricEntry] = []


def register_metric(
    name: str,
    task_type: str,
    fn: MetricFn,
    *,
    higher_is_better: bool,
    is_primary: bool = False,
) -> None:
    """Register a metric function for a given task type."""
    _REGISTRY.append(
        MetricEntry(
            name=name,
            task_type=task_type,
            fn=fn,
            higher_is_better=higher_is_better,
            is_primary=is_primary,
        )
    )


def metric_set_for(task_type: str) -> dict[str, MetricFn]:
    """Return all registered metric functions for the given task type."""
    metrics = {entry.name: entry.fn for entry in _REGISTRY if entry.task_type == task_type}
    if not metrics:
        raise ValueError(f"No metrics registered for task_type '{task_type}'.")
    return metrics


def higher_is_better_for(task_type: str) -> dict[str, int]:
    """Return sign convention (+1 / -1) per metric for the given task type."""
    return {
        entry.name: (1 if entry.higher_is_better else -1)
        for entry in _REGISTRY
        if entry.task_type == task_type
    }


def primary_metric_for(task_type: str) -> str | None:
    """Return the name of the primary metric for the given task type."""
    for entry in _REGISTRY:
        if entry.task_type == task_type and entry.is_primary:
            return entry.name
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_known(
    y_true: np.ndarray,
    pred: np.ndarray,
    target_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and optionally mask arrays to 1D float64."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    if target_mask is not None:
        keep = np.asarray(target_mask, dtype=bool).ravel()
        if keep.shape == y.shape:
            y, p = y[keep], p[keep]
    finite = np.isfinite(y) & np.isfinite(p)
    return y[finite], p[finite]


def _binary_or_macro(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    y_label = np.rint(y).astype(np.int8)
    p_label = np.rint(p).astype(np.int8)
    labels = np.unique(np.concatenate([y_label, p_label]))
    average = "binary" if labels.size <= 2 and set(labels.tolist()) <= {0, 1} else "macro"
    return y_label, p_label, average


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_mae(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    return float("nan") if y.size == 0 else float(np.mean(np.abs(p - y)))


def regression_rmse(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    return float("nan") if y.size == 0 else float(np.sqrt(np.mean((p - y) ** 2)))


def regression_r2(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2:
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0.0 else float(1.0 - np.sum((p - y) ** 2) / ss_tot)


def regression_kendall_tau(y_true, pred, target_mask=None) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return float("nan")
    tau = kendalltau(y, p).statistic
    return float("nan") if tau is None or not np.isfinite(tau) else float(tau)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_metric("mae",          "regression",     regression_mae,          higher_is_better=False, is_primary=True)
register_metric("rmse",         "regression",     regression_rmse,         higher_is_better=False)
register_metric("r2",           "regression",     regression_r2,           higher_is_better=True)
register_metric("kendall_tau",  "regression",     regression_kendall_tau,  higher_is_better=True)

register_metric("accuracy",     "classification", classification_accuracy,  higher_is_better=True)
register_metric("precision",    "classification", classification_precision, higher_is_better=True)
register_metric("recall",       "classification", classification_recall,    higher_is_better=True)
register_metric("f1",           "classification", classification_f1,        higher_is_better=True, is_primary=True)


# ---------------------------------------------------------------------------
# evaluate_predictions — single entry point used by training and bootstrap
# ---------------------------------------------------------------------------

def _flatten_for_eval(y_true, pred, target_mask=None):
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
    y_true,
    pred,
    task_type: str,
    target_mask=None,
    *,
    split: str,
) -> dict:
    """Compute all registered metrics for one split. Used by training and bootstrap."""
    y_flat, pred_flat = _flatten_for_eval(y_true, pred, target_mask)
    result: dict = {"split": split, "task_type": task_type, "n": int(y_flat.size)}
    if y_flat.size == 0:
        return result
    for name, fn in metric_set_for(task_type).items():
        result[name] = fn(y_flat, pred_flat)
    return result


__all__ = [
    "MetricEntry",
    "MetricFn",
    "evaluate_predictions",
    "higher_is_better_for",
    "metric_set_for",
    "primary_metric_for",
    "register_metric",
    "regression_mae",
    "regression_rmse",
    "regression_r2",
    "regression_kendall_tau",
    "classification_accuracy",
    "classification_precision",
    "classification_recall",
    "classification_f1",
]
