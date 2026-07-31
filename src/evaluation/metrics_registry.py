"""Central registry of evaluation metrics.

To add a new metric, create a function with the signature:

    def my_metric(y_true: np.ndarray, pred: np.ndarray, target_mask=None) -> float: ...

Then call register_metric() with the appropriate task_type, higher_is_better, and
is_primary flags. The metric will automatically appear in training metrics.json and
in the bootstrap post-training evaluation.

Example:
    Call ``evaluate_predictions(y_true, y_pred, task_type, split="test")``.
"""

from __future__ import annotations

import collections.abc
import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.stats
import sklearn.metrics

MetricFn = collections.abc.Callable[
    [np.ndarray, np.ndarray, np.ndarray | None],
    float,
]
Array = npt.NDArray[np.generic]


@dataclasses.dataclass
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
    metrics = {
        entry.name: entry.fn for entry in _REGISTRY if entry.task_type == task_type
    }
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


def _binary_or_macro(
    y: Array,
    p: Array,
) -> tuple[Array, Array, str]:
    y_label = np.rint(y).astype(np.int8)
    p_label = np.rint(p).astype(np.int8)
    labels = np.unique(np.concatenate([y_label, p_label]))
    is_binary = labels.size <= 2 and set(labels.tolist()) <= {0, 1}
    average = "binary" if is_binary else "macro"
    return y_label, p_label, average


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
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size < 2 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return float("nan")
    tau = scipy.stats.kendalltau(y, p).statistic
    return float("nan") if tau is None or not np.isfinite(tau) else float(tau)


def classification_accuracy(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, _ = _binary_or_macro(y, p)
    return float(sklearn.metrics.accuracy_score(y_label, p_label))


def classification_precision(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(
        sklearn.metrics.precision_score(
            y_label,
            p_label,
            average=average,
            zero_division=0,
        )
    )


def classification_recall(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(
        sklearn.metrics.recall_score(
            y_label,
            p_label,
            average=average,
            zero_division=0,
        )
    )


def classification_f1(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, average = _binary_or_macro(y, p)
    return float(
        sklearn.metrics.f1_score(
            y_label,
            p_label,
            average=average,
            zero_division=0,
        )
    )


def classification_auroc(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    """Calculate binary AUROC from saved hard-label predictions."""
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, _ = _binary_or_macro(y, p)
    if np.unique(y_label).size < 2:
        return float("nan")
    return float(sklearn.metrics.roc_auc_score(y_label, p_label))


def classification_auprc(
    y_true: Array,
    pred: Array,
    target_mask: Array | None = None,
) -> float:
    """Calculate binary AUPRC from saved hard-label predictions."""
    y, p = _flatten_known(y_true, pred, target_mask)
    if y.size == 0:
        return float("nan")
    y_label, p_label, _ = _binary_or_macro(y, p)
    if not np.any(y_label == 1):
        return 0.0
    return float(sklearn.metrics.average_precision_score(y_label, p_label))


register_metric(
    "mae",
    "regression",
    regression_mae,
    higher_is_better=False,
    is_primary=True,
)
register_metric(
    "rmse",
    "regression",
    regression_rmse,
    higher_is_better=False,
)
register_metric("r2", "regression", regression_r2, higher_is_better=True)
register_metric(
    "kendall_tau",
    "regression",
    regression_kendall_tau,
    higher_is_better=True,
)

register_metric(
    "accuracy",
    "classification",
    classification_accuracy,
    higher_is_better=True,
)
register_metric(
    "precision",
    "classification",
    classification_precision,
    higher_is_better=True,
)
register_metric(
    "recall",
    "classification",
    classification_recall,
    higher_is_better=True,
)
register_metric(
    "f1",
    "classification",
    classification_f1,
    higher_is_better=True,
    is_primary=True,
)
register_metric(
    "auroc",
    "classification",
    classification_auroc,
    higher_is_better=True,
)
register_metric(
    "auprc",
    "classification",
    classification_auprc,
    higher_is_better=True,
)


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
) -> dict[str, object]:
    """Compute all registered metrics for one split. Used by training and bootstrap."""
    y_flat, pred_flat = _flatten_for_eval(y_true, pred, target_mask)
    result: dict[str, object] = {
        "split": split,
        "task_type": task_type,
        "n": int(y_flat.size),
    }
    if y_flat.size == 0:
        return result
    for name, fn in metric_set_for(task_type).items():
        result[name] = fn(y_flat, pred_flat)
    return result


__all__ = [
    "MetricEntry",
    "MetricFn",
    "classification_accuracy",
    "classification_auprc",
    "classification_auroc",
    "classification_f1",
    "classification_precision",
    "classification_recall",
    "evaluate_predictions",
    "higher_is_better_for",
    "metric_set_for",
    "primary_metric_for",
    "register_metric",
    "regression_kendall_tau",
    "regression_mae",
    "regression_r2",
    "regression_rmse",
]
