"""Compute compact binary-classification statistics by resampling unit.

Example:
    Use ``count_blocks`` on prediction blocks, then call ``evaluate`` or
    ``bootstrap`` without flattening the original prediction tensors.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.evaluation import loaders

CountArray = npt.NDArray[np.int64]
MetricArrays = dict[str, npt.NDArray[np.float64]]


def _valid_mask(payload: loaders.PredictionPayload) -> npt.NDArray[np.bool_]:
    """Return a validated mask broadcastable to the prediction shape."""
    y_true = np.asarray(payload.y_true)
    predictions = np.asarray(payload.predictions)
    if y_true.shape != predictions.shape:
        raise ValueError(
            "y_true and predictions must have matching shapes; "
            f"got {y_true.shape} and {predictions.shape}."
        )

    valid = np.isfinite(y_true) & np.isfinite(predictions)
    if payload.target_mask is None:
        return valid

    target_mask = np.asarray(payload.target_mask, dtype=bool)
    if target_mask.shape == y_true.shape:
        return valid & target_mask
    if target_mask.ndim == 1 and target_mask.shape[0] == y_true.shape[0]:
        shape = (target_mask.shape[0],) + (1,) * (y_true.ndim - 1)
        return valid & target_mask.reshape(shape)
    raise ValueError(
        "target_mask must match y_true or contain one value per row; "
        f"got {target_mask.shape} for y_true {y_true.shape}."
    )


def _sum_rows(values: npt.NDArray[np.bool_]) -> CountArray:
    """Sum all non-row dimensions while preserving bootstrap units."""
    if values.ndim == 1:
        return values.astype(np.int64)
    axes = tuple(range(1, values.ndim))
    return values.sum(axis=axes, dtype=np.int64)


def count_payload(payload: loaders.PredictionPayload) -> CountArray:
    """Return per-row TP, FP, FN, and TN sufficient statistics."""
    with np.errstate(invalid="ignore"):
        y_label = np.rint(payload.y_true).astype(np.int8, copy=False)
        prediction_label = np.rint(payload.predictions).astype(
            np.int8,
            copy=False,
        )
    valid = _valid_mask(payload)

    true_positive = _sum_rows((y_label == 1) & (prediction_label == 1) & valid)
    false_positive = _sum_rows((y_label == 0) & (prediction_label == 1) & valid)
    false_negative = _sum_rows((y_label == 1) & (prediction_label == 0) & valid)
    true_negative = _sum_rows((y_label == 0) & (prediction_label == 0) & valid)
    return np.stack(
        [true_positive, false_positive, false_negative, true_negative],
        axis=1,
    )


def count_blocks(
    blocks: tuple[loaders.PredictionPayload, ...],
) -> CountArray:
    """Aggregate multiple EC heads without concatenating their dense tensors."""
    if not blocks:
        raise ValueError("At least one prediction block is required.")

    counts = count_payload(blocks[0])
    for block in blocks[1:]:
        block_counts = count_payload(block)
        if block_counts.shape != counts.shape:
            raise ValueError(
                "Prediction blocks must have the same number of rows; "
                f"got {counts.shape[0]} and {block_counts.shape[0]}."
            )
        counts += block_counts
    return counts


def _metrics(totals: CountArray) -> MetricArrays:
    """Convert one or more rows of confusion counts to metric arrays."""
    if totals.ndim != 2 or totals.shape[1] != 4:
        raise ValueError(f"Expected counts with shape (N, 4), got {totals.shape}.")

    true_positive = totals[:, 0].astype(np.float64)
    false_positive = totals[:, 1].astype(np.float64)
    false_negative = totals[:, 2].astype(np.float64)
    true_negative = totals[:, 3].astype(np.float64)

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    actual_negative = false_positive + true_negative
    total = true_positive + false_positive + false_negative + true_negative

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    accuracy = np.divide(
        true_positive + true_negative,
        total,
        out=np.zeros_like(true_positive),
        where=total > 0,
    )
    true_negative_rate = np.divide(
        true_negative,
        actual_negative,
        out=np.full_like(true_negative, np.nan),
        where=actual_negative > 0,
    )
    true_positive_rate = np.divide(
        true_positive,
        actual_positive,
        out=np.full_like(true_positive, np.nan),
        where=actual_positive > 0,
    )
    auroc = 0.5 * (true_positive_rate + true_negative_rate)
    prevalence = np.divide(
        actual_positive,
        total,
        out=np.zeros_like(actual_positive),
        where=total > 0,
    )
    auprc = precision * recall + prevalence * (1.0 - recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
    }


def evaluate(counts: CountArray, *, split: str) -> dict[str, float | int | str]:
    """Calculate point metrics from per-row sufficient statistics."""
    totals = counts.sum(axis=0, dtype=np.int64, keepdims=True)
    n_known = int(totals.sum())
    result: dict[str, float | int | str] = {
        "split": split,
        "task_type": "classification",
        "n": n_known,
    }
    if not n_known:
        return result
    metrics = _metrics(totals)
    return result | {name: float(values[0]) for name, values in metrics.items()}


def bootstrap(counts: CountArray, indices: CountArray) -> MetricArrays:
    """Calculate bootstrap metrics without materializing a 3D count tensor."""
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D bootstrap indices, got {indices.shape}.")
    if not counts.size or indices.shape[1] != counts.shape[0]:
        raise ValueError(
            f"Indices expect {indices.shape[1]} rows but counts have {counts.shape[0]}."
        )

    totals = np.empty((indices.shape[0], 4), dtype=np.int64)
    for column_index in range(4):
        totals[:, column_index] = counts[:, column_index][indices].sum(axis=1)
    return _metrics(totals)
