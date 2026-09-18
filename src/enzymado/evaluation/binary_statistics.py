"""Compute compact binary-classification statistics by resampling unit.

Example:
    Use ``count_blocks`` on prediction blocks, then call ``evaluate`` or
    ``bootstrap`` without flattening the original prediction tensors.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enzymado.evaluation import loaders
from enzymado.evaluation.metrics_registry import RANKING_BINS

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


def ranking_histograms(
    payload: loaders.PredictionPayload,
    *,
    n_bins: int = RANKING_BINS,
) -> np.ndarray:
    """Return per-row (n_rows, 2 * n_bins) score histograms split by true label."""
    scores = payload.proba
    if scores is None:
        raise ValueError("Ranking statistics require stored probabilities.")
    scores = np.asarray(scores, dtype=np.float64)
    levels = np.clip(np.rint(scores * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)
    labels = np.rint(np.asarray(payload.y_true)).astype(np.int8, copy=False)
    valid = _valid_mask(payload)

    # Pooled tasks store one prediction per protein (1D); residue and EC tasks
    # store a padded row per protein (2D). Both resample by row, so give the
    # pooled case an explicit width of one instead of branching downstream.
    if levels.ndim == 1:
        levels = levels.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        valid = np.asarray(valid).reshape(-1, 1)

    n_rows, width = levels.shape
    flat = np.zeros((n_rows, 2 * n_bins), dtype=np.int64)
    rows = np.repeat(np.arange(n_rows), width)
    offsets = (labels.ravel() == 1).astype(np.int64) * n_bins + levels.ravel()
    keep = np.asarray(valid).ravel()
    np.add.at(flat, (rows[keep], offsets[keep]), 1)
    return flat


def ranking_histogram_blocks(
    blocks: tuple[loaders.PredictionPayload, ...],
    *,
    n_bins: int = RANKING_BINS,
) -> np.ndarray | None:
    """Aggregate per-row score histograms across EC heads, mirroring count_blocks."""
    if not blocks or any(block.proba is None for block in blocks):
        return None

    histograms = ranking_histograms(blocks[0], n_bins=n_bins)
    for block in blocks[1:]:
        block_histograms = ranking_histograms(block, n_bins=n_bins)
        if block_histograms.shape != histograms.shape:
            raise ValueError(
                "Prediction blocks must have the same number of rows; "
                f"got {histograms.shape[0]} and {block_histograms.shape[0]}."
            )
        histograms += block_histograms
    return histograms
