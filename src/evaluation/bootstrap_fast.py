"""Fast bootstrap path for binary classification metrics.

The generic bootstrap loop in `bootstrap.py` re-runs `sklearn.f1_score` per
replicate, which is fine for tabular tasks but degenerates badly on big
2D inputs (multilabel `ec_number`, per-residue tasks). For those cases the
metric is mathematically a function of micro-aggregated TP/FP/FN/TN counts:
sklearn's `f1_score(y.ravel(), pred.ravel(), average="binary")` on
`(N, C)` arrays with binary labels is exactly micro-F1 over all flattened
cells.

Strategy: pre-aggregate per-row {TP, FP, FN, TN} once before the bootstrap
loop, then each replicate is just `precomputed[idx].sum(axis=0)` and four
arithmetic operations. The result is bit-exact with the generic path, and
roughly two orders of magnitude faster on big 2D inputs.

Regression tasks (kendall_tau, r2 are non-additive) and multi-class
classification stay on the generic path.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.loaders import PredictionPayload


def is_fast_classification_eligible(payload: PredictionPayload) -> bool:
    """Return True iff the predictions encode a binary {0,1} classification target.

    The fast path requires that, after rounding, both `y_true` and `predictions`
    contain only labels in {0, 1}. Multi-class problems would need a per-class
    aggregation and stay on the generic path.

    NaN entries (per-residue tasks pad with NaN at masked positions) are ignored
    in this check; they do not disqualify the payload because the mask removes
    them from the count anyway.
    """
    y = payload.y_true
    p = payload.predictions
    if y.size == 0 or p.size == 0:
        return False
    # Use NaN-aware reductions so we never materialise a float-filtered copy of
    # the full (N, C) array (that copy is what made this >1 GB on multi-head EC).
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
    # After rounding, values must land in {0, 1}. Allow a small margin for FP noise.
    return y_max <= 1.5 and p_max <= 1.5 and y_min >= -0.5 and p_min >= -0.5


def precompute_binary_micro_indicators(payload: PredictionPayload) -> np.ndarray:
    """Aggregate per-row {TP, FP, FN, TN} for a binary classification payload.

    Works for 1D arrays (tabular binary) and 2D arrays (per-residue binary or
    multilabel binary, where micro-F1 is the natural global aggregate).

    Returns an int64 array of shape `(n_units, 4)` whose columns are
    `[TP, FP, FN, TN]`. Bootstrap replicates can then sum rows in O(R · N).
    """
    y = payload.y_true
    p = payload.predictions
    m = payload.target_mask

    if m is None:
        valid = np.ones(y.shape, dtype=bool)
    else:
        valid = np.asarray(m, dtype=bool)

    # Cast to int8 directly. Masked-out cells may carry NaN → undefined int8,
    # but the boolean masks below gate every cell on `valid`, so those cells
    # never enter the TP/FP/FN/TN counts. Suppress the cast warning that
    # numpy issues for NaN→int conversions; the result is still correct.
    with np.errstate(invalid="ignore"):
        y_int = np.rint(y).astype(np.int8, copy=False)
        p_int = np.rint(p).astype(np.int8, copy=False)

    # Build the four indicator masks. For 2D arrays we sum within each row;
    # for 1D arrays the per-cell indicator already lives at the unit level.
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


def bootstrap_classification_fast(
    precomputed: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply resample indices to per-row counts and return one array per metric.

    `precomputed` has shape `(n_units, 4)` with columns `[TP, FP, FN, TN]`.
    `indices` has shape `(n_replicates, n_units)`. Output has the same metric
    keys as the classification half of `metrics.metric_set_for("classification")`.
    """
    if precomputed.ndim != 2 or precomputed.shape[1] != 4:
        raise ValueError(f"precomputed must have shape (n_units, 4), got {precomputed.shape}")
    if indices.shape[1] != precomputed.shape[0]:
        raise ValueError(
            f"indices last axis ({indices.shape[1]}) must match precomputed.shape[0] "
            f"({precomputed.shape[0]})."
        )

    # gathered: (R, N, 4) → totals: (R, 4)
    totals = precomputed[indices].sum(axis=1, dtype=np.int64)
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
