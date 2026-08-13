"""Level-aware evaluation for hierarchical EC-number predictions.

PyTorch prediction artifacts preserve the four EC heads, whereas XGBoost
artifacts contain the same targets concatenated into one matrix.  This module
normalizes both representations before computing metrics.  Thresholds are
selected independently for each EC level from validation predictions and are
then reused unchanged for test evaluation and protein-level bootstrap.
"""

from __future__ import annotations

import csv
import functools
import pathlib

import numpy as np

from src.evaluation import binary_statistics, loaders, metrics_registry

LEVEL_NAMES = ("level_1", "level_2", "level_3", "level_4")
THRESHOLD_METRICS = ("accuracy", "precision", "recall", "f1")
RANKING_METRICS = ("auroc", "auprc")
DEFAULT_THRESHOLD_GRID_SIZE = 1001
_ROW_CHUNK_SIZE = 512


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_dataset_path(raw_path: str) -> pathlib.Path:
    """Resolve historical absolute paths against the current checkout."""
    path = pathlib.Path(raw_path)
    if path.exists():
        return path

    parts = path.parts
    try:
        data_index = parts.index("data")
    except ValueError as exc:
        raise FileNotFoundError(f"EC dataset path does not exist: {path}") from exc

    checkout_path = _repo_root().joinpath(*parts[data_index:])
    if not checkout_path.exists():
        raise FileNotFoundError(
            f"EC dataset path does not exist at {path} or {checkout_path}."
        )
    return checkout_path


def _ec_dataset_metadata(run_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Return the training CSV and target column recorded for one EC run."""
    hparams = loaders.read_hparams(run_dir)
    task_config = hparams.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(f"Missing task_config in {run_dir / 'hparams.json'}.")

    dataset_paths = task_config.get("dataset_paths")
    split_variant = hparams.get("split_variant")
    target_column = task_config.get("target_column")
    if not isinstance(dataset_paths, dict) or not isinstance(split_variant, str):
        raise ValueError(f"Missing EC dataset/split metadata in {run_dir}.")
    raw_path = dataset_paths.get(split_variant)
    if not isinstance(raw_path, str) or not isinstance(target_column, str):
        raise ValueError(f"Incomplete EC label metadata in {run_dir}.")
    return _resolve_dataset_path(raw_path), target_column


@functools.lru_cache(maxsize=8)
def _vocabulary_widths(
    dataset_path: pathlib.Path,
    target_column: str,
) -> tuple[int, int, int, int]:
    """Rebuild the train-only EC vocabulary widths used during training."""
    labels_by_level = [set() for _ in LEVEL_NAMES]
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise ValueError(f"Missing split column in {dataset_path}.")
        if target_column not in reader.fieldnames:
            raise ValueError(
                f"Missing EC target column '{target_column}' in {dataset_path}."
            )
        for row in reader:
            if (row.get("split") or "").strip() != "train":
                continue
            raw_labels = (row.get(target_column) or "").strip()
            for ec_label in raw_labels.split(";"):
                ec_label = ec_label.strip()
                if not ec_label:
                    continue
                parts = tuple(part.strip() for part in ec_label.split("."))
                if len(parts) != len(LEVEL_NAMES):
                    raise ValueError(
                        f"Invalid EC label '{ec_label}' in {dataset_path}."
                    )
                known_parts: list[str] = []
                for level_index, part in enumerate(parts):
                    if part == "-":
                        break
                    known_parts.append(part)
                    labels_by_level[level_index].add(".".join(known_parts))

    widths = tuple(len(labels) for labels in labels_by_level)
    if any(width == 0 for width in widths):
        raise ValueError(
            f"EC train vocabulary has an empty level in {dataset_path}: {widths}."
        )
    return widths  # type: ignore[return-value]


def level_widths(run_dir: pathlib.Path) -> tuple[int, int, int, int]:
    """Return the four EC output widths for one experiment."""
    dataset_path, target_column = _ec_dataset_metadata(run_dir)
    return _vocabulary_widths(dataset_path, target_column)


def load_level_blocks(
    run_dir: pathlib.Path,
    split_name: str,
) -> dict[str, loaders.PredictionPayload] | None:
    """Load EC predictions as four named levels for either model family."""
    blocks = loaders.load_split_blocks(run_dir, split_name)
    if blocks is None:
        return None

    if len(blocks) == len(LEVEL_NAMES):
        widths = level_widths(run_dir)
        observed = tuple(int(block.y_true.shape[1]) for block in blocks)
        if observed != widths:
            raise ValueError(
                f"EC head widths {observed} do not match train vocabulary "
                f"widths {widths} in {run_dir}."
            )
        return dict(zip(LEVEL_NAMES, blocks, strict=True))

    if len(blocks) != 1:
        raise ValueError(
            f"Expected one flat EC block or four EC heads in {run_dir}; "
            f"found {len(blocks)} blocks."
        )

    flat = blocks[0]
    y_true = np.asarray(flat.y_true)
    predictions = np.asarray(flat.predictions)
    if y_true.ndim != 2 or predictions.shape != y_true.shape:
        raise ValueError(
            f"Flat EC payload requires matching 2D arrays, got "
            f"{y_true.shape} and {predictions.shape}."
        )

    widths = level_widths(run_dir)
    if sum(widths) != y_true.shape[1]:
        raise ValueError(
            f"Flat EC width {y_true.shape[1]} does not match train vocabulary "
            f"width {sum(widths)} in {run_dir}."
        )

    target_mask = None if flat.target_mask is None else np.asarray(flat.target_mask)
    if target_mask is not None and target_mask.ndim not in {1, 2}:
        raise ValueError(f"Unsupported flat EC mask shape {target_mask.shape}.")

    result: dict[str, loaders.PredictionPayload] = {}
    start = 0
    for level_name, width in zip(LEVEL_NAMES, widths, strict=True):
        stop = start + width
        level_mask = (
            target_mask
            if target_mask is None or target_mask.ndim == 1
            else target_mask[:, start:stop]
        )
        result[level_name] = loaders.PredictionPayload(
            y_true=y_true[:, start:stop],
            predictions=predictions[:, start:stop],
            target_mask=level_mask,
            sample_ids=flat.sample_ids,
        )
        start = stop
    return result


def _valid_mask(payload: loaders.PredictionPayload) -> np.ndarray:
    """Return the finite, known-label mask for one EC level."""
    y_true = np.asarray(payload.y_true)
    predictions = np.asarray(payload.predictions)
    if y_true.shape != predictions.shape:
        raise ValueError(
            f"EC targets and predictions differ: {y_true.shape} vs "
            f"{predictions.shape}."
        )
    valid = np.isfinite(y_true) & np.isfinite(predictions)
    if payload.target_mask is None:
        return valid
    target_mask = np.asarray(payload.target_mask, dtype=bool)
    if target_mask.shape == y_true.shape:
        return valid & target_mask
    if target_mask.ndim == 1 and target_mask.shape[0] == y_true.shape[0]:
        return valid & target_mask[:, None]
    raise ValueError(
        f"EC mask {target_mask.shape} is incompatible with {y_true.shape}."
    )


def calibrate_f1_threshold(
    payload: loaders.PredictionPayload,
    *,
    grid_size: int = DEFAULT_THRESHOLD_GRID_SIZE,
) -> dict[str, float | int | str]:
    """Select a validation F1 threshold using a bounded probability grid.

    Histogramming avoids sorting hundreds of millions of label probabilities.
    Ties are resolved toward 0.5 for a stable, conservative operating point.
    """
    if grid_size < 2:
        raise ValueError(f"grid_size must be at least 2, got {grid_size}.")

    positive_histogram = np.zeros(grid_size, dtype=np.int64)
    negative_histogram = np.zeros(grid_size, dtype=np.int64)
    y_true = np.asarray(payload.y_true)
    predictions = np.asarray(payload.predictions)
    valid = _valid_mask(payload)

    for start in range(0, y_true.shape[0], _ROW_CHUNK_SIZE):
        stop = min(start + _ROW_CHUNK_SIZE, y_true.shape[0])
        chunk_valid = valid[start:stop]
        if not np.any(chunk_valid):
            continue
        labels = np.rint(y_true[start:stop][chunk_valid]).astype(
            np.int8,
            copy=False,
        )
        scores = np.clip(
            predictions[start:stop][chunk_valid],
            0.0,
            1.0,
        )
        bin_indices = np.floor(scores * (grid_size - 1)).astype(
            np.int32,
            copy=False,
        )
        positive_histogram += np.bincount(
            bin_indices[labels == 1],
            minlength=grid_size,
        )
        negative_histogram += np.bincount(
            bin_indices[labels == 0],
            minlength=grid_size,
        )

    n_positive = int(positive_histogram.sum())
    n_negative = int(negative_histogram.sum())
    n_known = n_positive + n_negative
    if not n_known or not n_positive:
        return {
            "threshold": 0.5,
            "validation_f1": 0.0,
            "n": n_known,
            "n_positive": n_positive,
            "method": "validation_f1_grid",
            "grid_size": grid_size,
        }

    true_positive = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1]
    false_positive = np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1]
    false_negative = n_positive - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(grid_size, dtype=np.float64),
        where=denominator > 0,
    )
    best_f1 = float(np.max(f1))
    candidates = np.flatnonzero(np.isclose(f1, best_f1, rtol=0.0, atol=1e-12))
    thresholds = candidates.astype(np.float64) / float(grid_size - 1)
    best_index = int(candidates[np.argmin(np.abs(thresholds - 0.5))])
    return {
        "threshold": float(best_index / float(grid_size - 1)),
        "validation_f1": best_f1,
        "n": n_known,
        "n_positive": n_positive,
        "method": "validation_f1_grid",
        "grid_size": grid_size,
    }


def calibrate_thresholds(
    run_dir: pathlib.Path,
    *,
    grid_size: int = DEFAULT_THRESHOLD_GRID_SIZE,
) -> dict[str, dict[str, float | int | str]]:
    """Calibrate one validation threshold for each EC level."""
    blocks = load_level_blocks(run_dir, "val")
    if blocks is None:
        raise FileNotFoundError(f"Missing validation predictions in {run_dir}.")
    return {
        level_name: calibrate_f1_threshold(block, grid_size=grid_size)
        for level_name, block in blocks.items()
    }


def _mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(np.mean(finite))


def evaluate_split(
    run_dir: pathlib.Path,
    split_name: str,
    calibrations: dict[str, dict[str, float | int | str]],
    *,
    include_ranking: bool = True,
) -> dict[str, object]:
    """Evaluate one EC split by level and aggregate levels with equal weight."""
    blocks = load_level_blocks(run_dir, split_name)
    if blocks is None:
        raise FileNotFoundError(
            f"Missing {split_name} predictions in {run_dir}."
        )

    levels: dict[str, dict[str, object]] = {}
    counts_by_level: list[np.ndarray] = []
    for level_name in LEVEL_NAMES:
        block = blocks[level_name]
        threshold = float(calibrations[level_name]["threshold"])
        counts = binary_statistics.count_payload(block, threshold=threshold)
        counts_by_level.append(counts)
        level_result: dict[str, object] = dict(
            binary_statistics.evaluate(counts, split=split_name)
        )
        totals = counts.sum(axis=0, dtype=np.int64)
        n_positive = int(totals[0] + totals[2])
        level_result.update(
            {
                "threshold": threshold,
                "threshold_source": "validation",
                "n_positive": n_positive,
                "positive_prevalence": (
                    float(n_positive / level_result["n"])
                    if level_result["n"]
                    else float("nan")
                ),
            }
        )
        if include_ranking:
            level_result["auroc"] = metrics_registry.classification_auroc(
                block.y_true,
                block.predictions,
                block.target_mask,
            )
            level_result["auprc"] = metrics_registry.classification_auprc(
                block.y_true,
                block.predictions,
                block.target_mask,
            )
        levels[level_name] = level_result

    aggregate_metric_names = list(THRESHOLD_METRICS)
    if include_ranking:
        aggregate_metric_names.extend(RANKING_METRICS)
    aggregate = {
        name: _mean_finite(
            [float(levels[level_name][name]) for level_name in LEVEL_NAMES]
        )
        for name in aggregate_metric_names
    }

    micro_counts = counts_by_level[0].copy()
    for counts in counts_by_level[1:]:
        micro_counts += counts
    micro = binary_statistics.evaluate(micro_counts, split=split_name)
    micro.pop("split", None)
    micro.pop("task_type", None)

    return {
        "split": split_name,
        "task_type": "classification",
        "evaluation_kind": "hierarchical_ec",
        "aggregation": "unweighted_mean_over_ec_levels",
        "threshold_source": "validation",
        "n": int(sum(int(levels[name]["n"]) for name in LEVEL_NAMES)),
        "n_units": int(next(iter(blocks.values())).y_true.shape[0]),
        "n_levels": len(LEVEL_NAMES),
        "thresholds": {
            name: float(calibrations[name]["threshold"])
            for name in LEVEL_NAMES
        },
        "threshold_calibration": calibrations,
        "levels": levels,
        "micro_threshold_metrics": micro,
        **aggregate,
    }


def bootstrap_split(
    run_dir: pathlib.Path,
    indices: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    """Bootstrap threshold metrics by protein, preserving EC levels."""
    blocks = load_level_blocks(run_dir, "test")
    if blocks is None:
        raise FileNotFoundError(f"Missing test predictions in {run_dir}.")
    n_units = next(iter(blocks.values())).y_true.shape[0]
    if indices.ndim != 2 or indices.shape[1] != n_units:
        raise ValueError(
            f"Bootstrap indices have shape {indices.shape}; expected (*, {n_units})."
        )

    output: dict[str, np.ndarray] = {}
    level_replicates: dict[str, dict[str, np.ndarray]] = {}
    micro_counts = np.zeros((n_units, 4), dtype=np.int64)
    for level_name in LEVEL_NAMES:
        counts = binary_statistics.count_payload(
            blocks[level_name],
            threshold=float(thresholds[level_name]),
        )
        micro_counts += counts
        replicates = binary_statistics.bootstrap(counts, indices)
        level_replicates[level_name] = replicates
        for metric_name, values in replicates.items():
            output[f"{level_name}_{metric_name}"] = values

    for metric_name in THRESHOLD_METRICS:
        output[metric_name] = np.nanmean(
            np.stack(
                [
                    level_replicates[level_name][metric_name]
                    for level_name in LEVEL_NAMES
                ],
                axis=0,
            ),
            axis=0,
        )

    for metric_name, values in binary_statistics.bootstrap(
        micro_counts,
        indices,
    ).items():
        output[f"micro_{metric_name}"] = values
    return output


def n_resample_units(run_dir: pathlib.Path) -> int | None:
    """Return the number of test proteins in one EC experiment."""
    blocks = load_level_blocks(run_dir, "test")
    if blocks is None:
        return None
    return int(next(iter(blocks.values())).y_true.shape[0])


__all__ = [
    "DEFAULT_THRESHOLD_GRID_SIZE",
    "LEVEL_NAMES",
    "RANKING_METRICS",
    "THRESHOLD_METRICS",
    "bootstrap_split",
    "calibrate_f1_threshold",
    "calibrate_thresholds",
    "evaluate_split",
    "level_widths",
    "load_level_blocks",
    "n_resample_units",
]
