"""Output and persistence helpers for the training pipelines."""

from __future__ import annotations

import csv
import fcntl
import json
import lzma
from pathlib import Path
import math
from typing import Any

import joblib
import numpy as np
import torch

from models.results_card import (
    build_prediction_artifact_path,
    canonical_embedding_mode,
    canonical_model_family,
    canonical_split_family,
)


def _flatten_safetensors_payload(prefix: str, payload) -> dict[str, torch.Tensor]:
    """Flatten numeric prediction payloads into tensors accepted by safetensors."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        flattened = {}
        for key, value in payload.items():
            flattened.update(_flatten_safetensors_payload(f"{prefix}_{key}", value))
        return flattened

    array = np.asarray(payload)
    if array.dtype.kind in {"O", "S", "U"}:
        return {}
    array = np.ascontiguousarray(array)
    try:
        tensor = torch.from_numpy(array)
    except TypeError:
        return {}
    return {prefix: tensor}


def make_safetensors_predictions_file(
    y_true,
    pred,
    run_dir: Path,
    split: str,
    target_mask=None,
    *,
    artifact_path: Path | None = None,
) -> Path:
    """Persist one split as compressed safetensors."""
    from safetensors.torch import save as save_tensors

    payload = {}
    payload.update(_flatten_safetensors_payload("y_true", y_true))
    payload.update(_flatten_safetensors_payload("predictions", pred))
    payload.update(_flatten_safetensors_payload("target_mask", target_mask))
    if not payload:
        raise ValueError("Prediction payload does not contain tensors supported by safetensors.")

    path = artifact_path or (run_dir / f"{split}_predictions.safetensors.xz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_bytes = save_tensors(payload)
    with lzma.open(path, "wb") as handle:
        handle.write(tensor_bytes)
    return path


def save_prediction_artifacts(
    y_true,
    pred,
    target_mask,
    *,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    split_name: str,
    task_type: str,
) -> tuple[Path, dict[str, Any]]:
    """Persist safetensors predictions and compute metrics for one split."""
    prediction_path = build_prediction_artifact_path(
        run_dir,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        split_variant=split_variant,
        split_name=split_name,
    )
    prediction_path = make_safetensors_predictions_file(
        y_true,
        pred,
        run_dir,
        split_name,
        target_mask,
        artifact_path=prediction_path,
    )
    metrics = evaluate_predictions(y_true, pred, task_type, target_mask, split=split_name)
    return prediction_path, metrics


def _json_ready(value):
    """Convert NumPy/PyTorch scalars and paths into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def _as_flat_known_arrays(y_true, pred, target_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten prediction payloads while dropping masked, NaN, and shape-mismatched values."""
    if isinstance(y_true, dict):
        y_chunks = []
        pred_chunks = []
        for key, y_value in y_true.items():
            if not isinstance(pred, dict) or key not in pred:
                continue
            mask_value = target_mask.get(key) if isinstance(target_mask, dict) else None
            y_flat, pred_flat = _as_flat_known_arrays(y_value, pred[key], mask_value)
            if y_flat.size:
                y_chunks.append(y_flat)
                pred_chunks.append(pred_flat)
        if not y_chunks:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
        return np.concatenate(y_chunks), np.concatenate(pred_chunks)

    y_np = np.asarray(y_true)
    pred_np = np.asarray(pred)
    if y_np.shape != pred_np.shape:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    known = np.ones(y_np.shape, dtype=bool)
    if target_mask is not None:
        mask_np = np.asarray(target_mask, dtype=bool)
        if mask_np.shape == y_np.shape:
            known &= mask_np
        elif mask_np.ndim == 1 and y_np.ndim > 1 and mask_np.shape[0] == y_np.shape[0]:
            reshape = (mask_np.shape[0],) + (1,) * (y_np.ndim - 1)
            known &= np.broadcast_to(mask_np.reshape(reshape), y_np.shape)

    y_flat = y_np[known].reshape(-1)
    pred_flat = pred_np[known].reshape(-1)
    finite = np.isfinite(y_flat.astype(np.float64, copy=False)) & np.isfinite(
        pred_flat.astype(np.float64, copy=False)
    )
    return y_flat[finite], pred_flat[finite]


def _kendall_tau(y_true: np.ndarray, pred: np.ndarray) -> float | None:
    """Compute Kendall tau when SciPy is available and the split is non-degenerate."""
    if y_true.size < 2 or np.unique(y_true).size < 2 or np.unique(pred).size < 2:
        return None
    try:
        from scipy.stats import kendalltau
    except ModuleNotFoundError:
        return None
    value = kendalltau(y_true, pred).statistic
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def evaluate_predictions(
    y_true,
    pred,
    task_type: str,
    target_mask=None,
    *,
    split: str,
) -> dict[str, Any]:
    """Compute lightweight split metrics from saved prediction payloads."""
    y_flat, pred_flat = _as_flat_known_arrays(y_true, pred, target_mask)
    metrics: dict[str, Any] = {
        "split": split,
        "task_type": task_type,
        "n": int(y_flat.size),
    }
    if y_flat.size == 0:
        return metrics

    y_float = y_flat.astype(np.float64, copy=False)
    pred_float = pred_flat.astype(np.float64, copy=False)

    if task_type == "regression":
        residual = pred_float - y_float
        metrics["mae"] = float(np.mean(np.abs(residual)))
        metrics["rmse"] = float(np.sqrt(np.mean(residual**2)))
        ss_tot = float(np.sum((y_float - np.mean(y_float)) ** 2))
        metrics["r2"] = None if ss_tot == 0.0 else float(1.0 - np.sum(residual**2) / ss_tot)
        metrics["kendall_tau"] = _kendall_tau(y_float, pred_float)
        return metrics

    y_label = np.rint(y_float).astype(np.int64, copy=False)
    pred_label = np.rint(pred_float).astype(np.int64, copy=False)
    metrics["accuracy"] = float(np.mean(y_label == pred_label))
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
    except ModuleNotFoundError:
        return metrics

    labels = np.unique(np.concatenate([y_label, pred_label]))
    average = "binary" if labels.size <= 2 and set(labels.tolist()) <= {0, 1} else "macro"
    metrics["precision"] = float(
        precision_score(y_label, pred_label, average=average, zero_division=0)
    )
    metrics["recall"] = float(recall_score(y_label, pred_label, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_label, pred_label, average=average, zero_division=0))
    return metrics


def write_hparams_file(
    run_dir: Path,
    *,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    model_info: dict,
) -> Path:
    """Persist the model configuration used for one run."""
    path = run_dir / "hparams.json"
    payload = {
        "task_name": task_name,
        "embedding_model": emb_name,
        "model_name": model_name,
        "split_variant": split_variant,
        "wrapper": getattr(model_info["wrap"], "__name__", str(model_info["wrap"])),
        "params": model_info["params"],
    }
    path.write_text(json.dumps(_json_ready(payload), indent=4) + "\n", encoding="utf-8")
    return path


def write_metrics_file(run_dir: Path, metrics_by_split: dict[str, dict[str, Any]]) -> Path:
    """Persist split metrics as a compact JSON artifact."""
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(_json_ready(metrics_by_split), indent=4) + "\n", encoding="utf-8")
    return path


def write_training_log_file(trained, run_dir: Path) -> Path:
    """Persist available training progress in a CSV, falling back to a one-row summary."""
    rows = getattr(trained, "training_log_", None)
    if not rows:
        rows = [
            {
                "best_epoch": getattr(trained, "best_epoch_", None),
                "best_val_loss": getattr(trained, "best_val_loss_", None),
                "best_iteration": getattr(trained, "best_iteration", None),
                "best_score": getattr(trained, "best_score", None),
            }
        ]

    clean_rows = [_json_ready(row) for row in rows]
    fieldnames: list[str] = []
    for row in clean_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["best_epoch", "best_val_loss"]

    path = run_dir / "training_log.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(clean_rows)
    return path


def save_model(trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
    """Save Torch state dicts or joblib-compatible estimators to disk."""
    if hasattr(trained, "model") and isinstance(trained.model, torch.nn.Module):
        path = model_path or (run_dir / "model.pt")
        torch.save(trained.model.state_dict(), path)
    else:
        path = model_path or (run_dir / "model.joblib")
        joblib.dump(trained, path)
    return path


def save_model_safetensors(
    trained,
    run_dir: Path,
    *,
    model_path: Path | None = None,
) -> Path | None:
    """Save Torch model weights as safetensors when possible."""
    model = getattr(trained, "model", None)
    if not isinstance(model, torch.nn.Module):
        return None
    from safetensors.torch import save_model as save_torch_model

    path = model_path or (run_dir / "model.safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_model(model, path)
    return path


def make_result_entry(
    emb_name,
    model_name,
    model_info,
    run_dir: Path,
    model_file: Path,
    *,
    task_name: str | None = None,
    split_variant: str | None = None,
    task_scope: str = "properties",
    train_pred_file: Path | str | None = None,
    test_pred_file: Path | str | None = None,
    val_pred_file: Path | str | None = None,
    train_pred_safetensors_file: Path | str | None = None,
    test_pred_safetensors_file: Path | str | None = None,
    val_pred_safetensors_file: Path | str | None = None,
    model_safetensors_file: Path | str | None = None,
    metrics: dict[str, Any] | None = None,
    metrics_file: Path | str | None = None,
    hparams_file: Path | str | None = None,
    training_log_file: Path | str | None = None,
) -> dict:
    """Build the results-card payload describing one trained model artifact."""
    if train_pred_file is None and train_pred_safetensors_file is not None:
        train_pred_file = train_pred_safetensors_file
    if train_pred_file is None and task_name is not None and split_variant is not None:
        train_pred_file = build_prediction_artifact_path(
            run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name="train",
        )
    elif train_pred_file is None:
        train_pred_file = run_dir / "train_predictions.safetensors.xz"

    if test_pred_file is None and test_pred_safetensors_file is not None:
        test_pred_file = test_pred_safetensors_file
    if test_pred_file is None and task_name is not None and split_variant is not None:
        test_pred_file = build_prediction_artifact_path(
            run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name="test",
        )
    elif test_pred_file is None:
        test_pred_file = run_dir / "test_predictions.safetensors.xz"

    if val_pred_file is None and val_pred_safetensors_file is not None:
        val_pred_file = val_pred_safetensors_file
    if val_pred_file is None and task_name is not None and split_variant is not None:
        candidate_val = build_prediction_artifact_path(
            run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name="val",
        )
        if candidate_val.exists():
            val_pred_file = candidate_val
    elif val_pred_file is None:
        candidate_val = run_dir / "val_predictions.safetensors.xz"
        if candidate_val.exists():
            val_pred_file = candidate_val

    if train_pred_safetensors_file is None and str(train_pred_file).endswith(".safetensors.xz"):
        train_pred_safetensors_file = train_pred_file
    if test_pred_safetensors_file is None and str(test_pred_file).endswith(".safetensors.xz"):
        test_pred_safetensors_file = test_pred_file
    if (
        val_pred_safetensors_file is None
        and val_pred_file is not None
        and str(val_pred_file).endswith(".safetensors.xz")
    ):
        val_pred_safetensors_file = val_pred_file

    entry = {
        "result_key": f"{emb_name}__{model_name}",
        "embedding_model": emb_name,
        "name": model_name,
        "model_name": model_name,
        "params": model_info["params"],
        "run_dir": str(run_dir),
        "train_pred_file": str(train_pred_file),
        "test_pred_file": str(test_pred_file),
        "model_file": str(model_file),
    }
    if task_name is not None:
        entry["task_name"] = task_name
        entry["embedding_mode"] = canonical_embedding_mode(task_name, model_name)
    if split_variant is not None:
        entry["split_family"] = canonical_split_family(split_variant)
    entry["model_family"] = canonical_model_family(model_name)
    entry["task_scope"] = task_scope
    entry["metrics"] = metrics or {}
    if metrics_file is not None:
        entry["metrics_file"] = str(metrics_file)
    if hparams_file is not None:
        entry["hparams_file"] = str(hparams_file)
    if training_log_file is not None:
        entry["training_log_file"] = str(training_log_file)
    if val_pred_file is not None:
        entry["val_pred_file"] = str(val_pred_file)
    if train_pred_safetensors_file is not None:
        entry["train_pred_safetensors_file"] = str(train_pred_safetensors_file)
    if test_pred_safetensors_file is not None:
        entry["test_pred_safetensors_file"] = str(test_pred_safetensors_file)
    if val_pred_safetensors_file is not None:
        entry["val_pred_safetensors_file"] = str(val_pred_safetensors_file)
    if model_safetensors_file is not None:
        entry["model_safetensors_file"] = str(model_safetensors_file)
    return entry


def read_json(path) -> dict:
    """Read a JSON file into a Python dictionary."""
    with open(path) as f:
        return json.load(f)


def write_json(data: dict, path):
    """Write a JSON payload with stable indentation for diffs."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_card_locked(card_path: Path, task_name: str, result_key: str, result_entry: dict):
    """Thread/process-safe update of a single result entry in the benchmark card."""
    lock_path = card_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            card = read_json(card_path)
            card["properties"][task_name].setdefault("training_results", {})
            card["properties"][task_name]["training_results"][result_key] = result_entry
            write_json(card, card_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


__all__ = [
    "evaluate_predictions",
    "make_result_entry",
    "make_safetensors_predictions_file",
    "read_json",
    "save_model",
    "save_model_safetensors",
    "save_prediction_artifacts",
    "update_card_locked",
    "write_hparams_file",
    "write_metrics_file",
    "write_training_log_file",
    "write_json",
]
