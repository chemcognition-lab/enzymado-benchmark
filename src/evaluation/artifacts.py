"""Artifact persistence and evaluation helpers for trained models."""

from __future__ import annotations

import csv
import json
import lzma
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from . import sample_identity
from .metrics_registry import evaluate_predictions  # noqa: F401 — re-exported
from .results_layout import (
    INVALIDATION_MARKER,
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


def _payload_row_count(payload: Any) -> int:
    """Return a consistent leading dimension from a flat or structured payload."""
    if isinstance(payload, dict):
        counts = {_payload_row_count(value) for value in payload.values()}
        if len(counts) != 1:
            raise ValueError(
                f"Prediction payload blocks have inconsistent row counts: {counts}."
            )
        return counts.pop()

    array = np.asarray(payload)
    if array.ndim == 0:
        raise ValueError("Prediction payloads must have a sample dimension.")
    return int(array.shape[0])


def make_safetensors_predictions_file(
    y_true,
    pred,
    run_dir: Path,
    split: str,
    target_mask=None,
    *,
    artifact_path: Path | None = None,
    sample_ids: Sequence[str] | None = None,
) -> Path:
    """Persist one split as compressed safetensors."""
    from safetensors.torch import save as save_tensors

    payload = {}
    payload.update(_flatten_safetensors_payload("y_true", y_true))
    payload.update(_flatten_safetensors_payload("predictions", pred))
    payload.update(_flatten_safetensors_payload("target_mask", target_mask))
    if sample_ids is not None:
        normalized_ids = sample_identity.normalize(sample_ids)
        expected_rows = _payload_row_count(y_true)
        if len(normalized_ids) != expected_rows:
            raise ValueError(
                f"Received {len(normalized_ids)} sample IDs for "
                f"{expected_rows} prediction rows."
            )
        byte_values, offsets = sample_identity.encode(normalized_ids)
        payload[sample_identity.SAMPLE_ID_BYTES_KEY] = torch.from_numpy(
            byte_values
        )
        payload[sample_identity.SAMPLE_ID_OFFSETS_KEY] = torch.from_numpy(
            offsets
        )
    if not payload:
        raise ValueError(
            "Prediction payload does not contain tensors supported by safetensors."
        )

    path = artifact_path or (run_dir / f"{split}_predictions.safetensors.xz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_bytes = save_tensors(payload)
    with lzma.open(path, "wb") as handle:
        handle.write(tensor_bytes)
    return path


def save_raw_predictions(
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
    sample_ids: Sequence[str] | None = None,
) -> Path:
    """Persist one split's raw predictions as safetensors.

    Training only writes raw predictions; metrics.json is computed later by
    scripts/results/compute_metrics.py reading these files plus hparams.json,
    not inline during training.
    """
    prediction_path = build_prediction_artifact_path(
        run_dir,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        split_variant=split_variant,
        split_name=split_name,
    )
    return make_safetensors_predictions_file(
        y_true,
        pred,
        run_dir,
        split_name,
        target_mask,
        artifact_path=prediction_path,
        sample_ids=sample_ids,
    )


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




def write_hparams_file(
    run_dir: Path,
    *,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    model_info: dict,
    task_config=None,
    sample_identity_manifest: dict[str, dict[str, int | str]] | None = None,
) -> Path:
    """Persist the model configuration used for one run."""
    invalidation_marker = run_dir / INVALIDATION_MARKER
    if invalidation_marker.exists():
        invalidation_marker.unlink()

    path = run_dir / "hparams.json"
    payload = {
        "task_name": task_name,
        "embedding_model": emb_name,
        "model_name": model_name,
        "split_variant": split_variant,
        "wrapper": getattr(model_info["wrap"], "__name__", str(model_info["wrap"])),
        "params": model_info["params"],
    }
    if task_config is not None:
        payload["task_config"] = {
            "registry_name": getattr(task_config, "name", None),
            "batch_kind": getattr(task_config, "batch_kind", None),
            "label_kind": getattr(task_config, "label_kind", None),
            "target_column": getattr(task_config, "target_column", None),
            "dataset_paths": getattr(task_config, "dataset_paths", {}),
        }
    if sample_identity_manifest is not None:
        payload["sample_identity"] = sample_identity_manifest
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


__all__ = [
    "evaluate_predictions",
    "make_result_entry",
    "make_safetensors_predictions_file",
    "read_json",
    "save_model",
    "save_model_safetensors",
    "save_raw_predictions",
    "write_hparams_file",
    "write_metrics_file",
    "write_training_log_file",
    "write_json",
]
