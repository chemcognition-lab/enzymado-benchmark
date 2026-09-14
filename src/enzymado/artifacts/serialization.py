"""Artifact persistence and evaluation helpers for trained models."""

from __future__ import annotations

from collections.abc import Sequence
import csv
import json
import lzma
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import safetensors.torch
import torch

import enzymado.models.training_protocol

from . import sample_identity
from .results_layout import (
    INVALIDATION_MARKER,
    build_prediction_artifact_path,
)

HPARAMS_SCHEMA_VERSION = 1


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


PROBA_QUANT_LEVELS = 65535
"""Fixed-point levels used to store probabilities. Bounded in [0, 1], so integer
levels beat any float format of the same width: uint16 resolves the ranking
AUROC/AUPRC measure ~4000x finer than bfloat16 at identical size."""


def quantize_probabilities(payload):
    """Quantize probabilities in [0, 1] to uint16 fixed point, preserving order."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        return {key: quantize_probabilities(value) for key, value in payload.items()}
    array = np.asarray(payload, dtype=np.float64)
    scaled = np.rint(np.clip(array, 0.0, 1.0) * PROBA_QUANT_LEVELS)
    return scaled.astype(np.uint16, copy=False)


def dequantize_probabilities(array):
    """Restore float32 probabilities from the stored uint16 fixed-point levels."""
    return np.asarray(array, dtype=np.float32) / np.float32(PROBA_QUANT_LEVELS)


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
    pred_proba=None,
) -> Path:
    """Persist one split as compressed safetensors."""
    payload = {}
    payload.update(_flatten_safetensors_payload("y_true", y_true))
    payload.update(_flatten_safetensors_payload("predictions", pred))
    payload.update(_flatten_safetensors_payload("target_mask", target_mask))
    payload.update(
        _flatten_safetensors_payload("proba", quantize_probabilities(pred_proba))
    )
    if sample_ids is not None:
        normalized_ids = sample_identity.normalize(sample_ids)
        expected_rows = _payload_row_count(y_true)
        if len(normalized_ids) != expected_rows:
            raise ValueError(
                f"Received {len(normalized_ids)} sample IDs for "
                f"{expected_rows} prediction rows."
            )
        byte_values, offsets = sample_identity.encode(normalized_ids)
        payload[sample_identity.SAMPLE_ID_BYTES_KEY] = torch.from_numpy(byte_values)
        payload[sample_identity.SAMPLE_ID_OFFSETS_KEY] = torch.from_numpy(offsets)
    if not payload:
        raise ValueError(
            "Prediction payload does not contain tensors supported by safetensors."
        )

    path = artifact_path or (run_dir / f"{split}_predictions.safetensors.xz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_bytes = safetensors.torch.save(payload)
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
    pred_proba=None,
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
        pred_proba=pred_proba,
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
    random_seed: int | None = None,
    deterministic_algorithms: bool | None = None,
) -> Path:
    """Persist the model configuration used for one run."""
    invalidation_marker = run_dir / INVALIDATION_MARKER
    if invalidation_marker.exists():
        invalidation_marker.unlink()

    path = run_dir / "hparams.json"
    payload = {
        "schema_version": HPARAMS_SCHEMA_VERSION,
        "task_name": task_name,
        "embedding_model": emb_name,
        "model_name": model_name,
        "split_variant": split_variant,
        "wrapper": getattr(model_info["wrap"], "__name__", str(model_info["wrap"])),
        "params": model_info["params"],
        "selection_protocol": enzymado.models.training_protocol.selection_protocol(
            model_info["wrap"], model_info["params"]
        ),
    }
    if random_seed is not None:
        payload["random_seed"] = int(random_seed)
    if deterministic_algorithms is not None:
        payload["deterministic_algorithms"] = bool(deterministic_algorithms)
    if task_config is not None:
        payload["task_config"] = {
            "registry_name": getattr(task_config, "name", None),
            "batch_kind": getattr(task_config, "batch_kind", None),
            "label_kind": getattr(task_config, "label_kind", None),
            "target_column": getattr(task_config, "target_column", None),
            "numeric_feature_columns": getattr(
                task_config,
                "numeric_feature_columns",
                (),
            ),
            "dataset_paths": getattr(task_config, "dataset_paths", {}),
        }
    if sample_identity_manifest is not None:
        payload["sample_identity"] = sample_identity_manifest
    payload = _json_ready(payload)
    if (run_dir / "training_checkpoint.bin").exists():
        if not path.exists() or json.loads(path.read_text()) != payload:
            raise ValueError(
                "Checkpoint configuration changed; use a new run directory."
            )
        return path
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return path


def _first_attr(obj, *names):
    """Return the first attribute present on ``obj``, or None if none are."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def write_training_log_file(trained, run_dir: Path) -> Path:
    """Persist training progress or a one-row fitted-model summary."""
    rows = getattr(trained, "training_log_", None)
    if not rows:
        rows = [
            {
                "best_epoch": getattr(trained, "best_epoch_", None),
                "best_val_loss": getattr(trained, "best_val_loss_", None),
                # Two conventions are in play: this repo's boosting wrappers
                # use the sklearn trailing underscore, while xgboost's own
                # estimators expose the bare names. Reading only one silently
                # logged None for half the runs.
                "best_iteration": _first_attr(
                    trained, "best_iteration_", "best_iteration"
                ),
                "best_score": _first_attr(trained, "best_score_", "best_score"),
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

    path = model_path or (run_dir / "model.safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, path)
    return path


__all__ = [
    "dequantize_probabilities",
    "make_safetensors_predictions_file",
    "quantize_probabilities",
    "save_model",
    "save_model_safetensors",
    "save_raw_predictions",
    "write_hparams_file",
    "write_training_log_file",
]
