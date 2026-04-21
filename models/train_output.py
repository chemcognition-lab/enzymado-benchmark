"""Output and persistence helpers for the training pipelines."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from models.results_card import (
    build_prediction_artifact_path,
    canonical_embedding_mode,
    canonical_model_family,
    canonical_split_family,
)


def _flatten_npz_payload(prefix: str, payload) -> dict[str, np.ndarray]:
    """Flatten nested prediction payloads so they can be saved in one NPZ file."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        flattened = {}
        for key, value in payload.items():
            flattened.update(_flatten_npz_payload(f"{prefix}_{key}", value))
        return flattened
    return {prefix: np.asarray(payload)}


def make_predictions_file(y_true, pred, run_dir: Path, split: str, target_mask=None, *, artifact_path: Path | None = None) -> Path:
    """Persist predictions, labels, and optional masks for one data split."""
    payload = {}
    payload.update(_flatten_npz_payload("y_true", y_true))
    payload.update(_flatten_npz_payload("predictions", pred))
    payload.update(_flatten_npz_payload("target_mask", target_mask))
    path = artifact_path or (run_dir / f"{split}_predictions.npz")
    np.savez(str(path), **payload)
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
) -> dict:
    """Build the results-card payload describing one trained model artifact."""
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
        train_pred_file = run_dir / "train_predictions.npz"

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
        test_pred_file = run_dir / "test_predictions.npz"

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
        candidate_val = run_dir / "val_predictions.npz"
        if candidate_val.exists():
            val_pred_file = candidate_val

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
    entry["metrics"] = {}
    if val_pred_file is not None:
        entry["val_pred_file"] = str(val_pred_file)
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
    "make_predictions_file",
    "make_result_entry",
    "read_json",
    "save_model",
    "update_card_locked",
    "write_json",
]
