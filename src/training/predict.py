"""Prediction logic: generating and persisting model outputs for benchmark tasks."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_loading.stores import get_dataloader, get_target_payload
from evaluation.artifacts import save_prediction_artifacts


@dataclass
class ModelRunOutputs:
    """Predictions and targets produced by one trained model on one task."""

    trained: Any
    pred_train: Any
    pred_val: Any
    pred_test: Any
    y_train: Any
    y_val: Any
    y_test: Any
    target_mask_train: Any = None
    target_mask_val: Any = None
    target_mask_test: Any = None


@dataclass
class SavedModelRun:
    """Trained estimator plus the split artifacts saved for one model run."""

    trained: Any
    prediction_paths: dict[str, Path]
    metrics_by_split: dict[str, dict[str, Any]]


@dataclass
class ResidueTrainingPayload:
    """Residue-aware training payload kept small until predictions are saved."""

    trained: Any
    loaders: Any


def predict_from_arrays(
    trained,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    use_validation: bool,
    target_mask_train=None,
    target_mask_val=None,
    target_mask_test=None,
) -> ModelRunOutputs:
    """Generate predictions for all splits from an array-based trained model."""
    return ModelRunOutputs(
        trained=trained,
        pred_train=trained.predict(x_train),
        pred_val=trained.predict(x_val) if use_validation else None,
        pred_test=trained.predict(x_test),
        y_train=y_train,
        y_val=y_val if use_validation else None,
        y_test=y_test,
        target_mask_train=target_mask_train,
        target_mask_val=target_mask_val if use_validation else None,
        target_mask_test=target_mask_test,
    )


def predict_from_loaders(
    trained,
    train_predict_loader,
    val_loader,
    test_loader,
    prepared,
    use_validation: bool,
    batch_kind: str,
) -> ModelRunOutputs:
    """Generate predictions for all splits from a loader-based trained model."""
    pred_train = trained.predict_loader(train_predict_loader, batch_kind=batch_kind)
    pred_val = trained.predict_loader(val_loader, batch_kind=batch_kind) if use_validation else None
    pred_test = trained.predict_loader(test_loader, batch_kind=batch_kind)

    y_train, target_mask_train = get_target_payload(prepared.train_bundle)
    y_val, target_mask_val = (
        get_target_payload(prepared.val_bundle) if use_validation else (None, None)
    )
    y_test, target_mask_test = get_target_payload(prepared.test_bundle)

    return ModelRunOutputs(
        trained=trained,
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        target_mask_train=target_mask_train,
        target_mask_val=target_mask_val if use_validation else None,
        target_mask_test=target_mask_test,
    )


def save_split_predictions(
    outputs: ModelRunOutputs,
    *,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    task_type: str,
    use_validation: bool,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Save predictions and compute metrics for each split."""
    split_payloads = [("train", outputs.y_train, outputs.pred_train, outputs.target_mask_train)]
    if use_validation:
        split_payloads.append(("val", outputs.y_val, outputs.pred_val, outputs.target_mask_val))
    split_payloads.append(("test", outputs.y_test, outputs.pred_test, outputs.target_mask_test))

    saved_paths: dict[str, Path] = {}
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name, y_true, pred, target_mask in split_payloads:
        prediction_path, metrics = save_prediction_artifacts(
            y_true,
            pred,
            target_mask,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name=split_name,
            task_type=task_type,
        )
        saved_paths[split_name] = prediction_path
        metrics_by_split[split_name] = metrics
    return saved_paths, metrics_by_split


def save_residue_predictions(
    trained_payload: ResidueTrainingPayload,
    *,
    prepared,
    use_validation: bool,
    config,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    task_type: str,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Generate and save residue predictions split-by-split to manage memory."""
    from data_loading.stores import log

    split_jobs = [
        ("train", prepared.train_bundle, trained_payload.loaders.train_loader),
        ("test", prepared.test_bundle, trained_payload.loaders.test_loader),
    ]
    if use_validation:
        split_jobs.insert(1, ("val", prepared.val_bundle, trained_payload.loaders.val_loader))

    saved_paths: dict[str, Path] = {}
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name, bundle, loader in split_jobs:
        log(f"      Predicting/saving split='{split_name}'...")
        y_true, target_mask = get_target_payload(bundle)
        pred = trained_payload.trained.predict_loader(loader, batch_kind=config.batch_kind)
        prediction_path, metrics = save_prediction_artifacts(
            y_true,
            pred,
            target_mask,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name=split_name,
            task_type=task_type,
        )
        saved_paths[split_name] = prediction_path
        metrics_by_split[split_name] = metrics
        del y_true, target_mask, pred
        gc.collect()
    return saved_paths, metrics_by_split
