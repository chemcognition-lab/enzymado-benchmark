"""Prediction logic: generating model outputs for benchmark tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_loading import get_target_payload


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
    """Trained estimator plus the raw prediction paths saved for one model run."""

    trained: Any
    prediction_paths: dict[str, Path]


@dataclass
class ResidueTrainingPayload:
    """Residue-aware training payload kept small until predictions are saved."""

    trained: Any
    loaders: Any


def predict_split_from_arrays(trained, features, task_type: str):
    """Return regression values or continuous classification scores."""
    if task_type != "classification":
        return trained.predict(features)

    score_predictor = getattr(trained, "predict_scores", None)
    if score_predictor is not None:
        return score_predictor(features)

    probability_predictor = getattr(trained, "predict_proba", None)
    if probability_predictor is None:
        raise TypeError(
            f"{type(trained).__name__} must implement predict_scores() or "
            "predict_proba() for classification."
        )
    probabilities = probability_predictor(features)
    if isinstance(probabilities, dict):
        return probabilities

    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        return probabilities[:, 1]
    raise ValueError(
        "Binary classification probabilities must have shape (N,) or (N, 2); "
        f"got {probabilities.shape}."
    )


def predict_split_from_loader(trained, dataloader, batch_kind: str, task_type: str):
    """Return regression values or continuous classification scores from a loader."""
    if task_type != "classification":
        return trained.predict_loader(dataloader, batch_kind=batch_kind)

    score_predictor = getattr(trained, "predict_scores_loader", None)
    if score_predictor is None:
        raise TypeError(
            f"{type(trained).__name__} must implement predict_scores_loader() "
            "for loader-based classification."
        )
    return score_predictor(dataloader, batch_kind=batch_kind)


def predict_from_arrays(
    trained,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    *,
    use_validation: bool,
    task_type: str,
    target_mask_train=None,
    target_mask_val=None,
    target_mask_test=None,
) -> ModelRunOutputs:
    """Generate predictions for all splits from an array-based trained model."""
    return ModelRunOutputs(
        trained=trained,
        pred_train=predict_split_from_arrays(trained, x_train, task_type),
        pred_val=(
            predict_split_from_arrays(trained, x_val, task_type)
            if use_validation
            else None
        ),
        pred_test=predict_split_from_arrays(trained, x_test, task_type),
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
    *,
    use_validation: bool,
    batch_kind: str,
    task_type: str,
) -> ModelRunOutputs:
    """Generate predictions for all splits from a loader-based trained model."""
    pred_train = predict_split_from_loader(
        trained,
        train_predict_loader,
        batch_kind,
        task_type,
    )
    pred_val = (
        predict_split_from_loader(trained, val_loader, batch_kind, task_type)
        if use_validation
        else None
    )
    pred_test = predict_split_from_loader(trained, test_loader, batch_kind, task_type)

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
