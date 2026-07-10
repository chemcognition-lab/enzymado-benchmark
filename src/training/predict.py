"""Prediction logic: generating model outputs for benchmark tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
