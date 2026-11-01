"""Prediction logic: generating model outputs for benchmark tasks."""

# Keep positional compatibility for the existing prediction helper API.
# ruff: noqa: FBT001

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing

import enzymado.data_loading as data_loading
import enzymado.types


@dataclass
class ModelRunOutputs:
    """Predictions and targets produced by one trained model on one task."""

    trained: object
    pred_train: enzymado.types.ArrayPayload
    pred_val: enzymado.types.OptionalArrayPayload
    pred_test: enzymado.types.ArrayPayload
    y_train: enzymado.types.ArrayPayload
    y_val: enzymado.types.OptionalArrayPayload
    y_test: enzymado.types.ArrayPayload
    target_mask_train: enzymado.types.OptionalArrayPayload = None
    target_mask_val: enzymado.types.OptionalArrayPayload = None
    target_mask_test: enzymado.types.OptionalArrayPayload = None


@dataclass
class SavedModelRun:
    """Trained estimator plus the raw prediction paths saved for one model run."""

    trained: object
    prediction_paths: dict[str, Path]


def predict_from_arrays(
    trained: object,
    x_train: enzymado.types.Array,
    y_train: enzymado.types.ArrayPayload,
    x_val: enzymado.types.Array | None,
    y_val: enzymado.types.OptionalArrayPayload,
    x_test: enzymado.types.Array,
    y_test: enzymado.types.ArrayPayload,
    use_validation: bool,
    target_mask_train: enzymado.types.OptionalArrayPayload = None,
    target_mask_val: enzymado.types.OptionalArrayPayload = None,
    target_mask_test: enzymado.types.OptionalArrayPayload = None,
) -> ModelRunOutputs:
    """Generate predictions for all splits from an array-based trained model."""
    predictor = typing.cast(enzymado.types.ArrayPredictor, trained)
    if use_validation and x_val is None:
        raise ValueError("Validation features are required when validation is enabled.")

    return ModelRunOutputs(
        trained=trained,
        pred_train=predictor.predict(x_train),
        pred_val=predictor.predict(x_val) if use_validation else None,
        pred_test=predictor.predict(x_test),
        y_train=y_train,
        y_val=y_val if use_validation else None,
        y_test=y_test,
        target_mask_train=target_mask_train,
        target_mask_val=target_mask_val if use_validation else None,
        target_mask_test=target_mask_test,
    )


def predict_from_loaders(
    trained: object,
    train_predict_loader: object,
    val_loader: object | None,
    test_loader: object,
    prepared: data_loading.LoaderTaskData,
    use_validation: bool,
    batch_kind: str,
) -> ModelRunOutputs:
    """Generate predictions for all splits from a loader-based trained model."""
    predictor = typing.cast(enzymado.types.LoaderPredictor, trained)
    if use_validation and val_loader is None:
        raise ValueError("A validation loader is required when validation is enabled.")

    pred_train = predictor.predict_loader(
        train_predict_loader,
        batch_kind=batch_kind,
    )
    pred_val = (
        predictor.predict_loader(val_loader, batch_kind=batch_kind)
        if use_validation
        else None
    )
    pred_test = predictor.predict_loader(test_loader, batch_kind=batch_kind)

    y_train, target_mask_train = data_loading.get_target_payload(prepared.train_bundle)
    y_val, target_mask_val = (
        data_loading.get_target_payload(prepared.val_bundle)
        if use_validation and prepared.val_bundle is not None
        else (None, None)
    )
    y_test, target_mask_test = data_loading.get_target_payload(prepared.test_bundle)

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
