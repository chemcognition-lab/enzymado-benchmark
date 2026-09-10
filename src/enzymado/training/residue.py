"""Shared helpers for residue-aware training flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing

import torch.utils.data

import enzymado.data_loading as data_loading
import enzymado.training.model_registry as model_registry
import enzymado.types


@dataclass
class ResidueAwareLoaders:
    """Train/validation/test DataLoaders for residue-aware models."""

    train_loader: torch.utils.data.DataLoader[data_loading.Batch]
    train_predict_loader: torch.utils.data.DataLoader[data_loading.Batch]
    val_loader: torch.utils.data.DataLoader[data_loading.Batch] | None
    test_loader: torch.utils.data.DataLoader[data_loading.Batch]


@dataclass
class ResidueAwareOutputs:
    """Predictions and targets collected from residue-aware loaders."""

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
class ResidueTrainingPayload:
    """Residue-aware training payload kept small until predictions are saved."""

    trained: object
    loaders: ResidueAwareLoaders


def build_residue_aware_loaders(
    model_info: enzymado.types.ModelInfo,
    *,
    train_bundle: data_loading.TaskDatasetBundle,
    val_bundle: data_loading.TaskDatasetBundle | None,
    test_bundle: data_loading.TaskDatasetBundle,
    use_validation: bool,
) -> ResidueAwareLoaders:
    """Create loaders for residue-aware training and prediction."""
    if use_validation and val_bundle is None:
        raise ValueError("A validation bundle is required when validation is enabled.")

    residue_batch_size = int(model_info.get("loader_batch_size", 64))
    num_workers = int(model_info.get("loader_num_workers", 4))
    loader_seed = int(model_info.get("loader_seed", 0))
    data_loading.log(
        f"      Using residue-aware loader batch_size={residue_batch_size} "
        f"num_workers={num_workers}"
    )
    residue_collate_fn = None
    if model_info.get("concat_residue_batch", False):
        data_loading.log(
            "      Using concatenated residue batches "
            "(sum of per-protein lengths per batch)."
        )
        residue_collate_fn = data_loading.ProteinResidueConcatCollator()

    # xgb.QuantileDMatrix streams batches from this loader through a DataIter
    # across multiple internal passes (one to build the quantile sketch, one
    # to write the compressed histogram matrix) and requires reset() to replay
    # the *same* batch sequence every pass. shuffle=True reshuffles on every
    # fresh iter(), so the streaming XGBoost path (concat_residue_batch) would
    # see a different batch grouping per pass and crash with
    # `vector::_M_range_check`. Order doesn't affect XGBoost's global
    # histogram/sketch construction, so disabling shuffle there is free.
    train_loader = data_loading.get_dataloader(
        train_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
        shuffle=not model_info.get("concat_residue_batch", False),
        seed=loader_seed,
        num_workers=num_workers,
    )
    train_predict_loader = data_loading.get_dataloader(
        train_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
        num_workers=num_workers,
    )
    val_loader = (
        data_loading.get_dataloader(
            val_bundle,
            batch_size=residue_batch_size,
            collate_fn=residue_collate_fn,
            num_workers=num_workers,
        )
        if use_validation
        else None
    )
    test_loader = data_loading.get_dataloader(
        test_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
        num_workers=num_workers,
    )
    return ResidueAwareLoaders(
        train_loader=train_loader,
        train_predict_loader=train_predict_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def train_residue_aware_model(
    model_info: enzymado.types.ModelInfo,
    loaders: ResidueAwareLoaders,
    *,
    batch_kind: str,
    progress_label: str,
    checkpoint_path: Path,
) -> object:
    """Train one residue-aware model using loader-based APIs."""
    return model_registry.train_model_loader(
        model_info,
        loaders.train_loader,
        batch_kind,
        val_loader=loaders.val_loader,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
    )


def collect_residue_aware_outputs(
    trained: object,
    *,
    loaders: ResidueAwareLoaders,
    train_bundle: data_loading.TaskDatasetBundle,
    val_bundle: data_loading.TaskDatasetBundle | None,
    test_bundle: data_loading.TaskDatasetBundle,
    batch_kind: str,
    use_validation: bool,
) -> ResidueAwareOutputs:
    """Predict across residue-aware loaders and collect targets in memory."""
    predictor = typing.cast(enzymado.types.LoaderPredictor, trained)
    if use_validation and val_bundle is None:
        raise ValueError("A validation bundle is required when validation is enabled.")

    y_val, target_mask_val = (
        data_loading.get_target_payload(val_bundle)
        if use_validation and val_bundle is not None
        else (None, None)
    )
    y_test, target_mask_test = data_loading.get_target_payload(test_bundle)
    y_train, target_mask_train = data_loading.get_target_payload(train_bundle)
    return ResidueAwareOutputs(
        trained=trained,
        pred_train=predictor.predict_loader(
            loaders.train_predict_loader, batch_kind=batch_kind
        ),
        pred_val=(
            predictor.predict_loader(loaders.val_loader, batch_kind=batch_kind)
            if use_validation
            else None
        ),
        pred_test=predictor.predict_loader(
            loaders.test_loader,
            batch_kind=batch_kind,
        ),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        target_mask_train=target_mask_train,
        target_mask_val=target_mask_val,
        target_mask_test=target_mask_test,
    )


__all__ = [
    "ResidueAwareLoaders",
    "ResidueAwareOutputs",
    "ResidueTrainingPayload",
    "build_residue_aware_loaders",
    "collect_residue_aware_outputs",
    "train_residue_aware_model",
]
