"""Shared helpers for residue-aware training flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_loading import ProteinResidueConcatCollator

from data_loading.stores import get_dataloader, get_target_payload, log
from models.registry import train_model_loader


@dataclass
class ResidueAwareLoaders:
    """Train/validation/test DataLoaders for residue-aware models."""

    train_loader: Any
    train_predict_loader: Any
    val_loader: Any | None
    test_loader: Any


@dataclass
class ResidueAwareOutputs:
    """Predictions and targets collected from residue-aware loaders."""

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


def build_residue_aware_loaders(
    model_info: dict,
    *,
    train_bundle,
    val_bundle,
    test_bundle,
    use_validation: bool,
) -> ResidueAwareLoaders:
    """Create loaders for residue-aware training and prediction."""
    residue_batch_size = int(model_info.get("loader_batch_size", 64))
    log(f"      Using residue-aware loader batch_size={residue_batch_size}")
    residue_collate_fn = None
    if model_info.get("concat_residue_batch", False):
        log("      Using concatenated residue batches (sum of per-protein lengths per batch).")
        residue_collate_fn = ProteinResidueConcatCollator()

    train_loader = get_dataloader(
        train_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
        shuffle=True,
    )
    train_predict_loader = get_dataloader(
        train_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
    )
    val_loader = (
        get_dataloader(
            val_bundle,
            batch_size=residue_batch_size,
            collate_fn=residue_collate_fn,
        )
        if use_validation else None
    )
    test_loader = get_dataloader(
        test_bundle,
        batch_size=residue_batch_size,
        collate_fn=residue_collate_fn,
    )
    return ResidueAwareLoaders(
        train_loader=train_loader,
        train_predict_loader=train_predict_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def train_residue_aware_model(
    model_info: dict,
    loaders: ResidueAwareLoaders,
    *,
    batch_kind: str,
    progress_label: str,
    checkpoint_path: Path,
):
    """Train one residue-aware model using loader-based APIs."""
    return train_model_loader(
        model_info,
        loaders.train_loader,
        batch_kind,
        val_loader=loaders.val_loader,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
    )


def collect_residue_aware_outputs(
    trained,
    *,
    loaders: ResidueAwareLoaders,
    train_bundle,
    val_bundle,
    test_bundle,
    batch_kind: str,
    use_validation: bool,
) -> ResidueAwareOutputs:
    """Predict across residue-aware loaders and collect targets in memory."""
    y_val, target_mask_val = (
        get_target_payload(val_bundle) if use_validation else (None, None)
    )
    y_test, target_mask_test = get_target_payload(test_bundle)
    y_train, target_mask_train = get_target_payload(train_bundle)
    return ResidueAwareOutputs(
        trained=trained,
        pred_train=trained.predict_loader(loaders.train_predict_loader, batch_kind=batch_kind),
        pred_val=(
            trained.predict_loader(loaders.val_loader, batch_kind=batch_kind)
            if use_validation else None
        ),
        pred_test=trained.predict_loader(loaders.test_loader, batch_kind=batch_kind),
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
    "build_residue_aware_loaders",
    "collect_residue_aware_outputs",
    "train_residue_aware_model",
]
