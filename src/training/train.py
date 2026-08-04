"""Training routes: model fitting functions for benchmark tasks.

Data preparation logic now lives in data_loading.task_data (load_task_data).
This module owns only the training, saving, and route-selection pieces.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_loading import (
    LoaderTaskData,
    PooledTaskData,
    get_dataloader,
    load_loader_task_data,
    log,
)
from .model_registry import get_model_map, train_model, train_model_loader
from .predict import ResidueTrainingPayload
from .residue import build_residue_aware_loaders, train_residue_aware_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_arrays(
    *,
    model_info: dict,
    x_train,
    y_train,
    x_val,
    y_val,
    use_validation: bool,
    progress_label: str,
    checkpoint_path: Path,
    target_mask_train=None,
    target_mask_val=None,
) -> Any:
    """Fit one array-based model and return the trained estimator."""
    fit_kwargs: dict[str, Any] = {
        "x_val": x_val if use_validation else None,
        "y_val": y_val if use_validation else None,
        "progress_label": progress_label,
        "checkpoint_path": checkpoint_path,
    }
    if target_mask_train is not None:
        fit_kwargs["target_mask"] = target_mask_train
    if use_validation and target_mask_val is not None:
        fit_kwargs["target_mask_val"] = target_mask_val
    return train_model(model_info, x_train, y_train, **fit_kwargs)


def _select_global_arrays(model_info: dict, data: PooledTaskData) -> tuple:
    """Return (x_tr, y_tr, x_va, y_va, x_te, y_te, m_tr, m_va, m_te) for a global model."""
    if model_info.get("flat_ec_labels"):
        return (
            data.x_train_flat, data.y_train_flat,
            data.x_val_flat, data.y_val_flat,
            data.x_test_flat, data.y_test_flat,
            data.mask_train_flat, data.mask_val_flat, data.mask_test_flat,
        )
    return (
        data.x_train, data.y_train,
        data.x_val, data.y_val,
        data.x_test, data.y_test,
        data.target_mask_train, data.target_mask_val, data.target_mask_test,
    )


def _dataset_sizes(data: PooledTaskData | LoaderTaskData) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) independent of the chosen training route."""
    if isinstance(data, LoaderTaskData):
        n_train = len(data.train_bundle.dataset)
        n_val = len(data.val_bundle.dataset) if data.val_bundle is not None else 0
        n_test = len(data.test_bundle.dataset)
        return n_train, n_val, n_test
    # PooledTaskData: prefer hierarchical EC arrays, then flat EC, then plain arrays
    if data.x_train is not None:
        n_val = len(data.x_val) if data.x_val is not None else 0
        return len(data.x_train), n_val, len(data.x_test)
    if data.x_train_flat is not None:
        n_val = len(data.x_val_flat) if data.x_val_flat is not None else 0
        return len(data.x_train_flat), n_val, len(data.x_test_flat)
    raise RuntimeError("PooledTaskData contains no arrays.")


# ---------------------------------------------------------------------------
# Training route functions
# ---------------------------------------------------------------------------

def train_global(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    data: PooledTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> Any:
    """Fit an array-based model (MLP/Linear/XGB) on pooled embeddings."""
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    x_tr, y_tr, x_va, y_va, _, _, m_tr, m_va, _ = _select_global_arrays(model_info, data)

    if model_info.get("flat_ec_labels"):
        log("      Training on pooled flat EC embeddings for XGBoost...")
        log(
            f"      Flat EC tensors: x_train={x_tr.shape}, y_train={y_tr.shape},"
            f" known_train={int(m_tr.sum()):,}"
        )
    elif config.label_kind == "multilabel_ec":
        log("      Training on pooled EC embeddings...")
    else:
        log(f"      Training on pooled embeddings (train={x_tr.shape})...")

    trained = _fit_arrays(
        model_info=model_info,
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        use_validation=use_validation,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
        target_mask_train=m_tr,
        target_mask_val=m_va,
    )
    log(f"      Training done ({time.time() - t0:.1f}s).")
    return trained


def train_residue(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    data: LoaderTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> ResidueTrainingPayload:
    """Fit a residue-aware model using DataLoaders built from LoaderTaskData bundles."""
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    loaders = build_residue_aware_loaders(
        model_info,
        train_bundle=data.train_bundle,
        val_bundle=data.val_bundle,
        test_bundle=data.test_bundle,
        use_validation=use_validation,
    )
    log("      Training (residue-aware)...")
    trained = train_residue_aware_model(
        model_info,
        loaders,
        batch_kind=config.batch_kind,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
    )
    log(f"      Training done ({time.time() - t0:.1f}s).")
    return ResidueTrainingPayload(trained=trained, loaders=loaders)


def train_attention(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    data: LoaderTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> Any:
    """Fit an attention-pooling model using DataLoaders built from LoaderTaskData bundles."""
    batch_size = int(model_info.get("loader_batch_size", 16))
    num_workers = int(model_info.get("loader_num_workers", 4))
    loader_seed = int(model_info.get("loader_seed", 0))
    checkpoint_path = run_dir / "training_checkpoint.bin"
    train_loader = get_dataloader(
        data.train_bundle,
        batch_size=batch_size,
        shuffle=True,
        seed=loader_seed,
        num_workers=num_workers,
    )
    val_loader = (
        get_dataloader(data.val_bundle, batch_size=batch_size, num_workers=num_workers)
        if use_validation and data.val_bundle is not None else None
    )
    log("      Training attention model...")
    t0 = time.time()
    trained = train_model_loader(
        model_info, train_loader, config.batch_kind,
        val_loader=val_loader, checkpoint_path=checkpoint_path,
    )
    log(f"      Training done ({time.time() - t0:.1f}s).")
    return trained


# ---------------------------------------------------------------------------
# Attention data-preparation thin wrapper
# ---------------------------------------------------------------------------

def prepare_attention(
    config,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    *,
    allowed_ids: set[str] | None = None,
    sequence_to_id: dict[str, str] | None = None,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
) -> LoaderTaskData:
    """Thin wrapper: load data as LoaderTaskData for attention models on any batch_kind.

    Attention models always need per-residue DataLoaders regardless of the
    task's batch_kind. ``residue_store_mode`` is forwarded so per-residue tasks
    materialize the embedding subset into RAM when it fits. Delegates to
    data_loading.load_loader_task_data.
    """
    return load_loader_task_data(
        task_name, emb_path, split_variant,
        allowed_ids=allowed_ids,
        sequence_to_id=sequence_to_id,
        residue_store_mode=residue_store_mode,
        embedding_dim=embedding_dim,
    )


# ---------------------------------------------------------------------------
# Route specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingRouteSpec:
    """Bundle of plain functions describing one training route."""

    name: str  # "global" | "residue" | "attention"
    train: Callable
    prepare: Callable | None = None  # only ATTENTION_ROUTE uses this


GLOBAL_ROUTE = TrainingRouteSpec(
    name="global",
    train=train_global,
)
RESIDUE_ROUTE = TrainingRouteSpec(
    name="residue",
    train=train_residue,
)
ATTENTION_ROUTE = TrainingRouteSpec(
    name="attention",
    prepare=prepare_attention,
    train=train_attention,
)


def _select_training_route(config) -> TrainingRouteSpec:
    """Resolve a config into the route spec that owns the train function."""
    if config.batch_kind == "protein_residue":
        return RESIDUE_ROUTE
    return GLOBAL_ROUTE
