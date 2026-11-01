"""Training routes: model fitting functions for benchmark tasks.

Data preparation logic now lives in enzymado.data_loading.task_data (load_task_data).
This module owns only the training, saving, and route-selection pieces.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from pathlib import Path
import time

import enzymado.data_loading as data_loading
import enzymado.training.model_registry as model_registry
import enzymado.training.residue as residue
import enzymado.types


@dataclass(frozen=True)
class GlobalArrays:
    """Named train/validation/test arrays selected for one global model."""

    x_train: enzymado.types.Array
    y_train: enzymado.types.ArrayPayload
    x_val: enzymado.types.Array | None
    y_val: enzymado.types.OptionalArrayPayload
    x_test: enzymado.types.Array
    y_test: enzymado.types.ArrayPayload
    mask_train: enzymado.types.OptionalArrayPayload
    mask_val: enzymado.types.OptionalArrayPayload
    mask_test: enzymado.types.OptionalArrayPayload


def _fit_arrays(
    *,
    model_info: enzymado.types.ModelInfo,
    x_train: enzymado.types.Array,
    y_train: enzymado.types.ArrayPayload,
    x_val: enzymado.types.Array | None,
    y_val: enzymado.types.OptionalArrayPayload,
    use_validation: bool,
    progress_label: str,
    checkpoint_path: Path,
    target_mask_train: enzymado.types.OptionalArrayPayload = None,
    target_mask_val: enzymado.types.OptionalArrayPayload = None,
    unscaled_columns: tuple[int, int] | None = None,
) -> object:
    """Fit one array-based model and return the trained estimator."""
    fit_kwargs: dict[str, object] = {
        "x_val": x_val if use_validation else None,
        "y_val": y_val if use_validation else None,
        "progress_label": progress_label,
        "checkpoint_path": checkpoint_path,
    }
    if target_mask_train is not None:
        fit_kwargs["target_mask"] = target_mask_train
    if use_validation and target_mask_val is not None:
        fit_kwargs["target_mask_val"] = target_mask_val
    if unscaled_columns is not None:
        fit_kwargs["unscaled_columns"] = unscaled_columns
    return model_registry.train_model(model_info, x_train, y_train, **fit_kwargs)


def _select_global_arrays(
    model_info: enzymado.types.ModelInfo,
    data: data_loading.PooledTaskData,
) -> GlobalArrays:
    """Select named global arrays for flat-EC or standard pooled routes."""
    if model_info.get("flat_ec_labels"):
        return GlobalArrays(
            x_train=data.x_train_flat,
            y_train=data.y_train_flat,
            x_val=data.x_val_flat,
            y_val=data.y_val_flat,
            x_test=data.x_test_flat,
            y_test=data.y_test_flat,
            mask_train=data.mask_train_flat,
            mask_val=data.mask_val_flat,
            mask_test=data.mask_test_flat,
        )
    return GlobalArrays(
        x_train=data.x_train,
        y_train=data.y_train,
        x_val=data.x_val,
        y_val=data.y_val,
        x_test=data.x_test,
        y_test=data.y_test,
        mask_train=data.target_mask_train,
        mask_val=data.target_mask_val,
        mask_test=data.target_mask_test,
    )


def _dataset_sizes(
    data: data_loading.PooledTaskData | data_loading.LoaderTaskData,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for any training route."""
    if isinstance(data, data_loading.LoaderTaskData):
        n_train = len(data.train_bundle.dataset)
        n_val = len(data.val_bundle.dataset) if data.val_bundle is not None else 0
        n_test = len(data.test_bundle.dataset)
        return n_train, n_val, n_test
    if data.x_train is not None:
        n_val = len(data.x_val) if data.x_val is not None else 0
        return len(data.x_train), n_val, len(data.x_test)
    if data.x_train_flat is not None:
        n_val = len(data.x_val_flat) if data.x_val_flat is not None else 0
        return len(data.x_train_flat), n_val, len(data.x_test_flat)
    raise RuntimeError("PooledTaskData contains no arrays.")


def train_global(
    *,
    model_name: str,
    emb_name: str,
    model_info: enzymado.types.ModelInfo,
    data: data_loading.PooledTaskData,
    config: data_loading.TaskConfig,
    use_validation: bool,
    run_dir: Path,
) -> object:
    """Fit an array-based model (MLP/Linear/XGB) on pooled embeddings."""
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    arrays = _select_global_arrays(model_info, data)

    if model_info.get("flat_ec_labels"):
        data_loading.log("      Training on pooled flat EC embeddings for XGBoost...")
        data_loading.log(
            "      Flat EC tensors: "
            f"x_train={arrays.x_train.shape}, "
            f"y_train={arrays.y_train.shape}, "
            f"known_train={int(arrays.mask_train.sum()):,}"
        )
    elif config.label_kind == "multilabel_ec":
        data_loading.log("      Training on pooled EC embeddings...")
    else:
        data_loading.log(
            f"      Training on pooled embeddings (train={arrays.x_train.shape})..."
        )

    trained = _fit_arrays(
        model_info=model_info,
        x_train=arrays.x_train,
        y_train=arrays.y_train,
        x_val=arrays.x_val,
        y_val=arrays.y_val,
        use_validation=use_validation,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
        target_mask_train=arrays.mask_train,
        target_mask_val=arrays.mask_val,
        unscaled_columns=data.unscaled_columns,
    )
    data_loading.log(f"      Training done ({time.time() - t0:.1f}s).")
    return trained


def train_residue(
    *,
    model_name: str,
    emb_name: str,
    model_info: enzymado.types.ModelInfo,
    data: data_loading.LoaderTaskData,
    config: data_loading.TaskConfig,
    use_validation: bool,
    run_dir: Path,
) -> residue.ResidueTrainingPayload:
    """Fit a residue-aware model using DataLoaders built from LoaderTaskData bundles."""
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    loaders = residue.build_residue_aware_loaders(
        model_info,
        train_bundle=data.train_bundle,
        val_bundle=data.val_bundle,
        test_bundle=data.test_bundle,
        use_validation=use_validation,
    )
    data_loading.log("      Training (residue-aware)...")
    trained = residue.train_residue_aware_model(
        model_info,
        loaders,
        batch_kind=config.batch_kind,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
    )
    data_loading.log(f"      Training done ({time.time() - t0:.1f}s).")
    return residue.ResidueTrainingPayload(trained=trained, loaders=loaders)


def train_attention(
    *,
    model_name: str,
    emb_name: str,
    model_info: enzymado.types.ModelInfo,
    data: data_loading.LoaderTaskData,
    config: data_loading.TaskConfig,
    use_validation: bool,
    run_dir: Path,
) -> object:
    """Fit an attention-pooling model using loaders built from task bundles."""
    batch_size = int(model_info.get("loader_batch_size", 16))
    num_workers = int(model_info.get("loader_num_workers", 4))
    loader_seed = int(model_info.get("loader_seed", 0))
    checkpoint_path = run_dir / "training_checkpoint.bin"
    train_loader = data_loading.get_dataloader(
        data.train_bundle,
        batch_size=batch_size,
        shuffle=True,
        seed=loader_seed,
        num_workers=num_workers,
    )
    val_loader = (
        data_loading.get_dataloader(
            data.val_bundle,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if use_validation and data.val_bundle is not None
        else None
    )
    data_loading.log("      Training attention model...")
    t0 = time.time()
    trained = model_registry.train_model_loader(
        model_info,
        train_loader,
        config.batch_kind,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
    )
    data_loading.log(f"      Training done ({time.time() - t0:.1f}s).")
    return trained


def prepare_attention(
    config: data_loading.TaskConfig,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    *,
    allowed_ids: set[str] | None = None,
    sequence_to_id: dict[str, str] | None = None,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
) -> data_loading.LoaderTaskData:
    """Thin wrapper: load data as LoaderTaskData for attention models on any batch_kind.

    Attention models always need per-residue DataLoaders regardless of the
    task's batch_kind. ``residue_store_mode`` is forwarded so per-residue tasks
    materialize the embedding subset into RAM when it fits. Delegates to
    enzymado.data_loading.load_loader_task_data.
    """
    return data_loading.load_loader_task_data(
        task_name,
        emb_path,
        split_variant,
        allowed_ids=allowed_ids,
        sequence_to_id=sequence_to_id,
        residue_store_mode=residue_store_mode,
        embedding_dim=embedding_dim,
    )


@dataclass(frozen=True)
class TrainingRouteSpec:
    """Bundle of plain functions describing one training route."""

    name: str
    train: collections.abc.Callable[..., object]
    prepare: collections.abc.Callable[..., data_loading.LoaderTaskData] | None = None


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


def _select_training_route(config: data_loading.TaskConfig) -> TrainingRouteSpec:
    """Resolve a config into the route spec that owns the train function."""
    if config.batch_kind == "protein_residue":
        return RESIDUE_ROUTE
    return GLOBAL_ROUTE
