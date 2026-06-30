"""Training routes: data preparation and model fitting for benchmark tasks."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from data_loading import (
    FlatECLabelBuilder,
    HierarchicalECLabelBuilder,
    build_ec_arrays,
    build_flat_ec_arrays,
    build_pooled_embedding_store,
    build_task_dataset,
    build_xy_pooled,
    get_dataloader,
    get_task_config,
    load_lazy_safetensors_embedding_store,
    load_pooled_embedding_subset,
    load_safetensors_embedding_store_subset,
    log,
)
from evaluation.artifacts import save_model, save_model_safetensors
from training.attention_wrappers import save_attention_checkpoint
from training.model_registry import get_model_map, train_model, train_model_loader
from training.predict import ResidueTrainingPayload
from training.residue import build_residue_aware_loaders, train_residue_aware_model
from training.task_rows import LoadedTaskRows


@dataclass(frozen=True)
class TaskRoute:
    """Describe which training branch a task follows inside the pipeline."""

    is_pooled_task: bool


@dataclass
class PreparedTaskData:
    """Prepared arrays or dataset bundles for one task/embedding pair."""

    route: TaskRoute
    store: Any | None = None
    train_bundle: Any | None = None
    val_bundle: Any | None = None
    test_bundle: Any | None = None
    x_train: Any | None = None
    y_train: Any | None = None
    x_val: Any | None = None
    y_val: Any | None = None
    x_test: Any | None = None
    y_test: Any | None = None
    target_mask_train: Any | None = None
    target_mask_val: Any | None = None
    target_mask_test: Any | None = None
    x_train_flat_ec: Any | None = None
    y_train_flat_ec: Any | None = None
    target_mask_train_flat_ec: Any | None = None
    x_val_flat_ec: Any | None = None
    y_val_flat_ec: Any | None = None
    target_mask_val_flat_ec: Any | None = None
    x_test_flat_ec: Any | None = None
    y_test_flat_ec: Any | None = None
    target_mask_test_flat_ec: Any | None = None


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


def _select_global_arrays(model_info: dict, prepared: PreparedTaskData) -> tuple:
    """Return (x_tr, y_tr, x_va, y_va, x_te, y_te, m_tr, m_va, m_te) for a GlobalRoute model."""
    if model_info.get("flat_ec_labels"):
        return (
            prepared.x_train_flat_ec, prepared.y_train_flat_ec,
            prepared.x_val_flat_ec, prepared.y_val_flat_ec,
            prepared.x_test_flat_ec, prepared.y_test_flat_ec,
            prepared.target_mask_train_flat_ec,
            prepared.target_mask_val_flat_ec,
            prepared.target_mask_test_flat_ec,
        )
    return (
        prepared.x_train, prepared.y_train,
        prepared.x_val, prepared.y_val,
        prepared.x_test, prepared.y_test,
        prepared.target_mask_train,
        prepared.target_mask_val,
        prepared.target_mask_test,
    )


def prepare_global(
    config,
    registry_name: str,
    emb_path,
    split_variant: str,
    task_rows: LoadedTaskRows,
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
    needs_hierarchical_ec: bool = True,
    needs_flat_ec: bool = False,
) -> PreparedTaskData:
    """Route for all non-residue tasks: loads pooled embeddings and builds x/y arrays."""
    del residue_store_mode, embedding_dim
    prepared = PreparedTaskData(route=_resolve_task_route(config))

    log("    Loading pooled embeddings from safetensors...")
    t0 = time.time()
    by_id, by_seq = load_pooled_embedding_subset(
        emb_path, task_rows.needed_ids, task_rows.needed_sequences,
    )
    log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

    if config.label_kind == "multilabel_ec":
        log("    Fitting EC vocabulary on train rows...")
        ec_label_builder = HierarchicalECLabelBuilder(
            target_column=str(config.target_column),
            separator=config.ec_separator,
        ).fit(task_rows.train_rows)
        flat_ec_label_builder = FlatECLabelBuilder(
            target_column=str(config.target_column),
            separator=config.ec_separator,
            vocabulary_by_level=ec_label_builder.vocabulary_by_level,
        )
        if needs_hierarchical_ec:
            log("    Building pooled EC arrays...")
            (prepared.x_train, prepared.y_train, prepared.target_mask_train) = build_ec_arrays(
                task_rows.train_rows, by_id, by_seq, config, ec_label_builder
            )
            (prepared.x_val, prepared.y_val, prepared.target_mask_val) = build_ec_arrays(
                task_rows.val_rows, by_id, by_seq, config, ec_label_builder
            )
            (prepared.x_test, prepared.y_test, prepared.target_mask_test) = build_ec_arrays(
                task_rows.test_rows, by_id, by_seq, config, ec_label_builder
            )
        if needs_flat_ec:
            log("    Building pooled flat EC arrays for XGBoost...")
            (prepared.x_train_flat_ec, prepared.y_train_flat_ec, prepared.target_mask_train_flat_ec) = build_flat_ec_arrays(
                task_rows.train_rows, by_id, by_seq, config, flat_ec_label_builder
            )
            (prepared.x_val_flat_ec, prepared.y_val_flat_ec, prepared.target_mask_val_flat_ec) = build_flat_ec_arrays(
                task_rows.val_rows, by_id, by_seq, config, flat_ec_label_builder
            )
            (prepared.x_test_flat_ec, prepared.y_test_flat_ec, prepared.target_mask_test_flat_ec) = build_flat_ec_arrays(
                task_rows.test_rows, by_id, by_seq, config, flat_ec_label_builder
            )
    else:
        log("    Validating rows against embedding store...")
        store = build_pooled_embedding_store(by_id, by_seq)
        prepared.train_bundle = build_task_dataset(
            registry_name, store, split_name="train", split_variant=split_variant, missing_policy="skip",
        )
        prepared.val_bundle = build_task_dataset(
            registry_name, store, split_name="val", split_variant=split_variant, missing_policy="skip",
        )
        prepared.test_bundle = build_task_dataset(
            registry_name, store, split_name="test", split_variant=split_variant, missing_policy="skip",
        )
        log(f"    Pooling embeddings ({config.batch_kind})...")
        prepared.x_train, prepared.y_train = build_xy_pooled(prepared.train_bundle, config.batch_kind)
        if len(prepared.val_bundle.dataset) > 0:
            prepared.x_val, prepared.y_val = build_xy_pooled(prepared.val_bundle, config.batch_kind)
        prepared.x_test, prepared.y_test = build_xy_pooled(prepared.test_bundle, config.batch_kind)
    return prepared


def train_global(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    prepared: PreparedTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> Any:
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    x_tr, y_tr, x_va, y_va, _, _, m_tr, m_va, _ = _select_global_arrays(model_info, prepared)

    if model_info.get("flat_ec_labels"):
        log("      Training on pooled flat EC embeddings for XGBoost...")
        log(
            f"      Flat EC tensors: x_train={x_tr.shape}, y_train={y_tr.shape}, "
            f"known_train={int(m_tr.sum()):,}"
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
    log(f"      Training done ({time.time()-t0:.1f}s).")
    return trained


def prepare_residue(
    config,
    registry_name: str,
    emb_path,
    split_variant: str,
    task_rows: LoadedTaskRows,
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
    needs_hierarchical_ec: bool = True,
    needs_flat_ec: bool = False,
) -> PreparedTaskData:
    """Route for protein_residue tasks: lazy or materialized residue store + DataLoaders."""
    del needs_hierarchical_ec, needs_flat_ec
    prepared = PreparedTaskData(route=_resolve_task_route(config))

    if config.batch_kind == "protein_residue":
        if residue_store_mode == "materialized":
            log("    Materializing residue embedding subset into RAM...")
        elif residue_store_mode == "lazy":
            log("    Building lazy residue embedding store...")
        else:
            log("    Choosing residue embedding store strategy...")
    else:
        log("    Scanning safetensors keys for lazy store...")

    t0 = time.time()
    prepared.store, store_mode = load_task_embedding_store(
        config,
        emb_path,
        task_rows.needed_ids,
        task_rows.needed_sequences,
        residue_store_mode=residue_store_mode,
        embedding_dim=embedding_dim,
    )
    if store_mode == "materialized":
        log(
            f"    Materialized store ready ({time.time()-t0:.1f}s) — "
            f"{len(prepared.store.by_id)} ids, {len(prepared.store.by_sequence)} seqs cached"
        )
    else:
        log(
            f"    Lazy store ready ({time.time()-t0:.1f}s) — "
            f"{len(prepared.store._available_ids)} ids, {len(prepared.store._seq_to_key)} seqs indexed"
        )

    log("    Validating rows against embedding store...")
    t0 = time.time()
    prepared.train_bundle = build_task_dataset(
        registry_name, prepared.store, split_name="train", split_variant=split_variant, missing_policy="skip",
    )
    ec_vocabulary = None
    if config.label_kind == "multilabel_ec":
        ec_vocabulary = prepared.train_bundle.label_builder.vocabulary_by_level
    prepared.val_bundle = build_task_dataset(
        registry_name, prepared.store, split_name="val", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    prepared.test_bundle = build_task_dataset(
        registry_name, prepared.store, split_name="test", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    n_train = len(prepared.train_bundle.dataset)
    n_val = len(prepared.val_bundle.dataset)
    n_test = len(prepared.test_bundle.dataset)
    log(f"    Row validation done ({time.time()-t0:.1f}s) — train={n_train:,} val={n_val:,} test={n_test:,}")
    return prepared


def train_residue(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    prepared: PreparedTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> ResidueTrainingPayload:
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"
    t0 = time.time()

    loaders = build_residue_aware_loaders(
        model_info,
        train_bundle=prepared.train_bundle,
        val_bundle=prepared.val_bundle,
        test_bundle=prepared.test_bundle,
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
    log(f"      Training done ({time.time()-t0:.1f}s).")
    return ResidueTrainingPayload(trained=trained, loaders=loaders)


def prepare_attention(
    config,
    registry_name: str,
    emb_path,
    split_variant: str,
    task_rows: LoadedTaskRows,
    **kwargs,
) -> PreparedTaskData:
    """Route for attention-pooled models: lazy residue store + DataLoader-based training."""
    del kwargs
    t0 = time.time()
    store = load_lazy_safetensors_embedding_store(
        emb_path, task_rows.needed_ids, task_rows.needed_sequences
    )
    log(
        f"    Lazy safetensors store ready ({time.time()-t0:.1f}s) — "
        f"{len(store._available_ids)} ids, {len(store._seq_to_key)} seqs indexed"
    )
    t0 = time.time()
    train_bundle = build_task_dataset(
        registry_name, store, split_name="train", split_variant=split_variant, missing_policy="skip",
    )
    ec_vocabulary = (
        train_bundle.label_builder.vocabulary_by_level
        if config.label_kind == "multilabel_ec" else None
    )
    val_bundle = build_task_dataset(
        registry_name, store, split_name="val", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    test_bundle = build_task_dataset(
        registry_name, store, split_name="test", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    n_train, n_val, n_test = len(train_bundle.dataset), len(val_bundle.dataset), len(test_bundle.dataset)
    log(f"    Row validation done ({time.time()-t0:.1f}s) — train={n_train:,} val={n_val:,} test={n_test:,}")
    return PreparedTaskData(
        route=TaskRoute(is_pooled_task=False),
        store=store,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
    )


def train_attention(
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    prepared: PreparedTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> Any:
    batch_size = int(model_info.get("loader_batch_size", 16))
    num_workers = int(model_info.get("loader_num_workers", 4))
    train_loader = get_dataloader(
        prepared.train_bundle, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = (
        get_dataloader(prepared.val_bundle, batch_size=batch_size, num_workers=num_workers)
        if use_validation else None
    )
    log("      Training attention model...")
    t0 = time.time()
    trained = train_model_loader(model_info, train_loader, config.batch_kind, val_loader=val_loader)
    log(f"      Training done ({time.time()-t0:.1f}s).")
    return trained


def save_trained_model_default(trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
    return save_model(trained, run_dir)


def save_trained_model_attention(trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
    return save_attention_checkpoint(trained, run_dir, checkpoint_path=model_path)


@dataclass(frozen=True)
class TrainingRouteSpec:
    """Describe one training route as a bundle of plain functions."""

    name: str  # "global" | "residue" | "attention"
    prepare: Callable
    train: Callable
    save_trained_model: Callable


GLOBAL_ROUTE = TrainingRouteSpec(
    name="global", prepare=prepare_global, train=train_global, save_trained_model=save_trained_model_default,
)
RESIDUE_ROUTE = TrainingRouteSpec(
    name="residue", prepare=prepare_residue, train=train_residue, save_trained_model=save_trained_model_default,
)
ATTENTION_ROUTE = TrainingRouteSpec(
    name="attention", prepare=prepare_attention, train=train_attention, save_trained_model=save_trained_model_attention,
)


def load_task_embedding_store(
    config,
    emb_path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
):
    """Return the embedding store strategy appropriate for one task."""
    if config.batch_kind == "protein_residue":
        resolved_mode = residue_store_mode
        estimate_gib = _estimate_materialized_residue_store_gib(needed_sequences, embedding_dim)
        if residue_store_mode == "auto":
            available_gib = psutil.virtual_memory().available / 1024 ** 3
            safe_limit_gib = available_gib * 0.8
            if estimate_gib is not None:
                resolved_mode = "materialized" if estimate_gib <= safe_limit_gib else "lazy"
                log(
                    "    Residue store auto-selection: "
                    f"estimated={estimate_gib:.1f} GiB "
                    f"available={available_gib:.1f} GiB "
                    f"(limit={safe_limit_gib:.1f} GiB) -> {resolved_mode}"
                )
            else:
                resolved_mode = "materialized"
                log("    Residue store auto-selection: missing embedding_dim estimate -> materialized")

        if resolved_mode == "materialized":
            return load_safetensors_embedding_store_subset(emb_path, needed_ids, needed_sequences), resolved_mode
        if resolved_mode == "lazy":
            return load_lazy_safetensors_embedding_store(emb_path, needed_ids, needed_sequences), resolved_mode
        raise ValueError(
            "residue_store_mode must be one of 'auto', 'materialized', or 'lazy'. "
            f"Received '{residue_store_mode}'."
        )

    store = load_lazy_safetensors_embedding_store(emb_path, needed_ids, needed_sequences)
    return store, "lazy"


def _select_training_route(config) -> TrainingRouteSpec:
    """Resolve a config into the route spec that owns prepare and train."""
    if config.batch_kind == "protein_residue":
        return RESIDUE_ROUTE
    return GLOBAL_ROUTE


def _resolve_task_route(config) -> TaskRoute:
    return TaskRoute(is_pooled_task=config.batch_kind != "protein_residue")


def _estimate_materialized_residue_store_gib(
    needed_sequences: set[str],
    embedding_dim: int | None,
) -> float | None:
    if embedding_dim is None or embedding_dim <= 0:
        return None
    total_residues = sum(len(s) for s in needed_sequences if s)
    return (total_residues * int(embedding_dim) * 4) / float(1024 ** 3)


def _dataset_sizes(prepared: PreparedTaskData) -> tuple[int, int, int]:
    """Return train/val/test sizes independent of the chosen training route."""
    if prepared.train_bundle is not None:
        return (
            len(prepared.train_bundle.dataset),
            len(prepared.val_bundle.dataset),
            len(prepared.test_bundle.dataset),
        )
    if prepared.x_train is not None:
        return len(prepared.x_train), len(prepared.x_val) if prepared.x_val is not None else 0, len(prepared.x_test)
    if prepared.x_train_flat_ec is not None:
        return len(prepared.x_train_flat_ec), len(prepared.x_val_flat_ec), len(prepared.x_test_flat_ec)
    raise RuntimeError("PreparedTaskData has no data.")


def _cleanup_prepared_task_data(prepared: PreparedTaskData | None) -> None:
    """Release references to per-task embedding stores before the next task."""
    if prepared is not None:
        prepared.store = None
