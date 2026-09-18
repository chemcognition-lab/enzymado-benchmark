"""Single load_task_data() entry point returning PooledTaskData or LoaderTaskData.

CLI example:
    cd /home/josechemistry/GitHub/enzymado-benchmark && conda run -n chem python -c "
    import sys; sys.path.insert(0, 'src')
    from data_loading import load_task_data, PooledTaskData, LoaderTaskData
    print('task_data imports OK')
    "
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from .data_registry import TaskConfig, TaskType, get_task_config
from .datasets import TaskDatasetBundle, build_task_dataset
from .labels import FlatECLabelBuilder, HierarchicalECLabelBuilder
from .stores import (
    build_ec_arrays,
    build_flat_ec_arrays,
    build_pooled_embedding_store,
    build_xy_pooled,
    collect_needed_lookup_keys,
    filter_rows_to_allowed_ids,
    get_task_rows,
    load_lazy_safetensors_embedding_store,
    load_pooled_embedding_subset,
    load_safetensors_embedding_store_subset,
    log,
)


def _log_skipped_rows(split_name: str, bundle: TaskDatasetBundle) -> None:
    """Report rows dropped by ``missing_policy="skip"`` so shrinkage is never silent.

    Rows failing validation (unresolvable embedding, missing or unparseable
    SMILES, unusable label) are dropped rather than raised on. Without this
    log a typo'd SMILES or a sequence missing from the embedding store simply
    yields a smaller dataset and a plausible-looking metric.
    """
    skipped = getattr(bundle.dataset, "skipped_rows", 0)
    if not skipped:
        return
    loaded = len(bundle.rows)
    log(
        f"    WARNING: {split_name} dropped {skipped:,} of {loaded:,} rows"
        f" failing validation ({len(bundle.dataset):,} kept)."
    )
    for reason in getattr(bundle.dataset, "skipped_reasons", []):
        log(f"      - {reason}")
    if skipped > len(getattr(bundle.dataset, "skipped_reasons", [])):
        log("      - (further reasons not shown)")


@dataclass
class PooledTaskData:
    """Output for global route (MLP/Linear/XGB). Arrays ready for .fit(X, y)."""

    config: TaskConfig
    x_train: torch.Tensor | None
    y_train: torch.Tensor | dict[str, torch.Tensor] | None
    x_val: torch.Tensor | None
    y_val: torch.Tensor | dict[str, torch.Tensor] | None
    x_test: torch.Tensor | None
    y_test: torch.Tensor | dict[str, torch.Tensor] | None
    target_mask_train: torch.Tensor | dict[str, torch.Tensor] | None
    target_mask_val: torch.Tensor | dict[str, torch.Tensor] | None
    target_mask_test: torch.Tensor | dict[str, torch.Tensor] | None
    # EC flat arrays (only populated when config.label_kind == "multilabel_ec")
    x_train_flat: torch.Tensor | None
    y_train_flat: torch.Tensor | None
    mask_train_flat: torch.Tensor | None
    x_val_flat: torch.Tensor | None
    y_val_flat: torch.Tensor | None
    mask_val_flat: torch.Tensor | None
    x_test_flat: torch.Tensor | None
    y_test_flat: torch.Tensor | None
    mask_test_flat: torch.Tensor | None
    cleanup: Callable[[], None]


@dataclass
class LoaderTaskData:
    """Output for residue/attention route. Bundles ready for get_dataloader()."""

    config: TaskConfig
    train_bundle: TaskDatasetBundle
    val_bundle: TaskDatasetBundle | None
    test_bundle: TaskDatasetBundle
    cleanup: Callable[[], None]


@dataclass
class _TaskRows:
    """Internal: CSV rows for all splits plus lookup keys."""

    train_rows: list[dict]
    val_rows: list[dict]
    test_rows: list[dict]
    needed_ids: set[str]
    needed_sequences: set[str]


def _load_and_filter_rows(
    config: TaskConfig,
    split_variant: str,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
) -> _TaskRows:
    """Load CSV rows for all splits, apply allowed-id filter, collect lookup keys."""
    log("    Loading CSV rows...")
    train_rows = get_task_rows(config, "train", split_variant)
    val_rows = get_task_rows(config, "val", split_variant)
    test_rows = get_task_rows(config, "test", split_variant)

    if allowed_ids is not None:
        orig_train = len(train_rows)
        orig_val = len(val_rows)
        orig_test = len(test_rows)
        train_rows = filter_rows_to_allowed_ids(train_rows, config, allowed_ids, sequence_to_id)
        val_rows = filter_rows_to_allowed_ids(val_rows, config, allowed_ids, sequence_to_id)
        test_rows = filter_rows_to_allowed_ids(test_rows, config, allowed_ids, sequence_to_id)
        log(
            "    Applied allowed-id filter:"
            f" train={len(train_rows):,}/{orig_train:,}"
            f" val={len(val_rows):,}/{orig_val:,}"
            f" test={len(test_rows):,}/{orig_test:,}"
        )

    needed_ids, needed_sequences = collect_needed_lookup_keys(
        train_rows + val_rows + test_rows, config
    )
    log(
        f"    Rows loaded: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}"
        f" | need {len(needed_ids)} ids, {len(needed_sequences)} seqs"
    )
    return _TaskRows(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        needed_ids=needed_ids,
        needed_sequences=needed_sequences,
    )


def _build_pooled_data(
    config: TaskConfig,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    rows: _TaskRows,
    embedding_dim: int | None,  # unused; accepted for uniform call-site signature
) -> PooledTaskData:
    """Build pooled x/y arrays for the global training route (MLP/Linear/XGB)."""
    del embedding_dim
    log("    Loading pooled embeddings from safetensors...")
    t0 = time.time()
    by_id, by_seq = load_pooled_embedding_subset(
        emb_path, rows.needed_ids, rows.needed_sequences
    )
    log(
        f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings"
        f" ({time.time() - t0:.1f}s)"
    )

    if config.label_kind == "multilabel_ec":
        log("    Fitting EC vocabulary on train rows...")
        ec_builder = HierarchicalECLabelBuilder(
            target_column=str(config.target_column),
            separator=config.ec_separator,
        ).fit(rows.train_rows)
        flat_builder = FlatECLabelBuilder(
            target_column=str(config.target_column),
            separator=config.ec_separator,
            vocabulary_by_level=ec_builder.vocabulary_by_level,
        )

        log("    Building pooled hierarchical EC arrays...")
        x_tr, y_tr, m_tr = build_ec_arrays(rows.train_rows, by_id, by_seq, config, ec_builder)
        x_va, y_va, m_va = build_ec_arrays(rows.val_rows, by_id, by_seq, config, ec_builder)
        x_te, y_te, m_te = build_ec_arrays(rows.test_rows, by_id, by_seq, config, ec_builder)

        log("    Building pooled flat EC arrays for XGBoost...")
        xf_tr, yf_tr, mf_tr = build_flat_ec_arrays(
            rows.train_rows, by_id, by_seq, config, flat_builder
        )
        xf_va, yf_va, mf_va = build_flat_ec_arrays(
            rows.val_rows, by_id, by_seq, config, flat_builder
        )
        xf_te, yf_te, mf_te = build_flat_ec_arrays(
            rows.test_rows, by_id, by_seq, config, flat_builder
        )

        return PooledTaskData(
            config=config,
            x_train=x_tr, y_train=y_tr,
            x_val=x_va if len(x_va) > 0 else None, y_val=y_va if len(x_va) > 0 else None,
            x_test=x_te, y_test=y_te,
            target_mask_train=m_tr,
            target_mask_val=m_va if len(x_va) > 0 else None,
            target_mask_test=m_te,
            x_train_flat=xf_tr, y_train_flat=yf_tr, mask_train_flat=mf_tr,
            x_val_flat=xf_va if len(xf_va) > 0 else None,
            y_val_flat=yf_va if len(xf_va) > 0 else None,
            mask_val_flat=mf_va if len(xf_va) > 0 else None,
            x_test_flat=xf_te, y_test_flat=yf_te, mask_test_flat=mf_te,
            cleanup=lambda: None,
        )

    # Non-EC: build datasets, pool embeddings into tensors, discard intermediate store
    log("    Validating rows against embedding store...")
    store = build_pooled_embedding_store(by_id, by_seq)
    train_bundle = build_task_dataset(
        task_name, store, split_name="train", split_variant=split_variant,
        missing_policy="skip",
    )
    val_bundle = build_task_dataset(
        task_name, store, split_name="val", split_variant=split_variant,
        missing_policy="skip",
    )
    test_bundle = build_task_dataset(
        task_name, store, split_name="test", split_variant=split_variant,
        missing_policy="skip",
    )
    for split_name, bundle in (
        ("train", train_bundle), ("val", val_bundle), ("test", test_bundle)
    ):
        _log_skipped_rows(split_name, bundle)
    log(
        f"    Row validation done — train={len(train_bundle.dataset):,}"
        f" val={len(val_bundle.dataset):,} test={len(test_bundle.dataset):,}"
    )
    log(f"    Pooling embeddings ({config.batch_kind})...")
    x_tr2, y_tr2 = build_xy_pooled(train_bundle, config.batch_kind)
    x_va2: torch.Tensor | None = None
    y_va2: torch.Tensor | None = None
    if len(val_bundle.dataset) > 0:
        x_va2, y_va2 = build_xy_pooled(val_bundle, config.batch_kind)
    x_te2, y_te2 = build_xy_pooled(test_bundle, config.batch_kind)

    return PooledTaskData(
        config=config,
        x_train=x_tr2, y_train=y_tr2,
        x_val=x_va2, y_val=y_va2,
        x_test=x_te2, y_test=y_te2,
        target_mask_train=None, target_mask_val=None, target_mask_test=None,
        x_train_flat=None, y_train_flat=None, mask_train_flat=None,
        x_val_flat=None, y_val_flat=None, mask_val_flat=None,
        x_test_flat=None, y_test_flat=None, mask_test_flat=None,
        cleanup=lambda: None,
    )


def _estimate_residue_store_gib(
    needed_sequences: set[str],
    embedding_dim: int | None,
) -> float | None:
    """Estimate GiB for a fully materialized residue embedding store."""
    if embedding_dim is None or embedding_dim <= 0:
        return None
    total_residues = sum(len(s) for s in needed_sequences if s)
    return (total_residues * int(embedding_dim) * 4) / float(1024 ** 3)


def _build_loader_data(
    config: TaskConfig,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    rows: _TaskRows,
    residue_store_mode: str,
    embedding_dim: int | None,
    sequence_to_id: dict[str, str],
) -> LoaderTaskData:
    """Build TaskDatasetBundle(s) for the residue or attention training route."""
    if config.batch_kind == TaskType.PROTEIN_RESIDUE:
        resolved_mode = residue_store_mode
        if residue_store_mode == "auto":
            available_gib = psutil.virtual_memory().available / 1024 ** 3
            safe_limit_gib = available_gib * 0.8
            estimate_gib = _estimate_residue_store_gib(rows.needed_sequences, embedding_dim)
            if estimate_gib is not None:
                resolved_mode = "materialized" if estimate_gib <= safe_limit_gib else "lazy"
                log(
                    "    Residue store auto-selection:"
                    f" estimated={estimate_gib:.1f} GiB"
                    f" available={available_gib:.1f} GiB"
                    f" (limit={safe_limit_gib:.1f} GiB) -> {resolved_mode}"
                )
            else:
                resolved_mode = "materialized"
                log("    Residue store auto-selection: missing embedding_dim estimate -> materialized")

        t0 = time.time()
        if resolved_mode == "materialized":
            log("    Materializing residue embedding subset into RAM...")
            store = load_safetensors_embedding_store_subset(
                emb_path, rows.needed_ids, rows.needed_sequences
            )
            log(
                f"    Materialized store ready ({time.time() - t0:.1f}s) —"
                f" {len(store.by_id)} ids, {len(store.by_sequence)} seqs cached"
            )
        elif resolved_mode == "lazy":
            log("    Building lazy residue embedding store...")
            store = load_lazy_safetensors_embedding_store(
                emb_path, rows.needed_ids, rows.needed_sequences
            )
            log(
                f"    Lazy store ready ({time.time() - t0:.1f}s) —"
                f" {len(store._available_ids)} ids, {len(store._seq_to_key)} seqs indexed"
            )
        else:
            raise ValueError(
                "residue_store_mode must be one of 'auto', 'materialized', or 'lazy'."
                f" Received '{residue_store_mode}'."
            )
    else:
        # Attention route: always use lazy store regardless of batch_kind
        log("    Building lazy safetensors store (attention route)...")
        t0 = time.time()
        store = load_lazy_safetensors_embedding_store(
            emb_path, rows.needed_ids, rows.needed_sequences
        )
        log(
            f"    Lazy safetensors store ready ({time.time() - t0:.1f}s) —"
            f" {len(store._available_ids)} ids, {len(store._seq_to_key)} seqs indexed"
        )

    log("    Validating rows against embedding store...")
    t0 = time.time()
    train_bundle = build_task_dataset(
        task_name, store, split_name="train", split_variant=split_variant,
        missing_policy="skip",
    )
    ec_vocabulary = (
        train_bundle.label_builder.vocabulary_by_level
        if config.label_kind == "multilabel_ec" else None
    )
    raw_val_bundle = build_task_dataset(
        task_name, store, split_name="val", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    val_bundle: TaskDatasetBundle | None = raw_val_bundle if len(raw_val_bundle.dataset) > 0 else None
    test_bundle = build_task_dataset(
        task_name, store, split_name="test", split_variant=split_variant,
        missing_policy="skip", ec_vocabulary=ec_vocabulary,
    )
    for split_name, bundle in (
        ("train", train_bundle), ("val", raw_val_bundle), ("test", test_bundle)
    ):
        _log_skipped_rows(split_name, bundle)
    n_train = len(train_bundle.dataset)
    n_val = len(val_bundle.dataset) if val_bundle is not None else 0
    n_test = len(test_bundle.dataset)
    log(
        f"    Row validation done ({time.time() - t0:.1f}s) —"
        f" train={n_train:,} val={n_val:,} test={n_test:,}"
    )

    store_ref = [store]  # closure captures a mutable container so cleanup can release

    def cleanup() -> None:
        store_ref[0] = None

    return LoaderTaskData(
        config=config,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        cleanup=cleanup,
    )


def load_task_data(
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    *,
    allowed_ids: set[str] | None = None,
    sequence_to_id: dict[str, str] | None = None,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
) -> PooledTaskData | LoaderTaskData:
    """Load all data for one task/embedding pair.

    Returns PooledTaskData (x/y arrays) for global MLP/Linear/XGB routes,
    or LoaderTaskData (TaskDatasetBundle) for residue DataLoader routes.
    Dispatches on config.batch_kind: protein_residue -> LoaderTaskData, else PooledTaskData.
    """
    config = get_task_config(task_name)
    seq_to_id = sequence_to_id or {}
    rows = _load_and_filter_rows(config, split_variant, allowed_ids, seq_to_id)

    if config.batch_kind == TaskType.PROTEIN_RESIDUE:
        return _build_loader_data(
            config, task_name, emb_path, split_variant, rows,
            residue_store_mode, embedding_dim, seq_to_id,
        )
    return _build_pooled_data(
        config, task_name, emb_path, split_variant, rows, embedding_dim
    )


def load_loader_task_data(
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    *,
    allowed_ids: set[str] | None = None,
    sequence_to_id: dict[str, str] | None = None,
    embedding_dim: int | None = None,
) -> LoaderTaskData:
    """Load data as LoaderTaskData regardless of batch_kind.

    Used by the attention route, which needs per-residue DataLoaders even for
    tasks whose batch_kind would normally select the pooled-arrays path.
    Always uses a lazy embedding store.
    """
    config = get_task_config(task_name)
    seq_to_id = sequence_to_id or {}
    rows = _load_and_filter_rows(config, split_variant, allowed_ids, seq_to_id)
    return _build_loader_data(
        config, task_name, emb_path, split_variant, rows,
        "lazy", embedding_dim, seq_to_id,
    )


__all__ = [
    "LoaderTaskData",
    "PooledTaskData",
    "load_loader_task_data",
    "load_task_data",
]
