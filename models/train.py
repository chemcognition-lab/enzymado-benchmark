"""Train benchmark models across embedding sets and registered benchmark tasks."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import (
    FlatECLabelBuilder,
    HierarchicalECLabelBuilder,
    build_task_dataset,
    get_task_config,
)

from models.train_common import (
    SKIP_LABEL_KINDS,
    candidate_embedding_keys,
    build_result_run_dir,
    discover_embedding_names,
    filter_rows_to_allowed_ids,
    is_run_complete,
    load_allowed_ids,
    load_sequence_to_id_index,
    log,
    resolve_registry_name,
    resolve_sequence_index_path,
    resolve_split_variant,
    row_matches_allowed_ids,
)
from models.results_card import (
    build_artifact_stem,
    build_model_artifact_path,
    build_prediction_artifact_path,
    ensure_results_card,
    find_result_entry,
    is_result_entry_complete,
    read_results_card,
    update_results_card_locked,
)
from models.train_data import (
    LazyResidueEmbeddingStore,
    build_Xy_ligand_direct,
    build_Xy_pooled,
    build_flat_multilabel_ec_direct,
    build_multilabel_ec_direct,
    build_mutation_delta_Xy_direct,
    build_protein_global_Xy_direct,
    collect_needed_lookup_keys,
    get_dataloader,
    get_target_payload,
    get_task_rows,
    load_embedding_store_subset,
    load_lazy_embedding_store,
    load_pooled_h5_subset,
)
from models.train_output import (
    make_predictions_file,
    make_result_entry,
    read_json,
    save_model,
)
from models.train_runtime import (
    get_model_map,
    train_model,
)
from models.training.residue import (
    build_residue_aware_loaders,
    collect_residue_aware_outputs,
    save_residue_aware_predictions,
    train_residue_aware_model,
)
from models.training.task_rows import (
    LoadedTaskRows,
    load_task_rows_bundle,
    log_loaded_data_summary,
)


def load_task_embedding_store(
    config,
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
):
    """Return the embedding store strategy appropriate for one task.

    This wrapper intentionally lives in ``models.train`` so test monkeypatches
    against ``models.train.load_embedding_store_subset`` and
    ``models.train.load_lazy_embedding_store`` keep working.
    """
    if config.batch_kind == "protein_residue":
        resolved_mode = residue_store_mode
        estimate_gib = _estimate_materialized_residue_store_gib(
            needed_sequences,
            embedding_dim,
        )
        if residue_store_mode == "auto":
            auto_max_gib = float(os.getenv("ENZYMADO_RESIDUE_STORE_AUTO_MAX_GIB", "64"))
            if estimate_gib is not None and estimate_gib > auto_max_gib:
                resolved_mode = "lazy"
            else:
                resolved_mode = "materialized"
            if estimate_gib is not None:
                log(
                    "    Residue store auto-selection: "
                    f"estimated materialized cache={estimate_gib:.1f} GiB "
                    f"(threshold={auto_max_gib:.1f} GiB) -> {resolved_mode}"
                )
            else:
                log(
                    "    Residue store auto-selection: missing embedding_dim estimate "
                    f"-> {resolved_mode}"
                )

        if resolved_mode == "materialized":
            store = load_embedding_store_subset(h5_path, needed_ids, needed_sequences)
            return store, resolved_mode
        if resolved_mode == "lazy":
            store = load_lazy_embedding_store(h5_path, needed_ids, needed_sequences)
            return store, resolved_mode
        raise ValueError(
            "residue_store_mode must be one of 'auto', 'materialized', or 'lazy'. "
            f"Received '{residue_store_mode}'."
        )

    store = load_lazy_embedding_store(h5_path, needed_ids, needed_sequences)
    return store, "lazy"


@dataclass(frozen=True)
class TaskRoute:
    """Describe which training path a task follows inside the main pipeline."""

    is_ligand_task: bool
    is_pooled_ec_task: bool
    is_pooled_protein_global_task: bool
    is_pooled_mutation_task: bool

    @property
    def uses_direct_arrays(self) -> bool:
        return any(
            (
                self.is_ligand_task,
                self.is_pooled_ec_task,
                self.is_pooled_protein_global_task,
                self.is_pooled_mutation_task,
            )
        )


@dataclass
class PreparedTaskData:
    """Prepared arrays or dataset bundles for one task/embedding pair."""

    route: TaskRoute
    store: Any | None = None
    train_bundle: Any | None = None
    val_bundle: Any | None = None
    test_bundle: Any | None = None
    X_train: Any | None = None
    y_train: Any | None = None
    X_val: Any | None = None
    y_val: Any | None = None
    X_test: Any | None = None
    y_test: Any | None = None
    target_mask_train: Any | None = None
    target_mask_val: Any | None = None
    target_mask_test: Any | None = None
    X_train_flat_ec: Any | None = None
    y_train_flat_ec: Any | None = None
    target_mask_train_flat_ec: Any | None = None
    X_val_flat_ec: Any | None = None
    y_val_flat_ec: Any | None = None
    target_mask_val_flat_ec: Any | None = None
    X_test_flat_ec: Any | None = None
    y_test_flat_ec: Any | None = None
    target_mask_test_flat_ec: Any | None = None


@dataclass
class ModelRunOutputs:
    """Predictions and targets produced by training one model on one task."""

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


def _estimate_materialized_residue_store_gib(
    needed_sequences: set[str],
    embedding_dim: int | None,
) -> float | None:
    """Estimate RAM needed to cache one residue embedding array per unique sequence."""
    if embedding_dim is None or embedding_dim <= 0:
        return None
    total_residues = sum(len(sequence) for sequence in needed_sequences if sequence)
    bytes_per_float32 = 4
    return (total_residues * int(embedding_dim) * bytes_per_float32) / float(1024**3)


def _resolve_task_route(config) -> TaskRoute:
    """Map the task config onto one of the training branches."""
    return TaskRoute(
        is_ligand_task=config.batch_kind == "protein_ligand_global",
        is_pooled_ec_task=config.label_kind == "multilabel_ec",
        is_pooled_protein_global_task=(
            config.batch_kind == "protein_global"
            and config.label_kind in {"scalar_regression", "binary"}
        ),
        is_pooled_mutation_task=(
            config.batch_kind == "mutation_delta_global"
            and config.label_kind in {"scalar_regression", "binary"}
        ),
    )


def _select_model_map(
    task_type: str,
    label_kind: str,
    model_prefix_filter: str | None,
) -> dict[str, dict]:
    """Select and optionally filter the registered models for one task."""
    model_map = get_model_map(task_type, label_kind)
    if model_prefix_filter is None:
        return model_map
    return {
        name: info
        for name, info in model_map.items()
        if name.startswith(model_prefix_filter)
    }


def _completed_models(
    model_map: dict[str, dict],
    output_dir: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    results_card: dict[str, Any] | None = None,
) -> list[str]:
    """Return which candidate models already have finished artifacts on disk."""
    return [
        model_name for model_name in model_map
        if is_run_complete(
            build_result_run_dir(output_dir, split_variant, task_name, emb_name, model_name)
        )
        or is_result_entry_complete(
            find_result_entry(
                results_card or {},
                task_scope="properties",
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
            )
        )
    ]


def _prepare_task_data(
    config,
    registry_name: str,
    emb_path: str | Path,
    split_variant: str,
    task_rows: LoadedTaskRows,
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
    output_dir: Path = None,
    task_name: str = None,
    emb_name: str = None,
) -> PreparedTaskData:
    """Prepare arrays or bundles for the training route chosen by the task config."""
    route = _resolve_task_route(config)
    prepared = PreparedTaskData(route=route)

    # Setup Cache Path
    cache_file = None
    if route.is_ligand_task or route.is_pooled_protein_global_task or route.is_pooled_mutation_task:
        if output_dir and task_name and emb_name:
            cache_dir = output_dir.parent / ".data_cache" / task_name / emb_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{split_variant}_arrays.npz"

            # Check for Cache
            if cache_file.exists():
                try:
                    log(f"    [CACHE] Found pre-built arrays at {cache_file}!")
                    t0 = time.time()
                    with np.load(cache_file) as data:
                        prepared.X_train = data['X_train'].copy()
                        prepared.y_train = data['y_train'].copy()
                        prepared.X_test = data['X_test'].copy()
                        prepared.y_test = data['y_test'].copy()
                        if 'X_val' in data:
                            prepared.X_val = data['X_val'].copy()
                            prepared.y_val = data['y_val'].copy()
                    log(f"    [CACHE] Fast-loaded dataset in {time.time()-t0:.1f}s")
                    return prepared
                except Exception as e:
                    log(f"    [CACHE ERROR] Failed to load cache (corrupted?): {e}. Rebuilding...")

    # Helper function for safe, atomic saving
    def _save_cache_safely():
        if cache_file and not cache_file.exists():
            log(f"    [CACHE] Saving arrays to disk using atomic write...")
            save_dict = {
                'X_train': prepared.X_train, 'y_train': prepared.y_train,
                'X_test': prepared.X_test, 'y_test': prepared.y_test
            }
            if prepared.X_val is not None:
                save_dict['X_val'] = prepared.X_val
                save_dict['y_val'] = prepared.y_val
            
            # Save to .tmp first to prevent corruptions
            temp_file = cache_file.with_name(cache_file.stem + "_tmp.npz")
            np.savez_compressed(temp_file, **save_dict)
            temp_file.replace(cache_file)

    if route.is_ligand_task:
        log("    Loading pooled embeddings from H5 (ligand task)...")
        t0 = time.time()
        by_id, by_seq = load_pooled_h5_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
        )
        log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

        log("    Building X/y arrays...")
        prepared.X_train, prepared.y_train = build_Xy_ligand_direct(
            task_rows.train_rows, by_id, by_seq, config
        )
        prepared.X_val, prepared.y_val = build_Xy_ligand_direct(
            task_rows.val_rows, by_id, by_seq, config
        )
        prepared.X_test, prepared.y_test = build_Xy_ligand_direct(
            task_rows.test_rows, by_id, by_seq, config
        )

        _save_cache_safely()

        return prepared

    if route.is_pooled_ec_task:
        log("    Loading pooled embeddings from H5 (EC task)...")
        t0 = time.time()
        by_id, by_seq = load_pooled_h5_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
        )
        log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

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

        log("    Building pooled EC arrays...")
        prepared.X_train, prepared.y_train, prepared.target_mask_train = build_multilabel_ec_direct(
            task_rows.train_rows, by_id, by_seq, config, ec_label_builder
        )
        prepared.X_val, prepared.y_val, prepared.target_mask_val = build_multilabel_ec_direct(
            task_rows.val_rows, by_id, by_seq, config, ec_label_builder
        )
        prepared.X_test, prepared.y_test, prepared.target_mask_test = build_multilabel_ec_direct(
            task_rows.test_rows, by_id, by_seq, config, ec_label_builder
        )

        log("    Building pooled flat EC arrays for XGBoost...")
        (
            prepared.X_train_flat_ec,
            prepared.y_train_flat_ec,
            prepared.target_mask_train_flat_ec,
        ) = build_flat_multilabel_ec_direct(
            task_rows.train_rows, by_id, by_seq, config, flat_ec_label_builder
        )
        (
            prepared.X_val_flat_ec,
            prepared.y_val_flat_ec,
            prepared.target_mask_val_flat_ec,
        ) = build_flat_multilabel_ec_direct(
            task_rows.val_rows, by_id, by_seq, config, flat_ec_label_builder
        )
        (
            prepared.X_test_flat_ec,
            prepared.y_test_flat_ec,
            prepared.target_mask_test_flat_ec,
        ) = build_flat_multilabel_ec_direct(
            task_rows.test_rows, by_id, by_seq, config, flat_ec_label_builder
        )
        return prepared

    if route.is_pooled_protein_global_task:
        log("    Loading pooled embeddings from H5 (protein-global task)...")
        t0 = time.time()
        by_id, by_seq = load_pooled_h5_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
        )
        log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

        log("    Building pooled X/y arrays...")
        prepared.X_train, prepared.y_train = build_protein_global_Xy_direct(
            task_rows.train_rows, by_id, by_seq, config
        )
        prepared.X_val, prepared.y_val = build_protein_global_Xy_direct(
            task_rows.val_rows, by_id, by_seq, config
        )
        prepared.X_test, prepared.y_test = build_protein_global_Xy_direct(
            task_rows.test_rows, by_id, by_seq, config
        )

        _save_cache_safely()

        return prepared

    if route.is_pooled_mutation_task:
        log("    Loading pooled embeddings from H5 (mutation-delta task)...")
        t0 = time.time()
        by_id, by_seq = load_pooled_h5_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
        )
        log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

        log("    Building pooled mutation-delta X/y arrays...")
        prepared.X_train, prepared.y_train = build_mutation_delta_Xy_direct(
            task_rows.train_rows, by_id, by_seq, config
        )
        prepared.X_val, prepared.y_val = build_mutation_delta_Xy_direct(
            task_rows.val_rows, by_id, by_seq, config
        )
        prepared.X_test, prepared.y_test = build_mutation_delta_Xy_direct(
            task_rows.test_rows, by_id, by_seq, config
        )

        _save_cache_safely()

        return prepared

    if config.batch_kind == "protein_residue":
        if residue_store_mode == "materialized":
            log("    Materializing residue embedding subset into RAM...")
        elif residue_store_mode == "lazy":
            log("    Building lazy residue embedding store...")
        else:
            log("    Choosing residue embedding store strategy...")
    else:
        log("    Scanning H5 keys for lazy store...")

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
            f"{len(prepared.store._available_ids)} ids, {len(prepared.store._seq_to_h5key)} seqs indexed"
        )

    log("    Building task datasets...")
    t0 = time.time()
    prepared.train_bundle = build_task_dataset(
        registry_name,
        prepared.store,
        split_name="train",
        split_variant=split_variant,
        missing_policy="skip",
    )
    ec_vocabulary = None
    if config.label_kind == "multilabel_ec":
        ec_vocabulary = prepared.train_bundle.label_builder.vocabulary_by_level
    prepared.val_bundle = build_task_dataset(
        registry_name,
        prepared.store,
        split_name="val",
        split_variant=split_variant,
        missing_policy="skip",
        ec_vocabulary=ec_vocabulary,
    )
    prepared.test_bundle = build_task_dataset(
        registry_name,
        prepared.store,
        split_name="test",
        split_variant=split_variant,
        missing_policy="skip",
        ec_vocabulary=ec_vocabulary,
    )
    log(f"    Datasets built ({time.time()-t0:.1f}s)")
    return prepared


def _dataset_sizes(prepared: PreparedTaskData) -> tuple[int, int, int]:
    """Return train/val/test sizes independent of the chosen training route."""
    if prepared.route.uses_direct_arrays:
        return len(prepared.X_train), len(prepared.X_val), len(prepared.X_test)
    return (
        len(prepared.train_bundle.dataset),
        len(prepared.val_bundle.dataset),
        len(prepared.test_bundle.dataset),
    )


def _cleanup_prepared_task_data(prepared: PreparedTaskData | None) -> None:
    """Release references to per-task embedding stores before the next task."""
    if prepared is not None:
        prepared.store = None


def _is_residue_aware_training(model_info: dict, prepared: PreparedTaskData) -> bool:
    """Return whether this model should train through residue-aware loaders."""
    return model_info.get("residue_aware", False) and not prepared.route.uses_direct_arrays


def _train_one_model(
    model_name: str,
    emb_name: str,
    model_info: dict,
    prepared: PreparedTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
) -> ModelRunOutputs:
    """Run one model on the prepared task data and collect outputs for saving."""
    route = prepared.route
    t0 = time.time()
    progress_label = f"{emb_name}/{config.name}/{model_name}"
    checkpoint_path = run_dir / "training_checkpoint.bin"

    if route.is_ligand_task:
        log("      Training...")
        trained = train_model(
            model_info,
            prepared.X_train,
            prepared.y_train,
            X_val=prepared.X_val if use_validation else None,
            y_val=prepared.y_val if use_validation else None,
            progress_label=progress_label,
            checkpoint_path=checkpoint_path,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return ModelRunOutputs(
            trained=trained,
            pred_train=trained.predict(prepared.X_train),
            pred_val=trained.predict(prepared.X_val) if use_validation else None,
            pred_test=trained.predict(prepared.X_test),
            y_train=prepared.y_train,
            y_val=prepared.y_val if use_validation else None,
            y_test=prepared.y_test,
        )

    if route.is_pooled_protein_global_task:
        log("      Training on pooled protein-global embeddings...")
        trained = train_model(
            model_info,
            prepared.X_train,
            prepared.y_train,
            X_val=prepared.X_val if use_validation else None,
            y_val=prepared.y_val if use_validation else None,
            progress_label=progress_label,
            checkpoint_path=checkpoint_path,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return ModelRunOutputs(
            trained=trained,
            pred_train=trained.predict(prepared.X_train),
            pred_val=trained.predict(prepared.X_val) if use_validation else None,
            pred_test=trained.predict(prepared.X_test),
            y_train=prepared.y_train,
            y_val=prepared.y_val if use_validation else None,
            y_test=prepared.y_test,
        )

    if route.is_pooled_mutation_task:
        log("      Training on pooled mutation-delta embeddings...")
        trained = train_model(
            model_info,
            prepared.X_train,
            prepared.y_train,
            X_val=prepared.X_val if use_validation else None,
            y_val=prepared.y_val if use_validation else None,
            progress_label=progress_label,
            checkpoint_path=checkpoint_path,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return ModelRunOutputs(
            trained=trained,
            pred_train=trained.predict(prepared.X_train),
            pred_val=trained.predict(prepared.X_val) if use_validation else None,
            pred_test=trained.predict(prepared.X_test),
            y_train=prepared.y_train,
            y_val=prepared.y_val if use_validation else None,
            y_test=prepared.y_test,
        )

    if route.is_pooled_ec_task:
        if model_info.get("flat_ec_labels", False):
            log("      Training on pooled flat EC embeddings for XGBoost...")
            log(
                "      Flat EC tensors: "
                f"X_train={prepared.X_train_flat_ec.shape}, "
                f"y_train={prepared.y_train_flat_ec.shape}, "
                f"X_val={prepared.X_val_flat_ec.shape if use_validation else None}, "
                f"known_train={int(prepared.target_mask_train_flat_ec.sum()):,}"
            )
            trained = train_model(
                model_info,
                prepared.X_train_flat_ec,
                prepared.y_train_flat_ec,
                X_val=prepared.X_val_flat_ec if use_validation else None,
                y_val=prepared.y_val_flat_ec if use_validation else None,
                target_mask=prepared.target_mask_train_flat_ec,
                target_mask_val=prepared.target_mask_val_flat_ec if use_validation else None,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
            )
            log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
            return ModelRunOutputs(
                trained=trained,
                pred_train=trained.predict(prepared.X_train_flat_ec),
                pred_val=trained.predict(prepared.X_val_flat_ec) if use_validation else None,
                pred_test=trained.predict(prepared.X_test_flat_ec),
                y_train=prepared.y_train_flat_ec,
                y_val=prepared.y_val_flat_ec if use_validation else None,
                y_test=prepared.y_test_flat_ec,
                target_mask_train=prepared.target_mask_train_flat_ec,
                target_mask_val=prepared.target_mask_val_flat_ec if use_validation else None,
                target_mask_test=prepared.target_mask_test_flat_ec,
            )

        log("      Training on pooled EC embeddings...")
        trained = train_model(
            model_info,
            prepared.X_train,
            prepared.y_train,
            X_val=prepared.X_val if use_validation else None,
            y_val=prepared.y_val if use_validation else None,
            target_mask=prepared.target_mask_train,
            target_mask_val=prepared.target_mask_val if use_validation else None,
            progress_label=progress_label,
            checkpoint_path=checkpoint_path,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return ModelRunOutputs(
            trained=trained,
            pred_train=trained.predict(prepared.X_train),
            pred_val=trained.predict(prepared.X_val) if use_validation else None,
            pred_test=trained.predict(prepared.X_test),
            y_train=prepared.y_train,
            y_val=prepared.y_val if use_validation else None,
            y_test=prepared.y_test,
            target_mask_train=prepared.target_mask_train,
            target_mask_val=prepared.target_mask_val if use_validation else None,
            target_mask_test=prepared.target_mask_test,
        )

    if _is_residue_aware_training(model_info, prepared):
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
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        residue_outputs = collect_residue_aware_outputs(
            trained,
            loaders=loaders,
            train_bundle=prepared.train_bundle,
            val_bundle=prepared.val_bundle,
            test_bundle=prepared.test_bundle,
            batch_kind=config.batch_kind,
            use_validation=use_validation,
        )
        return ModelRunOutputs(
            trained=residue_outputs.trained,
            pred_train=residue_outputs.pred_train,
            pred_val=residue_outputs.pred_val,
            pred_test=residue_outputs.pred_test,
            y_train=residue_outputs.y_train,
            y_val=residue_outputs.y_val,
            y_test=residue_outputs.y_test,
            target_mask_train=residue_outputs.target_mask_train,
            target_mask_val=residue_outputs.target_mask_val,
            target_mask_test=residue_outputs.target_mask_test,
        )

    log("      Pooling embeddings...")
    X_train_p, y_train_p = build_Xy_pooled(prepared.train_bundle, config.batch_kind)
    X_val_p, y_val_p = (
        build_Xy_pooled(prepared.val_bundle, config.batch_kind)
        if use_validation else (None, None)
    )
    X_test_p, y_test_p = build_Xy_pooled(prepared.test_bundle, config.batch_kind)
    log(f"      Pooled shapes: train={X_train_p.shape}, test={X_test_p.shape}. Training...")
    trained = train_model(
        model_info,
        X_train_p,
        y_train_p,
        X_val=X_val_p,
        y_val=y_val_p,
        progress_label=progress_label,
        checkpoint_path=checkpoint_path,
    )
    log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
    return ModelRunOutputs(
        trained=trained,
        pred_train=trained.predict(X_train_p),
        pred_val=trained.predict(X_val_p) if use_validation else None,
        pred_test=trained.predict(X_test_p),
        y_train=y_train_p,
        y_val=y_val_p if use_validation else None,
        y_test=y_test_p,
    )


def _run_task(
    *,
    task_name: str,
    prop: dict,
    emb_name: str,
    emb_info: dict,
    results_card_path: Path,
    output_dir: Path,
    model_prefix_filter: str | None,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    skip_label_kinds: set[str],
    residue_store_mode: str,
    benchmark_card: dict[str, Any] | None = None,
    results_card: dict[str, Any] | None = None,
) -> None:
    """Train all selected models for one task/embedding pair."""
    task_type = prop.get("task_type")
    if not task_type:
        raise ValueError(f"'{task_name}' is missing 'task_type' in card")

    registry_name = resolve_registry_name(task_name)
    if registry_name is None:
        log(f"  Skipping '{task_name}' (not in TASK_REGISTRY)")
        return

    config = get_task_config(registry_name)
    if config.label_kind in skip_label_kinds:
        log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' filtered by CLI)")
        return
    if config.label_kind in SKIP_LABEL_KINDS:
        log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' not yet supported)")
        return

    dataset_info = prop["datasets"][0]
    default_split = dataset_info.get("splits", {}).get("default_split")
    split_variant = resolve_split_variant(default_split)
    if split_variant is None:
        log(f"  Skipping '{task_name}' (no usable split defined in card)")
        return

    log(f"\n  Task: {task_name} | type: {task_type} | split: {default_split}")
    prepared: PreparedTaskData | None = None
    try:
        model_map = _select_model_map(task_type, config.label_kind, model_prefix_filter)
        if not model_map:
            log("    No models selected after filters — skipping task.")
            return

        already_done = _completed_models(
            model_map,
            output_dir,
            split_variant,
            task_name,
            emb_name,
            results_card=results_card,
        )
        if len(already_done) == len(model_map):
            log(f"    All {len(already_done)} models already trained — skipping task.")
            return
        if already_done:
            log(f"    {len(already_done)}/{len(model_map)} models already done: {already_done}")

        task_rows = load_task_rows_bundle(config, split_variant, allowed_ids, sequence_to_id)
        if not task_rows.train_rows and not task_rows.val_rows and not task_rows.test_rows:
            log("    Skipping task: allowed-id filtering left no rows in train/val/test.")
            return

        prepared = _prepare_task_data(
            config,
            registry_name,
            emb_info["path"],
            split_variant,
            task_rows,
            residue_store_mode=residue_store_mode,
            embedding_dim=int(emb_info["dim"]) if emb_info.get("dim") is not None else None,
            output_dir=output_dir,
            task_name=task_name,
            emb_name=emb_name,
        )
        n_train, n_val, n_test = _dataset_sizes(prepared)
        log_loaded_data_summary(task_rows, n_train, n_val, n_test)

        if n_train == 0 or n_test == 0:
            log(f"    Skipping '{task_name}': empty dataset after filtering.")
            return

        use_validation = n_val > 0
        if use_validation:
            log(f"    Validation enabled: n_val={n_val:,}")
        else:
            log("    Validation disabled: empty val split after filtering")

        for model_name, model_info in model_map.items():
            run_dir = build_result_run_dir(output_dir, split_variant, task_name, emb_name, model_name)
            if is_run_complete(run_dir):
                log(f"    [{model_name}] Already trained — skipping.")
                continue

            residue_aware = _is_residue_aware_training(model_info, prepared)
            log(f"    [{model_name}] n_train={n_train}, n_test={n_test} | residue_aware={residue_aware}")
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_stem = build_artifact_stem(task_name, emb_name, model_name, split_variant)

            if residue_aware and config.batch_kind == "protein_residue":
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
                    progress_label=f"{emb_name}/{config.name}/{model_name}",
                    checkpoint_path=run_dir / "training_checkpoint.bin",
                )
                log(f"      Training done ({time.time()-t0:.1f}s). Saving predictions split-by-split...")
                saved_prediction_paths = save_residue_aware_predictions(
                    trained,
                    loaders=loaders,
                    train_bundle=prepared.train_bundle,
                    val_bundle=prepared.val_bundle,
                    test_bundle=prepared.test_bundle,
                    batch_kind=config.batch_kind,
                    use_validation=use_validation,
                    run_dir=run_dir,
                    task_name=task_name,
                    emb_name=emb_name,
                    model_name=model_name,
                    split_variant=split_variant,
                )
            else:
                outputs = _train_one_model(
                    model_name,
                    emb_name,
                    model_info,
                    prepared,
                    config,
                    use_validation,
                    run_dir,
                )

                # Save metrics for HPO runs
                if model_name.startswith("mlp_hpo_"):
                    try:
                        import json
                        from sklearn.metrics import mean_absolute_error, r2_score
                        from scipy.stats import kendalltau
                        
                        val_mae = val_r2 = val_kendall = None
                        
                        if outputs.y_val is not None and outputs.pred_val is not None:
                            val_mae = float(mean_absolute_error(outputs.y_val, outputs.pred_val))
                            val_r2 = float(r2_score(outputs.y_val, outputs.pred_val))
                            k_corr, _ = kendalltau(outputs.y_val, outputs.pred_val)
                            val_kendall = float(k_corr) if k_corr is not None else None

                        params = model_info.get("params", {})
                        
                        result_data = {
                            "run_id": model_name,
                            "task_name": task_name,
                            "embedding": emb_name,
                            "val_MAE": val_mae,
                            "val_R2": val_r2,
                            "val_Kendall": val_kendall,
                            "best_epoch": getattr(outputs.trained, "best_epoch_", ""),
                            **params
                        }
                        
                        hpo_dir = output_dir / "hpo_metrics"
                        hpo_dir.mkdir(parents=True, exist_ok=True)
                        
                        isolated_file = hpo_dir / f"{model_name}.json"
                        with open(isolated_file, "w") as f:
                            json.dump(result_data, f, indent=4)
                            
                        log(f"      [HPO] Saved metrics to {isolated_file.name}. Bypassing file saves.")
                    except Exception as e:
                        log(f"      [HPO ERROR] Failed to log metrics: {e}")
                    
                    # Skips saving predictions, model.pt, and results_card.json for HPO runs
                    continue

                saved_prediction_paths = {
                    "train": make_predictions_file(
                        outputs.y_train,
                        outputs.pred_train,
                        run_dir,
                        "train",
                        outputs.target_mask_train,
                        artifact_path=build_prediction_artifact_path(
                            run_dir,
                            task_name=task_name,
                            emb_name=emb_name,
                            model_name=model_name,
                            split_variant=split_variant,
                            split_name="train",
                        ),
                    ),
                    "test": None,
                }
                if use_validation:
                    saved_prediction_paths["val"] = make_predictions_file(
                        outputs.y_val,
                        outputs.pred_val,
                        run_dir,
                        "val",
                        outputs.target_mask_val,
                        artifact_path=build_prediction_artifact_path(
                            run_dir,
                            task_name=task_name,
                            emb_name=emb_name,
                            model_name=model_name,
                            split_variant=split_variant,
                            split_name="val",
                        ),
                    )
                saved_prediction_paths["test"] = make_predictions_file(
                    outputs.y_test,
                    outputs.pred_test,
                    run_dir,
                    "test",
                    outputs.target_mask_test,
                    artifact_path=build_prediction_artifact_path(
                        run_dir,
                        task_name=task_name,
                        emb_name=emb_name,
                        model_name=model_name,
                        split_variant=split_variant,
                        split_name="test",
                    ),
                )
                trained = outputs.trained

            model_file = save_model(trained, run_dir)
            canonical_model_path = build_model_artifact_path(
                run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                extension=model_file.suffix,
            )
            if model_file != canonical_model_path:
                model_file = model_file.replace(canonical_model_path)

            if getattr(trained, "best_epoch_", None) is not None:
                log(
                    "      Early stopping: "
                    f"best_epoch={trained.best_epoch_}, "
                    f"best_val_loss={trained.best_val_loss_:.6f}"
                )

            result_entry = make_result_entry(
                emb_name,
                model_name,
                model_info,
                run_dir,
                model_file,
                task_name=task_name,
                split_variant=split_variant,
                task_scope="properties",
                train_pred_file=saved_prediction_paths["train"],
                test_pred_file=saved_prediction_paths["test"],
                val_pred_file=saved_prediction_paths.get("val"),
            )

            update_results_card_locked(
                results_card_path,
                task_scope="properties",
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                result_entry=result_entry,
                benchmark_card=benchmark_card,
            )
            log(f"      Saved → {run_dir.relative_to(output_dir)} [{artifact_stem}]")
    except Exception as task_exc:
        log(f"  [ERROR] Task '{task_name}' failed: {task_exc}")
        log(traceback.format_exc())
    finally:
        _cleanup_prepared_task_data(prepared)


def main(
    card_path: str,
    embeddings_json: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    model_prefix_filter: str | None = None,
    task_name_filters: set[str] | None = None,
    skip_label_kinds: set[str] | None = None,
    allowed_ids_path: str | None = None,
    residue_store_mode: str = "auto",
    *,
    results_card_path: str,
):
    """Train all selected embedding-model and task combinations described by the card."""
    log(f"[startup] Python started. emb_name={emb_name_filter}")
    if model_prefix_filter is not None:
        log(f"[startup] Filtering model names by prefix: {model_prefix_filter}")

    task_name_filters = task_name_filters or set()
    if task_name_filters:
        log(f"[startup] Restricting tasks to: {sorted(task_name_filters)}")

    skip_label_kinds = skip_label_kinds or set()
    if skip_label_kinds:
        log(f"[startup] Skipping label kinds: {sorted(skip_label_kinds)}")
    log(f"[startup] Residue store mode: {residue_store_mode}")

    card = read_json(card_path)
    embeddings = read_json(embeddings_json)
    log(f"[startup] JSON loaded. {len(embeddings)} embedding models, {len(card['properties'])} tasks.")

    card_path_obj = Path(card_path)
    results_card_path_obj = Path(results_card_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    ensure_results_card(results_card_path_obj, benchmark_card=card)
    results_card = read_results_card(results_card_path_obj)

    allowed_ids: set[str] | None = None
    sequence_to_id: dict[str, str] = {}
    if allowed_ids_path is not None:
        allowed_ids = load_allowed_ids(allowed_ids_path)
        sequence_index_path = resolve_sequence_index_path()
        if sequence_index_path is not None:
            sequence_to_id = load_sequence_to_id_index(sequence_index_path)
        log(
            f"[startup] Restricting training rows to {len(allowed_ids):,} embedding IDs "
            f"from {allowed_ids_path}"
        )

    if emb_name_filter is not None and emb_name_filter not in embeddings:
        raise ValueError(f"Unknown embedding '{emb_name_filter}'. Available: {', '.join(embeddings)}")

    for emb_name, emb_info in embeddings.items():
        if emb_name_filter is not None and emb_name != emb_name_filter:
            continue

        log(f"\n=== Embedding: {emb_name} (dim={emb_info['dim']}) ===")

        for task_name, prop in card["properties"].items():
            if task_name_filters and task_name not in task_name_filters:
                log(f"  Skipping '{task_name}' (not selected by CLI task filter)")
                continue

            _run_task(
                task_name=task_name,
                prop=prop,
                emb_name=emb_name,
                emb_info=emb_info,
                results_card_path=results_card_path_obj,
                output_dir=output_dir_obj,
                model_prefix_filter=model_prefix_filter,
                allowed_ids=allowed_ids,
                sequence_to_id=sequence_to_id,
                skip_label_kinds=skip_label_kinds,
                residue_store_mode=residue_store_mode,
                benchmark_card=card,
                results_card=results_card,
            )

    log(f"\nDone. Dataset card read from {card_path_obj}")
    log(f"Done. Results written to {results_card_path_obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to embeddings_models_output.json")
    parser.add_argument("output_dir", help="Directory to write prediction and model files")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding (for SLURM arrays)")
    parser.add_argument(
        "--model_prefix",
        default=None,
        help="Train only models whose name starts with this prefix, e.g. 'mlp_'.",
    )
    parser.add_argument(
        "--task_name",
        action="append",
        default=[],
        help="Train only this benchmark task name. Repeatable.",
    )
    parser.add_argument(
        "--skip_label_kind",
        action="append",
        default=[],
        help="Skip tasks with this label_kind. Repeatable.",
    )
    parser.add_argument(
        "--allowed_ids",
        default=None,
        help="Optional path to common_ids.txt. Rows whose embeddings are not in this ID set are excluded.",
    )
    parser.add_argument(
        "--residue_store_mode",
        choices=("auto", "materialized", "lazy"),
        default="auto",
        help=(
            "How protein_residue tasks should access residue embeddings. "
            "'auto' switches to lazy mode when the estimated materialized cache is too large."
        ),
    )
    parser.add_argument(
        "--results_card",
        required=True,
        help="Path to results_card.json.",
    )
    args = parser.parse_args()
    main(
        args.card_path,
        args.embeddings_json,
        args.output_dir,
        args.emb_name,
        model_prefix_filter=args.model_prefix,
        task_name_filters=set(args.task_name),
        skip_label_kinds=set(args.skip_label_kind),
        allowed_ids_path=args.allowed_ids,
        residue_store_mode=args.residue_store_mode,
        results_card_path=args.results_card,
    )
