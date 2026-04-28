"""Train benchmark models across embedding sets and registered benchmark tasks."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    build_result_run_dir,
    is_run_complete,
    load_allowed_ids,
    load_sequence_to_id_index,
    log,
    resolve_registry_name,
    resolve_sequence_index_path,
    resolve_split_variant,
)
from models.results_card import (
    build_artifact_stem,
    build_model_artifact_path,
    canonical_embedding_mode,
    ensure_results_card,
    find_result_entry,
    is_result_entry_complete,
    read_results_card,
    update_results_card_locked,
)
from models.train_data import (
    build_Xy_ligand_direct,
    build_Xy_pooled,
    build_flat_multilabel_ec_direct,
    build_multilabel_ec_direct,
    build_mutation_delta_Xy_direct,
    build_protein_global_Xy_direct,
    get_target_payload,
    load_embedding_store_subset,
    load_lazy_embedding_store,
    load_lazy_safetensors_embedding_store,
    load_pooled_embedding_subset,
    load_safetensors_embedding_store_subset,
)
from models.train_output import (
    make_result_entry,
    read_json,
    save_model,
    save_model_safetensors,
    save_prediction_artifacts,
    write_hparams_file,
    write_metrics_file,
    write_training_log_file,
)
from models.train_runtime import (
    get_model_map,
    train_model,
)
from models.training.residue import (
    build_residue_aware_loaders,
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
    embedding_storage_format: str = "safetensors",
):
    """Return the embedding store strategy appropriate for one task.

    H5 remains available through ``embedding_storage_format='h5'`` for legacy runs.
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
            if embedding_storage_format == "h5":
                store = load_embedding_store_subset(h5_path, needed_ids, needed_sequences)
            else:
                store = load_safetensors_embedding_store_subset(
                    h5_path,
                    needed_ids,
                    needed_sequences,
                    storage_format=embedding_storage_format,
                )
            return store, resolved_mode
        if resolved_mode == "lazy":
            if embedding_storage_format == "h5":
                store = load_lazy_embedding_store(h5_path, needed_ids, needed_sequences)
            else:
                store = load_lazy_safetensors_embedding_store(
                    h5_path,
                    needed_ids,
                    needed_sequences,
                    storage_format=embedding_storage_format,
                )
            return store, resolved_mode
        raise ValueError(
            "residue_store_mode must be one of 'auto', 'materialized', or 'lazy'. "
            f"Received '{residue_store_mode}'."
        )

    if embedding_storage_format == "h5":
        store = load_lazy_embedding_store(h5_path, needed_ids, needed_sequences)
    else:
        store = load_lazy_safetensors_embedding_store(
            h5_path,
            needed_ids,
            needed_sequences,
            storage_format=embedding_storage_format,
        )
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


@dataclass
class SavedModelRun:
    """Trained estimator plus the split artifacts saved for one model run."""

    trained: Any
    prediction_paths: dict[str, Path]
    metrics_by_split: dict[str, dict[str, Any]]


@dataclass
class ResidueTrainingPayload:
    """Residue-aware training payload kept small until predictions are saved."""

    trained: Any
    loaders: Any


def _fit_predict_arrays(
    *,
    model_info: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    use_validation: bool,
    progress_label: str,
    checkpoint_path: Path,
    target_mask_train=None,
    target_mask_val=None,
    target_mask_test=None,
) -> ModelRunOutputs:
    """Fit one array-based model and collect predictions using a consistent contract."""
    fit_kwargs = {
        "X_val": X_val if use_validation else None,
        "y_val": y_val if use_validation else None,
        "progress_label": progress_label,
        "checkpoint_path": checkpoint_path,
    }
    if target_mask_train is not None:
        fit_kwargs["target_mask"] = target_mask_train
    if use_validation and target_mask_val is not None:
        fit_kwargs["target_mask_val"] = target_mask_val

    trained = train_model(model_info, X_train, y_train, **fit_kwargs)
    return ModelRunOutputs(
        trained=trained,
        pred_train=trained.predict(X_train),
        pred_val=trained.predict(X_val) if use_validation else None,
        pred_test=trained.predict(X_test),
        y_train=y_train,
        y_val=y_val if use_validation else None,
        y_test=y_test,
        target_mask_train=target_mask_train,
        target_mask_val=target_mask_val if use_validation else None,
        target_mask_test=target_mask_test,
    )


def _save_outputs_predictions(
    outputs: ModelRunOutputs,
    *,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    task_type: str,
    use_validation: bool,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Save predictions for standard in-memory outputs and compute split metrics."""
    split_payloads = [
        (
            "train",
            outputs.y_train,
            outputs.pred_train,
            outputs.target_mask_train,
        )
    ]
    if use_validation:
        split_payloads.append(
            (
                "val",
                outputs.y_val,
                outputs.pred_val,
                outputs.target_mask_val,
            )
        )
    split_payloads.append(
        (
            "test",
            outputs.y_test,
            outputs.pred_test,
            outputs.target_mask_test,
        )
    )

    saved_paths: dict[str, Path] = {}
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name, y_true, pred, target_mask in split_payloads:
        prediction_path, metrics = save_prediction_artifacts(
            y_true,
            pred,
            target_mask,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name=split_name,
            task_type=task_type,
        )
        saved_paths[split_name] = prediction_path
        metrics_by_split[split_name] = metrics
    return saved_paths, metrics_by_split


class TrainingRoute:
    """Base interface for a task route inside the non-attention training pipeline."""

    name = "base"

    def save_predictions(
        self,
        trained_payload,
        *,
        run_dir: Path,
        task_name: str,
        emb_name: str,
        model_name: str,
        split_variant: str,
        task_type: str,
        use_validation: bool,
        prepared: PreparedTaskData,
        config,
    ) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
        """Default split artifact saving for routes that train in-memory outputs."""
        del prepared, config
        return _save_outputs_predictions(
            trained_payload,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            task_type=task_type,
            use_validation=use_validation,
        )

    def run_model(
        self,
        *,
        model_name: str,
        emb_name: str,
        model_info: dict,
        prepared: PreparedTaskData,
        config,
        use_validation: bool,
        run_dir: Path,
        task_name: str,
        split_variant: str,
        task_type: str,
    ) -> SavedModelRun:
        trained_payload = self.train(
            model_name=model_name,
            emb_name=emb_name,
            model_info=model_info,
            prepared=prepared,
            config=config,
            use_validation=use_validation,
            run_dir=run_dir,
        )
        prediction_paths, metrics_by_split = self.save_predictions(
            trained_payload,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            task_type=task_type,
            use_validation=use_validation,
            prepared=prepared,
            config=config,
        )
        return SavedModelRun(
            trained=trained_payload.trained,
            prediction_paths=prediction_paths,
            metrics_by_split=metrics_by_split,
        )


class DirectArrayTrainingRoute(TrainingRoute):
    """Route for tasks that can be fully prepared as pooled 2D arrays."""

    def __init__(
        self,
        *,
        name: str,
        load_label: str,
        build_label: str,
        train_label: str,
        array_builder: Callable[[list[dict], dict[str, Any], dict[str, Any], Any], tuple[Any, Any]],
    ) -> None:
        self.name = name
        self.load_label = load_label
        self.build_label = build_label
        self.train_label = train_label
        self.array_builder = array_builder

    def prepare(
        self,
        config,
        registry_name: str,
        emb_path: str | Path,
        split_variant: str,
        task_rows: LoadedTaskRows,
        *,
        residue_store_mode: str = "auto",
        embedding_dim: int | None = None,
        embedding_storage_format: str = "safetensors",
    ) -> PreparedTaskData:
        del registry_name, split_variant, residue_store_mode, embedding_dim
        route = _resolve_task_route(config)
        prepared = PreparedTaskData(route=route)

        log(f"    Loading pooled embeddings from {embedding_storage_format} ({self.load_label})...")
        t0 = time.time()
        by_id, by_seq = load_pooled_embedding_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
            storage_format=embedding_storage_format,
        )
        log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

        log(f"    Building {self.build_label}...")
        prepared.X_train, prepared.y_train = self.array_builder(
            task_rows.train_rows, by_id, by_seq, config
        )
        prepared.X_val, prepared.y_val = self.array_builder(
            task_rows.val_rows, by_id, by_seq, config
        )
        prepared.X_test, prepared.y_test = self.array_builder(
            task_rows.test_rows, by_id, by_seq, config
        )
        return prepared

    def train(
        self,
        *,
        model_name: str,
        emb_name: str,
        model_info: dict,
        prepared: PreparedTaskData,
        config,
        use_validation: bool,
        run_dir: Path,
    ) -> ModelRunOutputs:
        t0 = time.time()
        log(f"      Training on {self.train_label}...")
        outputs = _fit_predict_arrays(
            model_info=model_info,
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
            use_validation=use_validation,
            progress_label=f"{emb_name}/{config.name}/{model_name}",
            checkpoint_path=run_dir / "training_checkpoint.bin",
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return outputs

class PooledECTrainingRoute(TrainingRoute):
    """Route for EC tasks with hierarchical labels plus a flat XGBoost view."""

    name = "pooled_ec"

    def prepare(
        self,
        config,
        registry_name: str,
        emb_path: str | Path,
        split_variant: str,
        task_rows: LoadedTaskRows,
        *,
        residue_store_mode: str = "auto",
        embedding_dim: int | None = None,
        embedding_storage_format: str = "safetensors",
    ) -> PreparedTaskData:
        del registry_name, split_variant, residue_store_mode, embedding_dim
        prepared = PreparedTaskData(route=_resolve_task_route(config))

        log(f"    Loading pooled embeddings from {embedding_storage_format} (EC task)...")
        t0 = time.time()
        by_id, by_seq = load_pooled_embedding_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
            storage_format=embedding_storage_format,
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

    def train(
        self,
        *,
        model_name: str,
        emb_name: str,
        model_info: dict,
        prepared: PreparedTaskData,
        config,
        use_validation: bool,
        run_dir: Path,
    ) -> ModelRunOutputs:
        progress_label = f"{emb_name}/{config.name}/{model_name}"
        checkpoint_path = run_dir / "training_checkpoint.bin"
        t0 = time.time()

        if model_info.get("flat_ec_labels", False):
            log("      Training on pooled flat EC embeddings for XGBoost...")
            log(
                "      Flat EC tensors: "
                f"X_train={prepared.X_train_flat_ec.shape}, "
                f"y_train={prepared.y_train_flat_ec.shape}, "
                f"X_val={prepared.X_val_flat_ec.shape if use_validation else None}, "
                f"known_train={int(prepared.target_mask_train_flat_ec.sum()):,}"
            )
            outputs = _fit_predict_arrays(
                model_info=model_info,
                X_train=prepared.X_train_flat_ec,
                y_train=prepared.y_train_flat_ec,
                X_val=prepared.X_val_flat_ec,
                y_val=prepared.y_val_flat_ec,
                X_test=prepared.X_test_flat_ec,
                y_test=prepared.y_test_flat_ec,
                use_validation=use_validation,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
                target_mask_train=prepared.target_mask_train_flat_ec,
                target_mask_val=prepared.target_mask_val_flat_ec,
                target_mask_test=prepared.target_mask_test_flat_ec,
            )
        else:
            log("      Training on pooled EC embeddings...")
            outputs = _fit_predict_arrays(
                model_info=model_info,
                X_train=prepared.X_train,
                y_train=prepared.y_train,
                X_val=prepared.X_val,
                y_val=prepared.y_val,
                X_test=prepared.X_test,
                y_test=prepared.y_test,
                use_validation=use_validation,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
                target_mask_train=prepared.target_mask_train,
                target_mask_val=prepared.target_mask_val,
                target_mask_test=prepared.target_mask_test,
            )

        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return outputs

class DatasetBundleTrainingRoute(TrainingRoute):
    """Route for tasks that need dataset bundles and optionally residue-aware loaders."""

    name = "dataset_bundle"

    def prepare(
        self,
        config,
        registry_name: str,
        emb_path: str | Path,
        split_variant: str,
        task_rows: LoadedTaskRows,
        *,
        residue_store_mode: str = "auto",
        embedding_dim: int | None = None,
        embedding_storage_format: str = "safetensors",
    ) -> PreparedTaskData:
        prepared = PreparedTaskData(route=_resolve_task_route(config))
        if config.batch_kind == "protein_residue":
            if residue_store_mode == "materialized":
                log("    Materializing residue embedding subset into RAM...")
            elif residue_store_mode == "lazy":
                log("    Building lazy residue embedding store...")
            else:
                log("    Choosing residue embedding store strategy...")
        else:
            log(f"    Scanning {embedding_storage_format} keys for lazy store...")

        t0 = time.time()
        prepared.store, store_mode = load_task_embedding_store(
            config,
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
            residue_store_mode=residue_store_mode,
            embedding_dim=embedding_dim,
            embedding_storage_format=embedding_storage_format,
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

    def train(
        self,
        *,
        model_name: str,
        emb_name: str,
        model_info: dict,
        prepared: PreparedTaskData,
        config,
        use_validation: bool,
        run_dir: Path,
    ):
        progress_label = f"{emb_name}/{config.name}/{model_name}"
        checkpoint_path = run_dir / "training_checkpoint.bin"
        t0 = time.time()

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
            log(f"      Training done ({time.time()-t0:.1f}s). Saving predictions split-by-split...")
            return ResidueTrainingPayload(trained=trained, loaders=loaders)

        log("      Pooling embeddings...")
        X_train_p, y_train_p = build_Xy_pooled(prepared.train_bundle, config.batch_kind)
        X_val_p, y_val_p = (
            build_Xy_pooled(prepared.val_bundle, config.batch_kind)
            if use_validation else (None, None)
        )
        X_test_p, y_test_p = build_Xy_pooled(prepared.test_bundle, config.batch_kind)
        log(f"      Pooled shapes: train={X_train_p.shape}, test={X_test_p.shape}. Training...")
        outputs = _fit_predict_arrays(
            model_info=model_info,
            X_train=X_train_p,
            y_train=y_train_p,
            X_val=X_val_p,
            y_val=y_val_p,
            X_test=X_test_p,
            y_test=y_test_p,
            use_validation=use_validation,
            progress_label=progress_label,
            checkpoint_path=checkpoint_path,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return outputs

    def save_predictions(
        self,
        trained_payload,
        *,
        run_dir: Path,
        task_name: str,
        emb_name: str,
        model_name: str,
        split_variant: str,
        task_type: str,
        use_validation: bool,
        prepared: PreparedTaskData,
        config,
    ) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
        if isinstance(trained_payload, ResidueTrainingPayload):
            return self._save_residue_predictions(
                trained_payload,
                run_dir=run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                task_type=task_type,
                use_validation=use_validation,
                prepared=prepared,
                config=config,
            )
        return _save_outputs_predictions(
            trained_payload,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            task_type=task_type,
            use_validation=use_validation,
        )

    def _save_residue_predictions(
        self,
        trained_payload: ResidueTrainingPayload,
        *,
        run_dir: Path,
        task_name: str,
        emb_name: str,
        model_name: str,
        split_variant: str,
        task_type: str,
        use_validation: bool,
        prepared: PreparedTaskData,
        config,
    ) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
        import gc

        split_jobs: list[tuple[str, Any, Any]] = [
            ("train", prepared.train_bundle, trained_payload.loaders.train_loader),
            ("test", prepared.test_bundle, trained_payload.loaders.test_loader),
        ]
        if use_validation:
            split_jobs.insert(1, ("val", prepared.val_bundle, trained_payload.loaders.val_loader))

        saved_paths: dict[str, Path] = {}
        metrics_by_split: dict[str, dict[str, Any]] = {}
        for split_name, bundle, loader in split_jobs:
            log(f"      Predicting/saving split='{split_name}'...")
            y_true, target_mask = get_target_payload(bundle)
            pred = trained_payload.trained.predict_loader(loader, batch_kind=config.batch_kind)
            prediction_path, metrics = save_prediction_artifacts(
                y_true,
                pred,
                target_mask,
                run_dir=run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                split_name=split_name,
                task_type=task_type,
            )
            saved_paths[split_name] = prediction_path
            metrics_by_split[split_name] = metrics
            del y_true, target_mask, pred
            gc.collect()
        return saved_paths, metrics_by_split


def _select_training_route(config) -> TrainingRoute:
    """Resolve a config into the route object that owns prepare/train/save."""
    route = _resolve_task_route(config)
    if route.is_ligand_task:
        return DirectArrayTrainingRoute(
            name="protein_ligand_global",
            load_label="ligand task",
            build_label="X/y arrays",
            train_label="pooled protein + ligand embeddings",
            array_builder=build_Xy_ligand_direct,
        )
    if route.is_pooled_ec_task:
        return PooledECTrainingRoute()
    if route.is_pooled_protein_global_task:
        return DirectArrayTrainingRoute(
            name="protein_global",
            load_label="protein-global task",
            build_label="pooled X/y arrays",
            train_label="pooled protein-global embeddings",
            array_builder=build_protein_global_Xy_direct,
        )
    if route.is_pooled_mutation_task:
        return DirectArrayTrainingRoute(
            name="mutation_delta_global",
            load_label="mutation-delta task",
            build_label="pooled mutation-delta X/y arrays",
            train_label="pooled mutation-delta embeddings",
            array_builder=build_mutation_delta_Xy_direct,
        )
    return DatasetBundleTrainingRoute()


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


def _filter_model_map_by_embedding_modes(
    model_map: dict[str, dict],
    task_name: str,
    embedding_mode_filters: set[str],
) -> dict[str, dict]:
    """Keep only models whose artifact embedding mode matches requested modes."""
    if not embedding_mode_filters:
        return model_map
    return {
        name: info
        for name, info in model_map.items()
        if canonical_embedding_mode(task_name, name) in embedding_mode_filters
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
    split_variant_filter: str | None,
    embedding_mode_filters: set[str],
    embedding_storage_format: str,
    overwrite_results: bool,
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
    split_variant = split_variant_filter or resolve_split_variant(default_split)
    if split_variant is None:
        log(f"  Skipping '{task_name}' (no usable split defined in card)")
        return
    if split_variant_filter is not None and split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        log(
            f"  Skipping '{task_name}' (requested split '{split_variant}' is not available; "
            f"available: {available})"
        )
        return

    log(f"\n  Task: {task_name} | type: {task_type} | split: {split_variant}")
    prepared: PreparedTaskData | None = None
    try:
        model_map = _select_model_map(task_type, config.label_kind, model_prefix_filter)
        model_map = _filter_model_map_by_embedding_modes(
            model_map,
            task_name,
            embedding_mode_filters,
        )
        if not model_map:
            log("    No models selected after filters — skipping task.")
            return

        if not overwrite_results:
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
        else:
            log("    Overwrite enabled: existing artifacts for selected models will be replaced.")

        task_rows = load_task_rows_bundle(config, split_variant, allowed_ids, sequence_to_id)
        if not task_rows.train_rows and not task_rows.val_rows and not task_rows.test_rows:
            log("    Skipping task: allowed-id filtering left no rows in train/val/test.")
            return

        training_route = _select_training_route(config)
        prepared = training_route.prepare(
            config,
            registry_name,
            emb_info["path"],
            split_variant,
            task_rows,
            residue_store_mode=residue_store_mode,
            embedding_dim=int(emb_info["dim"]) if emb_info.get("dim") is not None else None,
            embedding_storage_format=embedding_storage_format,
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
            if not overwrite_results and is_run_complete(run_dir):
                log(f"    [{model_name}] Already trained — skipping.")
                continue
            if overwrite_results and run_dir.exists():
                shutil.rmtree(run_dir)

            residue_aware = _is_residue_aware_training(model_info, prepared)
            log(
                f"    [{model_name}] n_train={n_train}, n_test={n_test} "
                f"| route={training_route.name} | residue_aware={residue_aware}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_stem = build_artifact_stem(task_name, emb_name, model_name, split_variant)

            hparams_file = write_hparams_file(
                run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                model_info=model_info,
            )
            saved_run = training_route.run_model(
                model_name=model_name,
                emb_name=emb_name,
                model_info=model_info,
                prepared=prepared,
                config=config,
                use_validation=use_validation,
                run_dir=run_dir,
                task_name=task_name,
                split_variant=split_variant,
                task_type=task_type,
            )
            trained = saved_run.trained
            saved_prediction_paths = saved_run.prediction_paths
            metrics_file = write_metrics_file(run_dir, saved_run.metrics_by_split)
            training_log_file = write_training_log_file(trained, run_dir)

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
            model_safetensors_file = save_model_safetensors(
                trained,
                run_dir,
                model_path=build_model_artifact_path(
                    run_dir,
                    task_name=task_name,
                    emb_name=emb_name,
                    model_name=model_name,
                    split_variant=split_variant,
                    extension=".safetensors",
                ),
            )

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
                train_pred_safetensors_file=saved_prediction_paths["train"],
                test_pred_safetensors_file=saved_prediction_paths["test"],
                val_pred_safetensors_file=saved_prediction_paths.get("val"),
                model_safetensors_file=model_safetensors_file,
                metrics=saved_run.metrics_by_split,
                metrics_file=metrics_file,
                hparams_file=hparams_file,
                training_log_file=training_log_file,
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
    split_variant_filter: str | None = None,
    embedding_mode_filters: set[str] | None = None,
    embedding_storage_format: str = "safetensors",
    overwrite_results: bool = False,
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
    if split_variant_filter is not None:
        log(f"[startup] Restricting split variant to: {split_variant_filter}")
    embedding_mode_filters = embedding_mode_filters or set()
    if embedding_mode_filters:
        log(f"[startup] Restricting embedding modes to: {sorted(embedding_mode_filters)}")
    log(f"[startup] Embedding storage format: {embedding_storage_format}")
    log(f"[startup] Overwrite results: {overwrite_results}")

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
                split_variant_filter=split_variant_filter,
                embedding_mode_filters=embedding_mode_filters,
                embedding_storage_format=embedding_storage_format,
                overwrite_results=overwrite_results,
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
        "--split_variant",
        choices=("random", "similarity", "loco"),
        default=None,
        help="Train only this split variant. Defaults to each task's card default_split.",
    )
    parser.add_argument(
        "--embedding_mode",
        action="append",
        choices=("mean", "per_residue", "attention"),
        default=[],
        help=(
            "Train only model/task combinations in this results embedding mode. "
            "Use --embedding_mode mean for mean-pooled baselines."
        ),
    )
    parser.add_argument(
        "--embedding_storage_format",
        choices=("safetensors", "h5", "auto"),
        default=os.getenv("ENZYMADO_EMBEDDING_STORAGE_FORMAT", "safetensors"),
        help="Embedding storage backend to use. Defaults to safetensors.",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Replace existing artifacts for selected runs instead of skipping completed runs.",
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
        split_variant_filter=args.split_variant,
        embedding_mode_filters=set(args.embedding_mode),
        embedding_storage_format=args.embedding_storage_format,
        overwrite_results=args.overwrite_results,
        results_card_path=args.results_card,
    )
