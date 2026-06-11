"""Train benchmark models across embedding sets and registered benchmark tasks."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback

import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "predictions"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import (
    TASK_REGISTRY,
    FlatECLabelBuilder,
    HierarchicalECLabelBuilder,
    PooledEmbeddingStore,
    build_task_dataset,
    get_task_config,
)

from data_loading.stores import (
    build_ec_arrays,
    build_flat_ec_arrays,
    build_pooled_embedding_store,
    build_xy_pooled,
    get_dataloader,
    get_target_payload,
    load_allowed_ids,
    load_embeddings,
    load_lazy_safetensors_embedding_store,
    load_pooled_embedding_subset,
    load_safetensors_embedding_store_subset,
    load_sequence_to_id_index,
    log,
    resolve_sequence_index_path,
)
from evaluation.artifacts import (
    read_json,
    save_model,
    save_model_safetensors,
    save_prediction_artifacts,
    write_hparams_file,
    write_metrics_file,
    write_training_log_file,
)
from models.attention_registry import get_attention_model_map
from models.registry import get_model_map, train_model, train_model_loader
from training.attention_wrappers import save_attention_checkpoint
from training.results_card import (
    build_artifact_stem,
    build_model_artifact_path,
    build_results_run_dir,
    canonical_embedding_mode,
    is_run_complete,
)
from training.residue import (
    build_residue_aware_loaders,
    train_residue_aware_model,
)
from training.task_rows import (
    LoadedTaskRows,
    load_task_rows_bundle,
    log_loaded_data_summary,
)


def load_task_embedding_store(
    config,
    emb_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    residue_store_mode: str = "auto",
    embedding_dim: int | None = None,
):
    """Return the embedding store strategy appropriate for one task."""
    if config.batch_kind == "protein_residue":
        resolved_mode = residue_store_mode
        estimate_gib = _estimate_materialized_residue_store_gib(
            needed_sequences,
            embedding_dim,
        )
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
            store = load_safetensors_embedding_store_subset(
                emb_path, needed_ids, needed_sequences
            )
            return store, resolved_mode
        if resolved_mode == "lazy":
            store = load_lazy_safetensors_embedding_store(
                emb_path, needed_ids, needed_sequences
            )
            return store, resolved_mode
        raise ValueError(
            "residue_store_mode must be one of 'auto', 'materialized', or 'lazy'. "
            f"Received '{residue_store_mode}'."
        )

    store = load_lazy_safetensors_embedding_store(emb_path, needed_ids, needed_sequences)
    return store, "lazy"


@dataclass(frozen=True)
class TaskRoute:
    """Describe which training path a task follows inside the main pipeline."""

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
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
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
        "x_val": x_val if use_validation else None,
        "y_val": y_val if use_validation else None,
        "progress_label": progress_label,
        "checkpoint_path": checkpoint_path,
    }
    if target_mask_train is not None:
        fit_kwargs["target_mask"] = target_mask_train
    if use_validation and target_mask_val is not None:
        fit_kwargs["target_mask_val"] = target_mask_val

    trained = train_model(model_info, x_train, y_train, **fit_kwargs)
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

    def save_trained_model(self, trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
        """Persist the trained estimator to disk; returns the written file path."""
        return save_model(trained, run_dir)

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


class GlobalRoute(TrainingRoute):
    """Route for all non-residue tasks: loads pooled embeddings once and builds x/y arrays."""

    name = "global"

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
        needs_hierarchical_ec: bool = True,
        needs_flat_ec: bool = False,
    ) -> PreparedTaskData:
        del residue_store_mode, embedding_dim
        prepared = PreparedTaskData(route=_resolve_task_route(config))

        log("    Loading pooled embeddings from safetensors...")
        t0 = time.time()
        by_id, by_seq = load_pooled_embedding_subset(
            emb_path,
            task_rows.needed_ids,
            task_rows.needed_sequences,
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
                (
                    prepared.x_train_flat_ec,
                    prepared.y_train_flat_ec,
                    prepared.target_mask_train_flat_ec,
                ) = build_flat_ec_arrays(
                    task_rows.train_rows, by_id, by_seq, config, flat_ec_label_builder
                )
                (
                    prepared.x_val_flat_ec,
                    prepared.y_val_flat_ec,
                    prepared.target_mask_val_flat_ec,
                ) = build_flat_ec_arrays(
                    task_rows.val_rows, by_id, by_seq, config, flat_ec_label_builder
                )
                (
                    prepared.x_test_flat_ec,
                    prepared.y_test_flat_ec,
                    prepared.target_mask_test_flat_ec,
                ) = build_flat_ec_arrays(
                    task_rows.test_rows, by_id, by_seq, config, flat_ec_label_builder
                )
        else:
            log("    Validating rows against embedding store...")
            store = build_pooled_embedding_store(by_id, by_seq)
            ec_vocabulary = None
            prepared.train_bundle = build_task_dataset(
                registry_name, store, split_name="train", split_variant=split_variant, missing_policy="skip",
            )
            prepared.val_bundle = build_task_dataset(
                registry_name, store, split_name="val", split_variant=split_variant, missing_policy="skip",
            )
            prepared.test_bundle = build_task_dataset(
                registry_name, store, split_name="test", split_variant=split_variant, missing_policy="skip",
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
            if prepared.x_train_flat_ec is None:
                raise RuntimeError("Flat EC arrays were not prepared for this model.")
            log("      Training on pooled flat EC embeddings for XGBoost...")
            log(
                "      Flat EC tensors: "
                f"x_train={prepared.x_train_flat_ec.shape}, "
                f"y_train={prepared.y_train_flat_ec.shape}, "
                f"known_train={int(prepared.target_mask_train_flat_ec.sum()):,}"
            )
            outputs = _fit_predict_arrays(
                model_info=model_info,
                x_train=prepared.x_train_flat_ec,
                y_train=prepared.y_train_flat_ec,
                x_val=prepared.x_val_flat_ec,
                y_val=prepared.y_val_flat_ec,
                x_test=prepared.x_test_flat_ec,
                y_test=prepared.y_test_flat_ec,
                use_validation=use_validation,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
                target_mask_train=prepared.target_mask_train_flat_ec,
                target_mask_val=prepared.target_mask_val_flat_ec,
                target_mask_test=prepared.target_mask_test_flat_ec,
            )
        elif config.label_kind == "multilabel_ec":
            if prepared.x_train is None:
                raise RuntimeError("Hierarchical EC arrays were not prepared for this model.")
            log("      Training on pooled EC embeddings...")
            outputs = _fit_predict_arrays(
                model_info=model_info,
                x_train=prepared.x_train,
                y_train=prepared.y_train,
                x_val=prepared.x_val,
                y_val=prepared.y_val,
                x_test=prepared.x_test,
                y_test=prepared.y_test,
                use_validation=use_validation,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
                target_mask_train=prepared.target_mask_train,
                target_mask_val=prepared.target_mask_val,
                target_mask_test=prepared.target_mask_test,
            )
        else:
            log(f"      Pooling embeddings ({config.batch_kind})...")
            x_train, y_train = build_xy_pooled(prepared.train_bundle, config.batch_kind)
            x_val, y_val = (
                build_xy_pooled(prepared.val_bundle, config.batch_kind)
                if use_validation else (None, None)
            )
            x_test, y_test = build_xy_pooled(prepared.test_bundle, config.batch_kind)
            log(f"      Training on pooled embeddings (train={x_train.shape})...")
            outputs = _fit_predict_arrays(
                model_info=model_info,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                use_validation=use_validation,
                progress_label=progress_label,
                checkpoint_path=checkpoint_path,
            )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
        return outputs

class ResidueTasksRoute(TrainingRoute):
    """Route for protein_residue tasks: lazy or materialized residue store + DataLoaders."""

    name = "residue"

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
        needs_hierarchical_ec: bool = True,
        needs_flat_ec: bool = False,
    ) -> PreparedTaskData:
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
        n_train = len(prepared.train_bundle.dataset)
        n_val = len(prepared.val_bundle.dataset)
        n_test = len(prepared.test_bundle.dataset)
        log(f"    Row validation done ({time.time()-t0:.1f}s) — train={n_train:,} val={n_val:,} test={n_test:,}")
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


class AttentionRoute(TrainingRoute):
    """Route for attention-pooled models: lazy residue store + DataLoader-based training."""

    name = "attention"

    def prepare(
        self,
        config,
        registry_name: str,
        emb_path: str | Path,
        split_variant: str,
        task_rows: LoadedTaskRows,
        **kwargs,
    ) -> PreparedTaskData:
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
        n_train = len(train_bundle.dataset)
        n_val = len(val_bundle.dataset)
        n_test = len(test_bundle.dataset)
        log(f"    Row validation done ({time.time()-t0:.1f}s) — train={n_train:,} val={n_val:,} test={n_test:,}")
        return PreparedTaskData(
            route=TaskRoute(is_pooled_task=False),
            store=store,
            train_bundle=train_bundle,
            val_bundle=val_bundle,
            test_bundle=test_bundle,
        )

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
        batch_size = int(model_info.get("loader_batch_size", 16))
        num_workers = int(model_info.get("loader_num_workers", 4))
        train_loader = get_dataloader(prepared.train_bundle, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_predict_loader = get_dataloader(prepared.train_bundle, batch_size=batch_size, num_workers=num_workers)
        val_loader = (
            get_dataloader(prepared.val_bundle, batch_size=batch_size, num_workers=num_workers) if use_validation else None
        )
        test_loader = get_dataloader(prepared.test_bundle, batch_size=batch_size, num_workers=num_workers)

        log("      Training attention model...")
        t0 = time.time()
        trained = train_model_loader(
            model_info, train_loader, config.batch_kind, val_loader=val_loader,
        )
        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")

        pred_train = trained.predict_loader(train_predict_loader, batch_kind=config.batch_kind)
        pred_val = (
            trained.predict_loader(val_loader, batch_kind=config.batch_kind) if use_validation else None
        )
        pred_test = trained.predict_loader(test_loader, batch_kind=config.batch_kind)

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

    def save_trained_model(self, trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
        return save_attention_checkpoint(trained, run_dir, checkpoint_path=model_path)


def _select_training_route(config) -> TrainingRoute:
    """Resolve a config into the route object that owns prepare/train/save."""
    if config.batch_kind == "protein_residue":
        return ResidueTasksRoute()
    return GlobalRoute()


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
    return TaskRoute(is_pooled_task=config.batch_kind != "protein_residue")


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
) -> list[str]:
    """Return which candidate models already have finished artifacts on disk."""
    return [
        model_name for model_name in model_map
        if is_run_complete(
            build_results_run_dir(output_dir, split_variant, task_name, emb_name, model_name)
        )
    ]


def _dataset_sizes(prepared: PreparedTaskData) -> tuple[int, int, int]:
    """Return train/val/test sizes independent of the chosen training route."""
    if prepared.train_bundle is not None:
        return (
            len(prepared.train_bundle.dataset),
            len(prepared.val_bundle.dataset),
            len(prepared.test_bundle.dataset),
        )
    if prepared.x_train is not None:
        return len(prepared.x_train), len(prepared.x_val), len(prepared.x_test)
    if prepared.x_train_flat_ec is not None:
        return (
            len(prepared.x_train_flat_ec),
            len(prepared.x_val_flat_ec),
            len(prepared.x_test_flat_ec),
        )
    raise RuntimeError("PreparedTaskData has no data.")


def _cleanup_prepared_task_data(prepared: PreparedTaskData | None) -> None:
    """Release references to per-task embedding stores before the next task."""
    if prepared is not None:
        prepared.store = None


def _run_task(
    *,
    task_scope: str,
    task_name: str,
    emb_name: str,
    emb_info: dict,
    output_dir: Path,
    model_prefix_filter: str | None,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    skip_label_kinds: set[str],
    residue_store_mode: str,
    split_variant_filter: str | None,
    embedding_mode_filters: set[str],
    overwrite_results: bool,
    training_route_override: TrainingRoute | None = None,
    model_map_fn=None,
) -> None:
    """Train all selected models for one task/embedding pair."""
    config = get_task_config(task_name)
    task_type = "classification" if config.label_kind == "binary" else "regression"

    if config.label_kind in skip_label_kinds:
        log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' filtered by CLI)")
        return


    split_variant = split_variant_filter or "random"
    if split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        log(
            f"  Skipping '{task_name}' (requested split '{split_variant}' is not available; "
            f"available: {available})"
        )
        return

    log(f"\n  Task: {task_name} | type: {task_type} | split: {split_variant}")
    prepared: PreparedTaskData | None = None
    try:
        if model_map_fn is not None:
            model_map = model_map_fn(task_type, config.label_kind, model_prefix_filter)
        else:
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

        training_route = training_route_override if training_route_override is not None else _select_training_route(config)
        needs_flat_ec = any(info.get("flat_ec_labels", False) for info in model_map.values())
        needs_hierarchical_ec = any(not info.get("flat_ec_labels", False) for info in model_map.values())
        prepared = training_route.prepare(
            config,
            task_name,
            emb_info["path"],
            split_variant,
            task_rows,
            residue_store_mode=residue_store_mode,
            embedding_dim=int(emb_info["dim"]) if emb_info.get("dim") is not None else None,
            needs_hierarchical_ec=needs_hierarchical_ec,
            needs_flat_ec=needs_flat_ec,
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
            run_dir = build_results_run_dir(output_dir, split_variant, task_name, emb_name, model_name)
            if not overwrite_results and is_run_complete(run_dir):
                log(f"    [{model_name}] Already trained — skipping.")
                continue
            if overwrite_results and run_dir.exists():
                shutil.rmtree(run_dir)

            log(
                f"    [{model_name}] n_train={n_train}, n_test={n_test} "
                f"| route={training_route.name}"
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
                task_config=config,
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

            canonical_model_path = build_model_artifact_path(
                run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                extension=".safetensors" if isinstance(training_route, AttentionRoute) else ".pt",
            )
            model_file = training_route.save_trained_model(trained, run_dir, model_path=canonical_model_path)
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

            log(f"      Saved → {run_dir.relative_to(output_dir)} [{artifact_stem}]")
    except Exception as task_exc:
        log(f"  [ERROR] Task '{task_name}' failed: {task_exc}")
        log(traceback.format_exc())
    finally:
        _cleanup_prepared_task_data(prepared)


def _select_attention_model_map(
    task_type: str,
    label_kind: str,
    model_prefix_filter: str | None,
) -> dict[str, dict]:
    model_map = get_attention_model_map(task_type, label_kind)
    if model_prefix_filter is None:
        return model_map
    return {k: v for k, v in model_map.items() if k.startswith(model_prefix_filter)}


def main(
    embeddings_json: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    model_prefix_filter: str | None = None,
    task_name_filters: set[str] | None = None,
    task_scope_filters: set[str] | None = None,
    skip_label_kinds: set[str] | None = None,
    allowed_ids_path: str | None = None,
    residue_store_mode: str = "auto",
    split_variant_filter: str | None = None,
    embedding_mode_filters: set[str] | None = None,
    overwrite_results: bool = False,
    attention_embeddings_json: str | None = None,
):
    """Train all selected embedding-model and task combinations from the task registry."""
    log(f"[startup] Python started. emb_name={emb_name_filter}")
    if model_prefix_filter is not None:
        log(f"[startup] Filtering model names by prefix: {model_prefix_filter}")

    task_name_filters = task_name_filters or set()
    if task_name_filters:
        log(f"[startup] Restricting tasks to: {sorted(task_name_filters)}")
    task_scope_filters = task_scope_filters or {"properties"}
    invalid_task_scopes = task_scope_filters - {"properties", "real_case"}
    if invalid_task_scopes:
        raise ValueError(f"Unknown task scopes: {sorted(invalid_task_scopes)}")
    log(f"[startup] Restricting task scopes to: {sorted(task_scope_filters)}")

    skip_label_kinds = skip_label_kinds or set()
    if skip_label_kinds:
        log(f"[startup] Skipping label kinds: {sorted(skip_label_kinds)}")
    log(f"[startup] Residue store mode: {residue_store_mode}")
    if split_variant_filter is not None:
        log(f"[startup] Restricting split variant to: {split_variant_filter}")
    embedding_mode_filters = embedding_mode_filters or set()
    if embedding_mode_filters:
        log(f"[startup] Restricting embedding modes to: {sorted(embedding_mode_filters)}")
    log(f"[startup] Overwrite results: {overwrite_results}")

    embeddings = load_embeddings(embeddings_json)
    selected_tasks = [
        (name, cfg) for name, cfg in TASK_REGISTRY.items()
        if cfg.task_scope in task_scope_filters
    ]
    log(f"[startup] Loaded. {len(embeddings)} embedding models, {len(selected_tasks)} tasks.")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

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

        for task_name, config in selected_tasks:
            if task_name_filters and task_name not in task_name_filters:
                log(f"  Skipping '{task_name}' (not selected by CLI task filter)")
                continue

            _run_task(
                task_scope=config.task_scope,
                task_name=task_name,
                emb_name=emb_name,
                emb_info=emb_info,
                output_dir=output_dir_obj,
                model_prefix_filter=model_prefix_filter,
                allowed_ids=allowed_ids,
                sequence_to_id=sequence_to_id,
                skip_label_kinds=skip_label_kinds,
                residue_store_mode=residue_store_mode,
                split_variant_filter=split_variant_filter,
                embedding_mode_filters=embedding_mode_filters,
                overwrite_results=overwrite_results,
            )

    if attention_embeddings_json is not None:
        attention_embeddings = load_embeddings(attention_embeddings_json)
        if emb_name_filter is not None and emb_name_filter not in attention_embeddings:
            raise ValueError(
                f"Unknown attention embedding '{emb_name_filter}'. "
                f"Available: {', '.join(attention_embeddings)}"
            )
        log(f"[attention] {len(attention_embeddings)} attention embedding models found.")
        attention_route = AttentionRoute()
        for emb_name, emb_info in attention_embeddings.items():
            if emb_name_filter is not None and emb_name != emb_name_filter:
                continue
            log(f"\n=== Attention Embedding: {emb_name} (dim={emb_info['dim']}) ===")
            for task_name, config in selected_tasks:
                if task_name_filters and task_name not in task_name_filters:
                    continue
                _run_task(
                    task_scope=config.task_scope,
                    task_name=task_name,
                    emb_name=emb_name,
                    emb_info=emb_info,
                    output_dir=output_dir_obj,
                    model_prefix_filter=model_prefix_filter,
                    allowed_ids=allowed_ids,
                    sequence_to_id=sequence_to_id,
                    skip_label_kinds=skip_label_kinds,
                    residue_store_mode=residue_store_mode,
                    split_variant_filter=split_variant_filter,
                    embedding_mode_filters=embedding_mode_filters,
                    overwrite_results=overwrite_results,
                    training_route_override=attention_route,
                    model_map_fn=_select_attention_model_map,
                )

    log("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_json", help="Path to embeddings_models_output.json")
    parser.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Directory to write prediction and model files (default: {_DEFAULT_OUTPUT_DIR}).",
    )
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
        "--task_scope",
        action="append",
        choices=("properties", "real_case"),
        default=[],
        help="Train tasks from this benchmark-card section. Repeatable. Defaults to properties.",
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
        "--overwrite_results",
        action="store_true",
        help="Replace existing artifacts for selected runs instead of skipping completed runs.",
    )
    parser.add_argument(
        "--attention_embeddings_json",
        default=None,
        help="Path to a per-residue embeddings JSON. When provided, also trains attention models.",
    )
    args = parser.parse_args()
    main(
        args.embeddings_json,
        args.output_dir,
        args.emb_name,
        model_prefix_filter=args.model_prefix,
        task_name_filters=set(args.task_name),
        task_scope_filters=set(args.task_scope) or None,
        skip_label_kinds=set(args.skip_label_kind),
        allowed_ids_path=args.allowed_ids,
        residue_store_mode=args.residue_store_mode,
        split_variant_filter=args.split_variant,
        embedding_mode_filters=set(args.embedding_mode),
        overwrite_results=args.overwrite_results,
        attention_embeddings_json=args.attention_embeddings_json,
    )
