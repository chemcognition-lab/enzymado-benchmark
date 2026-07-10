"""Benchmark orchestration: run all model/task/embedding combinations."""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "predictions"
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loading import (
    LoaderTaskData,
    PooledTaskData,
    TASK_REGISTRY,
    get_dataloader,
    get_target_payload,
    get_task_config,
    load_allowed_ids,
    load_embeddings,
    load_sequence_to_id_index,
    load_task_data,
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
from evaluation.results_layout import (
    build_artifact_stem,
    build_model_artifact_path,
    build_results_run_dir,
    canonical_embedding_mode,
    is_run_complete,
)
from training.model_registry import get_attention_model_map, get_model_map
from training.predict import (
    ModelRunOutputs,
    ResidueTrainingPayload,
    SavedModelRun,
    predict_from_arrays,
    predict_from_loaders,
)
from training.train import (
    ATTENTION_ROUTE,
    GLOBAL_ROUTE,
    RESIDUE_ROUTE,
    TrainingRouteSpec,
    _dataset_sizes,
    _select_global_arrays,
    _select_training_route,
    train_attention,
    train_global,
    train_residue,
)


def save_split_predictions(
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
    """Save predictions and compute metrics for each split."""
    split_payloads = [("train", outputs.y_train, outputs.pred_train, outputs.target_mask_train)]
    if use_validation:
        split_payloads.append(("val", outputs.y_val, outputs.pred_val, outputs.target_mask_val))
    split_payloads.append(("test", outputs.y_test, outputs.pred_test, outputs.target_mask_test))

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


def save_residue_predictions(
    trained_payload: ResidueTrainingPayload,
    *,
    prepared,
    use_validation: bool,
    config,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    task_type: str,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Generate and save residue predictions split-by-split to manage memory."""
    split_jobs = [
        ("train", prepared.train_bundle, trained_payload.loaders.train_predict_loader),
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


def run_model(
    route: TrainingRouteSpec,
    *,
    model_name: str,
    emb_name: str,
    model_info: dict,
    prepared: PooledTaskData | LoaderTaskData,
    config,
    use_validation: bool,
    run_dir: Path,
    task_name: str,
    split_variant: str,
    task_type: str,
) -> SavedModelRun:
    """Train one model and generate/save predictions for all splits."""
    trained_payload = route.train(
        model_name=model_name,
        emb_name=emb_name,
        model_info=model_info,
        data=prepared,
        config=config,
        use_validation=use_validation,
        run_dir=run_dir,
    )

    save_kwargs = dict(
        run_dir=run_dir,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        split_variant=split_variant,
        task_type=task_type,
        use_validation=use_validation,
    )

    if route.name == "residue":
        log("      Predicting and saving residue outputs split-by-split...")
        prediction_paths, metrics_by_split = save_residue_predictions(
            trained_payload,
            prepared=prepared,
            use_validation=use_validation,
            config=config,
            **save_kwargs,
        )
        trained = trained_payload.trained

    elif route.name == "attention":
        log("      Predicting with attention model...")
        batch_size = int(model_info.get("loader_batch_size", 16))
        num_workers = int(model_info.get("loader_num_workers", 4))
        train_predict_loader = get_dataloader(
            prepared.train_bundle, batch_size=batch_size, num_workers=num_workers,
        )
        val_loader = (
            get_dataloader(prepared.val_bundle, batch_size=batch_size, num_workers=num_workers)
            if use_validation and prepared.val_bundle is not None else None
        )
        test_loader = get_dataloader(
            prepared.test_bundle, batch_size=batch_size, num_workers=num_workers,
        )
        outputs = predict_from_loaders(
            trained_payload,
            train_predict_loader,
            val_loader,
            test_loader,
            prepared,
            use_validation,
            config.batch_kind,
        )
        prediction_paths, metrics_by_split = save_split_predictions(outputs, **save_kwargs)
        trained = outputs.trained

    else:  # GlobalRoute
        log("      Predicting with pooled model...")
        x_tr, y_tr, x_va, y_va, x_te, y_te, m_tr, m_va, m_te = _select_global_arrays(
            model_info, prepared
        )
        outputs = predict_from_arrays(
            trained_payload,
            x_tr, y_tr,
            x_va, y_va,
            x_te, y_te,
            use_validation,
            m_tr, m_va, m_te,
        )
        prediction_paths, metrics_by_split = save_split_predictions(outputs, **save_kwargs)
        trained = outputs.trained

    return SavedModelRun(
        trained=trained,
        prediction_paths=prediction_paths,
        metrics_by_split=metrics_by_split,
    )


def _select_model_map(
    task_type: str,
    label_kind: str,
    model_prefix_filter: str | None,
) -> dict[str, dict]:
    model_map = get_model_map(task_type, label_kind)
    if model_prefix_filter is None:
        return model_map
    return {k: v for k, v in model_map.items() if k.startswith(model_prefix_filter)}


def _select_attention_model_map(
    task_type: str,
    label_kind: str,
    model_prefix_filter: str | None,
) -> dict[str, dict]:
    model_map = get_attention_model_map(task_type, label_kind)
    if model_prefix_filter is None:
        return model_map
    return {k: v for k, v in model_map.items() if k.startswith(model_prefix_filter)}


def _filter_model_map_by_embedding_modes(
    model_map: dict[str, dict],
    task_name: str,
    embedding_mode_filters: set[str],
) -> dict[str, dict]:
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
    return [
        model_name for model_name in model_map
        if is_run_complete(build_results_run_dir(output_dir, split_variant, task_name, emb_name, model_name))
    ]


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
    training_route_override: TrainingRouteSpec | None = None,
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
            f"  Skipping '{task_name}' (requested split '{split_variant}' is not available;"
            f" available: {available})"
        )
        return

    log(f"\n  Task: {task_name} | type: {task_type} | split: {split_variant}")
    prepared: PooledTaskData | LoaderTaskData | None = None
    try:
        if model_map_fn is not None:
            model_map = model_map_fn(task_type, config.label_kind, model_prefix_filter)
        else:
            model_map = _select_model_map(task_type, config.label_kind, model_prefix_filter)
        model_map = _filter_model_map_by_embedding_modes(model_map, task_name, embedding_mode_filters)
        if not model_map:
            log("    No models selected after filters — skipping task.")
            return

        if not overwrite_results:
            already_done = _completed_models(model_map, output_dir, split_variant, task_name, emb_name)
            if len(already_done) == len(model_map):
                log(f"    All {len(already_done)} models already trained — skipping task.")
                return
            if already_done:
                log(f"    {len(already_done)}/{len(model_map)} models already done: {already_done}")
        else:
            log("    Overwrite enabled: existing artifacts for selected models will be replaced.")

        embedding_dim = int(emb_info["dim"]) if emb_info.get("dim") is not None else None

        if training_route_override is ATTENTION_ROUTE:
            prepared = ATTENTION_ROUTE.prepare(
                config,
                task_name,
                emb_info["path"],
                split_variant,
                allowed_ids=allowed_ids,
                sequence_to_id=sequence_to_id,
                embedding_dim=embedding_dim,
            )
            training_route: TrainingRouteSpec = ATTENTION_ROUTE
        else:
            prepared = load_task_data(
                task_name,
                emb_info["path"],
                split_variant,
                allowed_ids=allowed_ids,
                sequence_to_id=sequence_to_id,
                residue_store_mode=residue_store_mode,
                embedding_dim=embedding_dim,
            )
            training_route = GLOBAL_ROUTE if isinstance(prepared, PooledTaskData) else RESIDUE_ROUTE

        n_train, n_val, n_test = _dataset_sizes(prepared)

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

            log(f"    [{model_name}] n_train={n_train}, n_test={n_test} | route={training_route.name}")
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_stem = build_artifact_stem(task_name, emb_name, model_name, split_variant)

            write_hparams_file(
                run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                model_info=model_info,
                task_config=config,
            )
            saved_run = run_model(
                training_route,
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
            write_metrics_file(run_dir, saved_run.metrics_by_split)
            write_training_log_file(trained, run_dir)

            canonical_model_path = build_model_artifact_path(
                run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                extension=".safetensors" if training_route.name == "attention" else ".pt",
            )
            if training_route.name == "attention":
                save_model_safetensors(trained, run_dir, model_path=canonical_model_path)
            else:
                save_model(trained, run_dir)
                save_model_safetensors(
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
                    f"      Early stopping: best_epoch={trained.best_epoch_},"
                    f" best_val_loss={trained.best_val_loss_:.6f}"
                )
            log(f"      Saved → {run_dir.relative_to(output_dir)} [{artifact_stem}]")

    except Exception as task_exc:
        log(f"  [ERROR] Task '{task_name}' failed: {task_exc}")
        log(traceback.format_exc())
    finally:
        if prepared is not None:
            prepared.cleanup()


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
) -> None:
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
            f"[startup] Restricting training rows to {len(allowed_ids):,} embedding IDs"
            f" from {allowed_ids_path}"
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
                f"Unknown attention embedding '{emb_name_filter}'."
                f" Available: {', '.join(attention_embeddings)}"
            )
        log(f"[attention] {len(attention_embeddings)} attention embedding models found.")
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
                    training_route_override=ATTENTION_ROUTE,
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
        choices=("attention", "no_attention"),
        default=[],
        help="Train only model/task combinations in this results embedding mode.",
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
