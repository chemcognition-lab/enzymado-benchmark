"""Train attention-pooled global models on per-residue embeddings."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import build_task_dataset, get_task_config

from models.attention_registry import get_attention_model_map
from training.attention_wrappers import save_attention_checkpoint
from training.results_card import (
    build_artifact_stem,
    build_model_artifact_path,
    ensure_results_card,
    find_result_entry,
    is_result_entry_complete,
    read_results_card,
    update_results_card_locked,
)
from training.train_common import (
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
from training.train_data import (
    get_dataloader,
    get_target_payload,
    load_lazy_safetensors_embedding_store,
)
from training.train_output import (
    make_result_entry,
    read_json,
    save_model_safetensors,
    save_prediction_artifacts,
    write_hparams_file,
    write_metrics_file,
    write_training_log_file,
)
from training.train_runtime import train_model_loader
from training.task_rows import (
    LoadedTaskRows,
    load_task_rows_bundle,
    log_loaded_data_summary,
)


def save_attention_model(trained, run_dir: Path, *, model_path: Path | None = None) -> Path:
    """Persist the full attention checkpoint, including the learned pooler."""
    return save_attention_checkpoint(trained, run_dir, checkpoint_path=model_path)


@dataclass
class AttentionTaskBundles:
    """Built datasets/loaders for one attention task."""

    store: Any
    train_bundle: Any
    val_bundle: Any
    test_bundle: Any


def _is_attention_supported_task(config, skip_label_kinds: set[str]) -> tuple[bool, str | None]:
    """Return whether the task can run on the attention pipeline and why if not."""
    if config.label_kind in skip_label_kinds:
        return False, f"label_kind='{config.label_kind}' filtered by CLI"
    if config.label_kind in SKIP_LABEL_KINDS:
        return False, f"label_kind='{config.label_kind}' not yet supported"
    return True, None


def _select_attention_model_map(task_type: str, label_kind: str, model_prefix_filter: str | None) -> dict[str, dict]:
    """Select and optionally filter attention models for one task."""
    model_map = get_attention_model_map(task_type, label_kind)
    if model_prefix_filter is None:
        return model_map
    return {
        name: info
        for name, info in model_map.items()
        if name.startswith(model_prefix_filter)
    }


def _completed_attention_models(
    model_map: dict[str, dict],
    output_dir: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    task_scope: str = "properties",
    results_card: dict[str, Any] | None = None,
) -> list[str]:
    """Return which candidate attention models already have finished artifacts."""
    return [
        model_name for model_name in model_map
        if is_run_complete(
            build_result_run_dir(output_dir, split_variant, task_name, emb_name, model_name)
        )
        or is_result_entry_complete(
            find_result_entry(
                results_card or {},
                task_scope=task_scope,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
            )
        )
    ]


def _build_attention_task_bundles(
    registry_name: str,
    config,
    split_variant: str,
    emb_path: str | Path,
    task_rows: LoadedTaskRows,
) -> AttentionTaskBundles:
    """Load the lazy store and build train/val/test bundles for one attention task."""
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
        registry_name,
        store,
        split_name="train",
        split_variant=split_variant,
        missing_policy="skip",
    )
    ec_vocabulary = None
    if config.label_kind == "multilabel_ec":
        ec_vocabulary = train_bundle.label_builder.vocabulary_by_level
    val_bundle = build_task_dataset(
        registry_name,
        store,
        split_name="val",
        split_variant=split_variant,
        missing_policy="skip",
        ec_vocabulary=ec_vocabulary,
    )
    test_bundle = build_task_dataset(
        registry_name,
        store,
        split_name="test",
        split_variant=split_variant,
        missing_policy="skip",
        ec_vocabulary=ec_vocabulary,
    )
    log(f"    Datasets built ({time.time()-t0:.1f}s)")
    return AttentionTaskBundles(
        store=store,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
    )


def _train_attention_model(
    model_info: dict,
    bundles: AttentionTaskBundles,
    config,
    batch_size: int,
    use_validation: bool,
):
    """Train one attention model and return predictions plus labels for saving."""
    train_loader = get_dataloader(bundles.train_bundle, batch_size=batch_size, shuffle=True)
    train_predict_loader = get_dataloader(bundles.train_bundle, batch_size=batch_size)
    val_loader = get_dataloader(bundles.val_bundle, batch_size=batch_size) if use_validation else None
    test_loader = get_dataloader(bundles.test_bundle, batch_size=batch_size)

    t0 = time.time()
    log("      Training attention model...")
    trained = train_model_loader(
        model_info,
        train_loader,
        config.batch_kind,
        val_loader=val_loader,
    )
    log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")

    pred_train = trained.predict_loader(train_predict_loader, batch_kind=config.batch_kind)
    pred_val = trained.predict_loader(val_loader, batch_kind=config.batch_kind) if use_validation else None
    pred_test = trained.predict_loader(test_loader, batch_kind=config.batch_kind)

    y_train, target_mask_train = get_target_payload(bundles.train_bundle)
    y_val, target_mask_val = get_target_payload(bundles.val_bundle) if use_validation else (None, None)
    y_test, target_mask_test = get_target_payload(bundles.test_bundle)
    return (
        trained,
        pred_train,
        pred_val,
        pred_test,
        y_train,
        y_val,
        y_test,
        target_mask_train,
        target_mask_val,
        target_mask_test,
    )


def _run_attention_task(
    *,
    task_scope: str,
    task_name: str,
    prop: dict,
    emb_name: str,
    emb_info: dict,
    results_card_path: Path,
    output_dir: Path,
    model_prefix_filter: str | None,
    skip_label_kinds: set[str],
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    batch_size: int,
    split_variant_filter: str | None,
    overwrite_results: bool,
    benchmark_card: dict[str, Any] | None = None,
    results_card: dict[str, Any] | None = None,
) -> None:
    """Train all attention models for one task/embedding pair."""
    task_type = prop.get("task_type")
    if not task_type:
        raise ValueError(f"'{task_name}' is missing 'task_type' in card")

    registry_name = resolve_registry_name(task_name)
    if registry_name is None:
        log(f"  Skipping '{task_name}' (not in TASK_REGISTRY)")
        return

    config = get_task_config(registry_name)
    is_supported, reason = _is_attention_supported_task(config, skip_label_kinds)
    if not is_supported:
        log(f"  Skipping '{task_name}' ({reason})")
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
    bundles: AttentionTaskBundles | None = None
    try:
        model_map = _select_attention_model_map(task_type, config.label_kind, model_prefix_filter)
        if not model_map:
            log("    No attention models selected after filters — skipping task.")
            return

        if not overwrite_results:
            already_done = _completed_attention_models(
                model_map,
                output_dir,
                split_variant,
                task_name,
                emb_name,
                task_scope=task_scope,
                results_card=results_card,
            )
            if len(already_done) == len(model_map):
                log(f"    All {len(already_done)} attention models already trained — skipping task.")
                return
            if already_done:
                log(f"    {len(already_done)}/{len(model_map)} attention models already done: {already_done}")
        else:
            log("    Overwrite enabled: existing attention artifacts for selected models will be replaced.")

        task_rows = load_task_rows_bundle(
            config,
            split_variant,
            allowed_ids,
            sequence_to_id,
            log_loading=False,
        )
        if not task_rows.train_rows and not task_rows.val_rows and not task_rows.test_rows:
            log("    Skipping task: allowed-id filtering left no rows in train/val/test.")
            return

        bundles = _build_attention_task_bundles(
            registry_name,
            config,
            split_variant,
            emb_info["path"],
            task_rows,
        )
        n_train = len(bundles.train_bundle.dataset)
        n_val = len(bundles.val_bundle.dataset)
        n_test = len(bundles.test_bundle.dataset)
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

            log(f"    [{model_name}] n_train={n_train}, n_test={n_test} | attention pooled")
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

            (
                trained,
                pred_train,
                pred_val,
                pred_test,
                y_train,
                y_val,
                y_test,
                target_mask_train,
                target_mask_val,
                target_mask_test,
            ) = _train_attention_model(model_info, bundles, config, batch_size, use_validation)

            (
                train_pred_file,
                train_metrics,
            ) = save_prediction_artifacts(
                y_train,
                pred_train,
                target_mask_train,
                run_dir=run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                split_name="train",
                task_type=task_type,
            )
            metrics_by_split = {"train": train_metrics}
            val_pred_file = None
            if use_validation:
                (
                    val_pred_file,
                    val_metrics,
                ) = save_prediction_artifacts(
                    y_val,
                    pred_val,
                    target_mask_val,
                    run_dir=run_dir,
                    task_name=task_name,
                    emb_name=emb_name,
                    model_name=model_name,
                    split_variant=split_variant,
                    split_name="val",
                    task_type=task_type,
                )
                metrics_by_split["val"] = val_metrics
            (
                test_pred_file,
                test_metrics,
            ) = save_prediction_artifacts(
                y_test,
                pred_test,
                target_mask_test,
                run_dir=run_dir,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                split_name="test",
                task_type=task_type,
            )
            metrics_by_split["test"] = test_metrics
            metrics_file = write_metrics_file(run_dir, metrics_by_split)
            training_log_file = write_training_log_file(trained, run_dir)
            model_file = save_attention_model(trained, run_dir)
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
                task_scope=task_scope,
                train_pred_file=train_pred_file,
                test_pred_file=test_pred_file,
                val_pred_file=val_pred_file,
                train_pred_safetensors_file=train_pred_file,
                test_pred_safetensors_file=test_pred_file,
                val_pred_safetensors_file=val_pred_file,
                model_safetensors_file=model_safetensors_file,
                metrics=metrics_by_split,
                metrics_file=metrics_file,
                hparams_file=hparams_file,
                training_log_file=training_log_file,
            )
            update_results_card_locked(
                results_card_path,
                task_scope=task_scope,
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
        if bundles is not None:
            bundles.store = None


def main(
    card_path: str,
    embeddings_json: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    model_prefix_filter: str | None = None,
    task_name_filters: set[str] | None = None,
    task_scope_filters: set[str] | None = None,
    skip_label_kinds: set[str] | None = None,
    allowed_ids_path: str | None = None,
    batch_size: int = 16,
    split_variant_filter: str | None = None,
    overwrite_results: bool = False,
    *,
    results_card_path: str,
):
    log(f"[startup] Attention training started. emb_name={emb_name_filter}")
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
    if split_variant_filter is not None:
        log(f"[startup] Restricting split variant to: {split_variant_filter}")
    log(f"[startup] Overwrite results: {overwrite_results}")

    card = read_json(card_path)
    embeddings = read_json(embeddings_json)
    selected_task_count = sum(
        len(card.get(scope_name, {}) or {})
        for scope_name in task_scope_filters
    )
    log(f"[startup] JSON loaded. {len(embeddings)} embedding models, {selected_task_count} tasks.")

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

        log(f"\n=== Attention Embedding: {emb_name} (dim={emb_info['dim']}) ===")

        for task_scope in ("properties", "real_case"):
            if task_scope not in task_scope_filters:
                continue
            for task_name, prop in (card.get(task_scope, {}) or {}).items():
                if task_name_filters and task_name not in task_name_filters:
                    log(f"  Skipping '{task_name}' (not selected by CLI task filter)")
                    continue
                _run_attention_task(
                    task_scope=task_scope,
                    task_name=task_name,
                    prop=prop,
                    emb_name=emb_name,
                    emb_info=emb_info,
                    results_card_path=results_card_path_obj,
                    output_dir=output_dir_obj,
                    model_prefix_filter=model_prefix_filter,
                    skip_label_kinds=skip_label_kinds,
                    allowed_ids=allowed_ids,
                    sequence_to_id=sequence_to_id,
                    batch_size=batch_size,
                    split_variant_filter=split_variant_filter,
                    overwrite_results=overwrite_results,
                    benchmark_card=card,
                    results_card=results_card,
                )

    log(f"\nDone. Dataset card read from {card_path_obj}")
    log(f"Done. Attention results written to {results_card_path_obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to per-residue embeddings_models_output.json")
    parser.add_argument("output_dir", help="Directory to write prediction and model files")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding.")
    parser.add_argument(
        "--model_prefix",
        default=None,
        help="Train only models whose name starts with this prefix, e.g. 'attention_linear_'.",
    )
    parser.add_argument(
        "--task_name",
        action="append",
        default=[],
        help="Train only this task name. Repeatable.",
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
        "--batch_size",
        type=int,
        default=16,
        help="Per-task dataloader batch size for attention training.",
    )
    parser.add_argument(
        "--split_variant",
        choices=("random", "similarity", "loco"),
        default=None,
        help="Train only this split variant. Defaults to each task's card default_split.",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Replace existing artifacts for selected attention runs instead of skipping completed runs.",
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
        emb_name_filter=args.emb_name,
        model_prefix_filter=args.model_prefix,
        task_name_filters=set(args.task_name),
        task_scope_filters=set(args.task_scope) or None,
        skip_label_kinds=set(args.skip_label_kind),
        allowed_ids_path=args.allowed_ids,
        batch_size=args.batch_size,
        split_variant_filter=args.split_variant,
        overwrite_results=args.overwrite_results,
        results_card_path=args.results_card,
    )
