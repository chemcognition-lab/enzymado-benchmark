"""Train attention-pooled global models on per-residue embeddings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import get_task_config

from models.attention_models import get_attention_model_map, save_attention_checkpoint
from models.train import (
    SKIP_LABEL_KINDS,
    build_result_run_dir,
    collect_needed_lookup_keys,
    filter_rows_to_allowed_ids,
    get_dataloader,
    get_target_payload,
    get_task_rows,
    is_run_complete,
    load_allowed_ids,
    load_lazy_embedding_store,
    load_sequence_to_id_index,
    log,
    make_predictions_file,
    make_result_entry,
    read_json,
    resolve_registry_name,
    resolve_sequence_index_path,
    resolve_split_variant,
    train_model_loader,
    update_card_locked,
)
from data_loading import build_task_dataset


def save_attention_model(trained, run_dir: Path) -> Path:
    """Persist the full attention checkpoint, including the learned pooler."""
    return save_attention_checkpoint(trained, run_dir)


def main(
    card_path: str,
    embeddings_json: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    model_prefix_filter: str | None = None,
    skip_label_kinds: set[str] | None = None,
    allowed_ids_path: str | None = None,
    batch_size: int = 16,
):
    log(f"[startup] Attention training started. emb_name={emb_name_filter}")
    if model_prefix_filter is not None:
        log(f"[startup] Filtering model names by prefix: {model_prefix_filter}")
    skip_label_kinds = skip_label_kinds or set()
    if skip_label_kinds:
        log(f"[startup] Skipping label kinds: {sorted(skip_label_kinds)}")

    card = read_json(card_path)
    embeddings = read_json(embeddings_json)
    log(f"[startup] JSON loaded. {len(embeddings)} embedding models, {len(card['properties'])} tasks.")

    card_path_obj = Path(card_path)
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

        log(f"\n=== Attention Embedding: {emb_name} (dim={emb_info['dim']}) ===")

        for task_name, prop in card["properties"].items():
            task_type = prop.get("task_type")
            if not task_type:
                raise ValueError(f"'{task_name}' is missing 'task_type' in card")

            registry_name = resolve_registry_name(task_name)
            if registry_name is None:
                log(f"  Skipping '{task_name}' (not in TASK_REGISTRY)")
                continue

            config = get_task_config(registry_name)
            if config.batch_kind == "protein_residue":
                log(f"  Skipping '{task_name}' (site tasks stay on the existing pipeline)")
                continue
            if config.label_kind in skip_label_kinds:
                log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' filtered by CLI)")
                continue
            if config.label_kind in SKIP_LABEL_KINDS:
                log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' not yet supported)")
                continue

            dataset_info = prop["datasets"][0]
            default_split = dataset_info.get("splits", {}).get("default_split")
            split_variant = resolve_split_variant(default_split)
            if split_variant is None:
                log(f"  Skipping '{task_name}' (no usable split defined in card)")
                continue

            log(f"\n  Task: {task_name} | type: {task_type} | split: {default_split}")
            store = None
            try:
                model_map = get_attention_model_map(task_type, config.label_kind)
                if model_prefix_filter is not None:
                    model_map = {
                        name: info
                        for name, info in model_map.items()
                        if name.startswith(model_prefix_filter)
                    }
                if not model_map:
                    log("    No attention models selected after filters — skipping task.")
                    continue

                already_done = [
                    model_name for model_name in model_map
                    if is_run_complete(
                        build_result_run_dir(output_dir_obj, split_variant, task_name, emb_name, model_name)
                    )
                ]
                if len(already_done) == len(model_map):
                    log(f"    All {len(already_done)} attention models already trained — skipping task.")
                    continue
                if already_done:
                    log(f"    {len(already_done)}/{len(model_map)} attention models already done: {already_done}")

                train_rows = get_task_rows(config, "train", split_variant)
                val_rows = get_task_rows(config, "val", split_variant)
                test_rows = get_task_rows(config, "test", split_variant)
                if allowed_ids is not None:
                    orig_train_rows = len(train_rows)
                    orig_val_rows = len(val_rows)
                    orig_test_rows = len(test_rows)
                    train_rows = filter_rows_to_allowed_ids(train_rows, config, allowed_ids, sequence_to_id)
                    val_rows = filter_rows_to_allowed_ids(val_rows, config, allowed_ids, sequence_to_id)
                    test_rows = filter_rows_to_allowed_ids(test_rows, config, allowed_ids, sequence_to_id)
                    log(
                        "    Applied allowed-id filter:"
                        f" train={len(train_rows):,}/{orig_train_rows:,}"
                        f" val={len(val_rows):,}/{orig_val_rows:,}"
                        f" test={len(test_rows):,}/{orig_test_rows:,}"
                    )
                if not train_rows and not val_rows and not test_rows:
                    log("    Skipping task: allowed-id filtering left no rows in train/val/test.")
                    continue

                needed_ids, needed_sequences = collect_needed_lookup_keys(train_rows + val_rows + test_rows, config)
                log(
                    f"    Rows loaded: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} "
                    f"| need {len(needed_ids)} ids, {len(needed_sequences)} seqs"
                )

                t0 = time.time()
                store = load_lazy_embedding_store(emb_info["path"], needed_ids, needed_sequences)
                log(
                    f"    Lazy store ready ({time.time()-t0:.1f}s) — "
                    f"{len(store._available_ids)} ids, {len(store._seq_to_h5key)} seqs indexed"
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

                n_train = len(train_bundle.dataset)
                n_val = len(val_bundle.dataset)
                n_test = len(test_bundle.dataset)
                orig_train = len(train_rows)
                orig_val = len(val_rows)
                orig_test = len(test_rows)
                orig_total = orig_train + orig_val + orig_test
                kept_total = n_train + n_val + n_test
                dropped = orig_total - kept_total
                dropped_pct = (100 * dropped / orig_total) if orig_total > 0 else 0.0
                log(
                    f"    Data summary:"
                    f"\n      original : train={orig_train:,}  val={orig_val:,}  test={orig_test:,}  total={orig_total:,}"
                    f"\n      kept     : train={n_train:,}  val={n_val:,}  test={n_test:,}  total={kept_total:,}"
                    f"\n      dropped  : {dropped:,} ({dropped_pct:.1f}% of original)"
                    + (f"  ← missing embeddings/ligand features" if dropped > 0 else "")
                )

                if n_train == 0 or n_test == 0:
                    log(f"    Skipping '{task_name}': empty dataset after filtering.")
                    continue

                use_validation = n_val > 0
                if use_validation:
                    log(f"    Validation enabled: n_val={n_val:,}")
                else:
                    log("    Validation disabled: empty val split after filtering")

                train_loader = get_dataloader(train_bundle, batch_size=batch_size)
                val_loader = get_dataloader(val_bundle, batch_size=batch_size) if use_validation else None
                test_loader = get_dataloader(test_bundle, batch_size=batch_size)

                for model_name, model_info in model_map.items():
                    run_dir = build_result_run_dir(
                        output_dir_obj, split_variant, task_name, emb_name, model_name
                    )
                    if is_run_complete(run_dir):
                        log(f"    [{model_name}] Already trained — skipping.")
                        continue

                    log(f"    [{model_name}] n_train={n_train}, n_test={n_test} | attention pooled")
                    run_dir.mkdir(parents=True, exist_ok=True)

                    t0 = time.time()
                    log("      Training attention model...")
                    trained = train_model_loader(
                        model_info,
                        train_loader,
                        config.batch_kind,
                        val_loader=val_loader,
                    )
                    log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")

                    pred_train = trained.predict_loader(train_loader, batch_kind=config.batch_kind)
                    pred_val = trained.predict_loader(val_loader, batch_kind=config.batch_kind) if use_validation else None
                    pred_test = trained.predict_loader(test_loader, batch_kind=config.batch_kind)

                    y_train_to_save, target_mask_train_to_save = get_target_payload(train_bundle)
                    y_val_to_save, target_mask_val_to_save = get_target_payload(val_bundle) if use_validation else (None, None)
                    y_test_to_save, target_mask_test_to_save = get_target_payload(test_bundle)

                    make_predictions_file(y_train_to_save, pred_train, run_dir, "train", target_mask_train_to_save)
                    if use_validation:
                        make_predictions_file(y_val_to_save, pred_val, run_dir, "val", target_mask_val_to_save)
                    make_predictions_file(y_test_to_save, pred_test, run_dir, "test", target_mask_test_to_save)
                    model_file = save_attention_model(trained, run_dir)

                    if getattr(trained, "best_epoch_", None) is not None:
                        log(
                            "      Early stopping: "
                            f"best_epoch={trained.best_epoch_}, "
                            f"best_val_loss={trained.best_val_loss_:.6f}"
                        )

                    result_key = f"{emb_name}__{model_name}"
                    result_entry = make_result_entry(emb_name, model_name, model_info, run_dir, model_file)
                    update_card_locked(card_path_obj, task_name, result_key, result_entry)
                    log(f"      Saved → {run_dir.relative_to(output_dir_obj)}")
            except Exception as task_exc:
                import traceback

                log(f"  [ERROR] Task '{task_name}' failed: {task_exc}")
                log(traceback.format_exc())
                continue
            finally:
                del store

    log(f"\nDone. Attention results written to {card_path_obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to per-residue embeddings_models_output.json")
    parser.add_argument("output_dir", help="Directory to write prediction and model files")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding.")
    parser.add_argument(
        "--model_prefix",
        default=None,
        help="Train only models whose name starts with this prefix, e.g. 'attention_sgd_'.",
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
    args = parser.parse_args()
    main(
        args.card_path,
        args.embeddings_json,
        args.output_dir,
        emb_name_filter=args.emb_name,
        model_prefix_filter=args.model_prefix,
        skip_label_kinds=set(args.skip_label_kind),
        allowed_ids_path=args.allowed_ids,
        batch_size=args.batch_size,
    )
