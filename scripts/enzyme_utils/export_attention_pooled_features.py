"""Export attention-pooled features from trained global attention models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import FlatECLabelBuilder, get_task_config

from models.attention_models import export_features_with_pooler, get_attention_model_map, load_pooler_from_checkpoint
from models.train import (
    SKIP_LABEL_KINDS,
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
    resolve_registry_name,
    resolve_sequence_index_path,
    resolve_split_variant,
)
from data_loading import build_task_dataset


def is_feature_export_complete(run_dir: Path) -> bool:
    return (
        (run_dir / "train_features.npz").exists()
        and (run_dir / "test_features.npz").exists()
        and (run_dir / "metadata.json").exists()
    )


def _flatten_npz_payload(prefix: str, payload) -> dict[str, np.ndarray]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        flattened = {}
        for key, value in payload.items():
            flattened.update(_flatten_npz_payload(f"{prefix}_{key}", value))
        return flattened
    return {prefix: np.asarray(payload)}


def save_feature_split(
    path: Path,
    sample_ids: list[str],
    features: np.ndarray,
    y_true,
    target_mask=None,
) -> None:
    payload = {
        "sample_id": np.asarray(sample_ids),
        "features": np.asarray(features, dtype=np.float32),
    }
    payload.update(_flatten_npz_payload("y_true", y_true))
    payload.update(_flatten_npz_payload("target_mask", target_mask))
    np.savez(str(path), **payload)


def build_flat_ec_payload(rows: list[dict[str, str]], config, vocabulary_by_level) -> tuple[np.ndarray, np.ndarray]:
    builder = FlatECLabelBuilder(
        target_column=str(config.target_column),
        separator=config.ec_separator,
        vocabulary_by_level=vocabulary_by_level,
    )
    builder._ensure_flat_index()
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    for row in rows:
        payload = builder.build(row)
        targets.append(np.asarray(payload["target"], dtype=np.float32))
        target_masks.append(np.asarray(payload["target_mask"], dtype=bool))
    if not targets:
        empty_dim = len(builder.flat_vocabulary or ())
        return (
            np.zeros((0, empty_dim), dtype=np.float32),
            np.zeros((0, empty_dim), dtype=bool),
        )
    return (
        np.stack(targets).astype(np.float32, copy=False),
        np.stack(target_masks).astype(bool, copy=False),
    )


def main(
    card_path: str,
    embeddings_json: str,
    checkpoints_dir: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    source_model_prefix: str | None = None,
    allowed_ids_path: str | None = None,
    batch_size: int = 16,
):
    log(f"[startup] Attention feature export started. emb_name={emb_name_filter}")
    if source_model_prefix is not None:
        log(f"[startup] Filtering source attention models by prefix: {source_model_prefix}")

    card = json.load(open(card_path))
    embeddings = json.load(open(embeddings_json))
    checkpoints_dir_obj = Path(checkpoints_dir)
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
            f"[startup] Restricting export rows to {len(allowed_ids):,} embedding IDs "
            f"from {allowed_ids_path}"
        )

    if emb_name_filter is not None and emb_name_filter not in embeddings:
        raise ValueError(f"Unknown embedding '{emb_name_filter}'. Available: {', '.join(embeddings)}")

    for emb_name, emb_info in embeddings.items():
        if emb_name_filter is not None and emb_name != emb_name_filter:
            continue

        log(f"\n=== Export Embedding: {emb_name} (dim={emb_info['dim']}) ===")

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
                continue
            if config.label_kind in SKIP_LABEL_KINDS:
                continue

            dataset_info = prop["datasets"][0]
            default_split = dataset_info.get("splits", {}).get("default_split")
            split_variant = resolve_split_variant(default_split)
            if split_variant is None:
                continue

            model_map = get_attention_model_map(task_type, config.label_kind)
            if source_model_prefix is not None:
                model_map = {
                    name: info
                    for name, info in model_map.items()
                    if name.startswith(source_model_prefix)
                }
            if not model_map:
                continue

            for source_model_name in model_map:
                checkpoint_run_dir = checkpoints_dir_obj / f"{emb_name}__{task_name}__{source_model_name}"
                checkpoint_path = checkpoint_run_dir / "model.pt"
                export_run_dir = output_dir_obj / f"{emb_name}__{task_name}__{source_model_name}"

                if not is_run_complete(checkpoint_run_dir) or not checkpoint_path.exists():
                    log(f"  Skipping '{task_name}' / '{source_model_name}' (missing trained checkpoint)")
                    continue
                if is_feature_export_complete(export_run_dir):
                    log(f"  [{task_name} | {source_model_name}] Feature export already complete — skipping.")
                    continue

                log(f"\n  Task: {task_name} | source model: {source_model_name}")
                store = None
                try:
                    train_rows = get_task_rows(config, "train", split_variant)
                    val_rows = get_task_rows(config, "val", split_variant)
                    test_rows = get_task_rows(config, "test", split_variant)
                    if allowed_ids is not None:
                        train_rows = filter_rows_to_allowed_ids(train_rows, config, allowed_ids, sequence_to_id)
                        val_rows = filter_rows_to_allowed_ids(val_rows, config, allowed_ids, sequence_to_id)
                        test_rows = filter_rows_to_allowed_ids(test_rows, config, allowed_ids, sequence_to_id)
                    if not train_rows and not val_rows and not test_rows:
                        log("    Skipping task: allowed-id filtering left no rows in train/val/test.")
                        continue

                    needed_ids, needed_sequences = collect_needed_lookup_keys(train_rows + val_rows + test_rows, config)
                    store = load_lazy_embedding_store(emb_info["path"], needed_ids, needed_sequences)

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

                    pooler, checkpoint = load_pooler_from_checkpoint(checkpoint_path)
                    train_loader = get_dataloader(train_bundle, batch_size=batch_size)
                    val_loader = get_dataloader(val_bundle, batch_size=batch_size) if len(val_bundle.dataset) else None
                    test_loader = get_dataloader(test_bundle, batch_size=batch_size)

                    train_export = export_features_with_pooler(pooler, train_loader, config.batch_kind)
                    val_export = export_features_with_pooler(pooler, val_loader, config.batch_kind) if val_loader is not None else None
                    test_export = export_features_with_pooler(pooler, test_loader, config.batch_kind)

                    if config.label_kind == "multilabel_ec":
                        y_train, target_mask_train = build_flat_ec_payload(
                            train_bundle.dataset.rows,
                            config,
                            train_bundle.label_builder.vocabulary_by_level,
                        )
                        if val_export is not None:
                            y_val, target_mask_val = build_flat_ec_payload(
                                val_bundle.dataset.rows,
                                config,
                                train_bundle.label_builder.vocabulary_by_level,
                            )
                        else:
                            y_val, target_mask_val = None, None
                        y_test, target_mask_test = build_flat_ec_payload(
                            test_bundle.dataset.rows,
                            config,
                            train_bundle.label_builder.vocabulary_by_level,
                        )
                    else:
                        y_train, target_mask_train = get_target_payload(train_bundle)
                        if val_export is not None:
                            y_val, target_mask_val = get_target_payload(val_bundle)
                        else:
                            y_val, target_mask_val = None, None
                        y_test, target_mask_test = get_target_payload(test_bundle)

                    export_run_dir.mkdir(parents=True, exist_ok=True)
                    save_feature_split(
                        export_run_dir / "train_features.npz",
                        train_export.sample_ids,
                        train_export.features,
                        y_train,
                        target_mask_train,
                    )
                    if val_export is not None:
                        save_feature_split(
                            export_run_dir / "val_features.npz",
                            val_export.sample_ids,
                            val_export.features,
                            y_val,
                            target_mask_val,
                        )
                    save_feature_split(
                        export_run_dir / "test_features.npz",
                        test_export.sample_ids,
                        test_export.features,
                        y_test,
                        target_mask_test,
                    )

                    metadata = {
                        "embedding_model": emb_name,
                        "task_name": task_name,
                        "registry_name": registry_name,
                        "source_model_name": source_model_name,
                        "checkpoint_path": str(checkpoint_path),
                        "batch_kind": config.batch_kind,
                        "label_kind": config.label_kind,
                        "input_dim": int(checkpoint["input_dim"]),
                        "feature_dim": int(train_export.features.shape[1]) if train_export.features.size else 0,
                        "allowed_ids_path": allowed_ids_path,
                    }
                    with open(export_run_dir / "metadata.json", "w", encoding="utf-8") as handle:
                        json.dump(metadata, handle, indent=2)
                    log(f"    Saved attention features → {export_run_dir.name}")
                finally:
                    del store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to per-residue embeddings_models_output.json")
    parser.add_argument("checkpoints_dir", help="Directory containing trained attention checkpoints")
    parser.add_argument("output_dir", help="Directory to write exported attention features")
    parser.add_argument("--emb_name", default=None, help="Export only this embedding.")
    parser.add_argument(
        "--source_model_prefix",
        default=None,
        help="Export only source attention models whose name starts with this prefix.",
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
        help="Per-task dataloader batch size for exporting attention-pooled features.",
    )
    args = parser.parse_args()
    main(
        args.card_path,
        args.embeddings_json,
        args.checkpoints_dir,
        args.output_dir,
        emb_name_filter=args.emb_name,
        source_model_prefix=args.source_model_prefix,
        allowed_ids_path=args.allowed_ids,
        batch_size=args.batch_size,
    )
