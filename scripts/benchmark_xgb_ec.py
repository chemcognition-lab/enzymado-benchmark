#!/usr/bin/env python3
"""Benchmark EC-number XGBoost variants on the same prepared flat-label data."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import FlatECLabelBuilder, HierarchicalECLabelBuilder, get_task_config
from models.train_common import (
    filter_rows_to_allowed_ids,
    load_allowed_ids,
    load_sequence_to_id_index,
    resolve_sequence_index_path,
)
from models.train_data import (
    build_flat_multilabel_ec_direct,
    collect_needed_lookup_keys,
    get_task_rows,
    load_pooled_h5_subset,
)
from models.train_output import read_json
from models.xgb_wrappers import (
    XGBFlatECBlockClassifier,
    XGBFlatECClassifier,
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sample_rows(rows: list[dict], max_rows: int | None, seed: int) -> list[dict]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    index = rng.choice(len(rows), size=max_rows, replace=False)
    index.sort()
    return [rows[i] for i in index]


def build_flat_ec_arrays(
    card_path: Path,
    embeddings_json: Path,
    emb_name: str,
    allowed_ids_path: Path | None,
    max_train_rows: int | None,
    seed: int,
):
    card = read_json(card_path)
    emb_info = read_json(embeddings_json)[emb_name]
    prop = card["properties"]["ec_number"]
    default_split = prop["datasets"][0].get("splits", {}).get("default_split")
    split_variant = "random" if default_split and "random" in default_split else "similarity"

    config = get_task_config("ec_number")
    train_rows_full = get_task_rows(config, "train", split_variant)
    val_rows = get_task_rows(config, "val", split_variant)
    test_rows = get_task_rows(config, "test", split_variant)

    if allowed_ids_path is not None:
        allowed_ids = load_allowed_ids(allowed_ids_path)
        seq_index = resolve_sequence_index_path()
        sequence_to_id = load_sequence_to_id_index(seq_index) if seq_index is not None else {}
        train_rows_full = filter_rows_to_allowed_ids(train_rows_full, config, allowed_ids, sequence_to_id)
        val_rows = filter_rows_to_allowed_ids(val_rows, config, allowed_ids, sequence_to_id)
        test_rows = filter_rows_to_allowed_ids(test_rows, config, allowed_ids, sequence_to_id)

    train_rows = sample_rows(train_rows_full, max_train_rows, seed)
    if len(train_rows) != len(train_rows_full):
        log(f"Sampled train rows before H5 load: {len(train_rows):,}/{len(train_rows_full):,}")

    needed_ids, needed_sequences = collect_needed_lookup_keys(train_rows + val_rows + test_rows, config)
    t0 = time.perf_counter()
    by_id, by_seq = load_pooled_h5_subset(emb_info["path"], needed_ids, needed_sequences)
    load_seconds = time.perf_counter() - t0

    ec_label_builder = HierarchicalECLabelBuilder(
        target_column=str(config.target_column),
        separator=config.ec_separator,
    ).fit(train_rows_full)
    flat_ec_label_builder = FlatECLabelBuilder(
        target_column=str(config.target_column),
        separator=config.ec_separator,
        vocabulary_by_level=ec_label_builder.vocabulary_by_level,
    )

    X_train, y_train, target_mask_train = build_flat_multilabel_ec_direct(
        train_rows, by_id, by_seq, config, flat_ec_label_builder
    )
    X_val, y_val, target_mask_val = build_flat_multilabel_ec_direct(
        val_rows, by_id, by_seq, config, flat_ec_label_builder
    )
    X_test, y_test, target_mask_test = build_flat_multilabel_ec_direct(
        test_rows, by_id, by_seq, config, flat_ec_label_builder
    )
    return {
        "load_seconds": load_seconds,
        "X_train": X_train,
        "y_train": y_train,
        "target_mask_train": target_mask_train,
        "X_val": X_val,
        "y_val": y_val,
        "target_mask_val": target_mask_val,
        "X_test": X_test,
        "y_test": y_test,
        "target_mask_test": target_mask_test,
    }

def benchmark_current(arrays: dict, params: dict, profile_path: Path | None):
    model = XGBFlatECClassifier(**params)
    profiler = cProfile.Profile() if profile_path is not None else None

    t0 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    model.fit(
        arrays["X_train"],
        arrays["y_train"],
        target_mask=arrays["target_mask_train"],
        X_val=arrays["X_val"],
        y_val=arrays["y_val"],
        target_mask_val=arrays["target_mask_val"],
    )
    if profiler is not None:
        profiler.disable()
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(arrays["X_test"])
    predict_seconds = time.perf_counter() - t1

    if profiler is not None:
        profiler.dump_stats(profile_path)

    trained_columns = sum(m is not None for m in model.models_)
    constant_columns = sum(c is not None for c in model.constant_predictions_)
    return {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "test_shape": list(pred.shape),
        "trained_columns": int(trained_columns),
        "constant_or_skipped_columns": int(constant_columns),
    }


def benchmark_experimental(arrays: dict, params: dict, profile_path: Path | None):
    model = XGBFlatECBlockClassifier(**params)
    profiler = cProfile.Profile() if profile_path is not None else None

    t0 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    model.fit(
        arrays["X_train"],
        arrays["y_train"],
        target_mask=arrays["target_mask_train"],
        X_val=arrays["X_val"],
        y_val=arrays["y_val"],
        target_mask_val=arrays["target_mask_val"],
    )
    if profiler is not None:
        profiler.disable()
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(arrays["X_test"])
    predict_seconds = time.perf_counter() - t1

    if profiler is not None:
        profiler.dump_stats(profile_path)

    return {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "test_shape": list(pred.shape),
        "block_count": len(model.block_slices_),
        "trained_blocks": int(sum(booster is not None for booster in model.block_boosters_)),
        "known_rows_per_block": [int(value) for value in model.block_known_rows_],
    }


def save_profile_summary(profile_path: Path) -> None:
    txt_path = profile_path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(str(profile_path), stream=handle)
        stats.sort_stats("cumtime")
        stats.print_stats(40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", type=Path)
    parser.add_argument("embeddings_json", type=Path)
    parser.add_argument("--emb_name", required=True)
    parser.add_argument("--allowed_ids", type=Path, default=None)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    log("Preparing flat EC arrays...")
    arrays = build_flat_ec_arrays(
        args.card_path,
        args.embeddings_json,
        args.emb_name,
        args.allowed_ids,
        args.max_train_rows,
        args.seed,
    )

    current_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "tree_method": "hist",
        "max_bin": 128,
        "n_jobs": -1,
        "device": "cuda",
        "random_state": args.seed,
        "eval_metric": "logloss",
        "early_stopping_rounds": 20,
    }
    experimental_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.1,
        "n_jobs": -1,
        "device": "cuda",
        "random_state": args.seed,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_bin": 64,
        "block_size": 256,
        "multi_strategy": "multi_output_tree",
        "early_stopping_rounds": 20,
    }

    profile_dir = args.profile_dir
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        current_profile = profile_dir / "current_xgb.prof"
        experimental_profile = profile_dir / "experimental_xgb.prof"
    else:
        current_profile = None
        experimental_profile = None

    log("Benchmarking current one-model-per-column XGB...")
    current = benchmark_current(arrays, current_params, current_profile)
    if current_profile is not None:
        save_profile_summary(current_profile)

    log("Benchmarking experimental multi-output XGB...")
    experimental = benchmark_experimental(arrays, experimental_params, experimental_profile)
    if experimental_profile is not None:
        save_profile_summary(experimental_profile)

    report = {
        "embedding": args.emb_name,
        "load_seconds": arrays["load_seconds"],
        "train_shape": list(arrays["X_train"].shape),
        "val_shape": list(arrays["X_val"].shape),
        "test_shape": list(arrays["X_test"].shape),
        "current": current,
        "experimental": experimental,
        "notes": [
            "Current benchmark uses the existing masked one-model-per-column wrapper.",
            "Train-row sampling happens before H5 loading so benchmark startup reflects the sampled training set.",
            "Experimental benchmark now trains multi-output XGBoost over EC blocks instead of one giant all-label booster.",
            "Each block only requires rows that are fully known for that block, which preserves more data than the old dense-row filtering.",
            "Experimental multi-output GPU XGBoost uses a lower max_bin and bounded block_size to avoid overflowing the histogram-bin limit on wide EC outputs.",
            "Use the cProfile outputs only as coarse Python-side diagnostics; most XGBoost work happens in native code.",
        ],
    }

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
