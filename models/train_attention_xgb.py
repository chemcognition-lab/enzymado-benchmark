"""Train XGBoost models on frozen attention-pooled features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import get_task_config

from models.attention_models import get_attention_model_map
from models.models import XGBClassifier, XGBFlatECClassifier, XGBRegressor
from models.train import (
    SKIP_LABEL_KINDS,
    is_run_complete,
    log,
    make_predictions_file,
    make_result_entry,
    read_json,
    resolve_registry_name,
    train_model,
    update_card_locked,
)


def attention_xgb_model_name(source_model_name: str) -> str:
    if not source_model_name.startswith("attention_"):
        raise ValueError(f"Unexpected source attention model name '{source_model_name}'.")
    return source_model_name.replace("attention_", "attention_xgb_", 1)


def get_attention_xgb_model_info(label_kind: str, task_type: str) -> dict:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if label_kind == "multilabel_ec":
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use the attention XGBoost pipeline.")
        return {
            "wrap": XGBFlatECClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "device": device_name,
                "random_state": 0,
                "eval_metric": "logloss",
                "early_stopping_rounds": 20,
            },
        }
    if task_type == "classification":
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use the attention XGBoost pipeline.")
        return {
            "wrap": XGBClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "device": device_name,
            },
        }
    if task_type == "regression":
        if XGBRegressor is None:
            raise ImportError("xgboost is required to use the attention XGBoost pipeline.")
        return {
            "wrap": XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "device": device_name,
            },
        }
    raise ValueError(f"Unsupported task_type '{task_type}' for attention XGBoost training.")


def load_feature_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    payload = np.load(path, allow_pickle=True)
    X = np.asarray(payload["features"], dtype=np.float32)
    y = np.asarray(payload["y_true"], dtype=np.float32)
    target_mask = np.asarray(payload["target_mask"]) if "target_mask" in payload else None
    if target_mask is not None and target_mask.size == 0:
        target_mask = None
    return X, y, target_mask


def save_xgb_model(trained, run_dir: Path) -> Path:
    path = run_dir / "model.joblib"
    joblib.dump(trained, path)
    return path


def main(
    card_path: str,
    features_dir: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    source_model_prefix: str | None = None,
):
    log(f"[startup] Attention XGBoost training started. emb_name={emb_name_filter}")
    if source_model_prefix is not None:
        log(f"[startup] Using frozen attention models with prefix: {source_model_prefix}")

    card = read_json(card_path)
    features_dir_obj = Path(features_dir)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    discovered_embeddings = sorted({
        path.name.split("__", 1)[0]
        for path in features_dir_obj.iterdir()
        if path.is_dir() and "__" in path.name
    })
    if emb_name_filter is not None and emb_name_filter not in discovered_embeddings:
        raise ValueError(
            f"Unknown embedding '{emb_name_filter}' in feature exports. Available: {', '.join(discovered_embeddings)}"
        )

    for task_name, prop in card["properties"].items():
        task_type = prop.get("task_type")
        if not task_type:
            raise ValueError(f"'{task_name}' is missing 'task_type' in card")

        registry_name = resolve_registry_name(task_name)
        if registry_name is None:
            log(f"Skipping '{task_name}' (not in TASK_REGISTRY)")
            continue

        config = get_task_config(registry_name)
        if config.batch_kind == "protein_residue":
            continue
        if config.label_kind in SKIP_LABEL_KINDS:
            continue

        source_model_map = get_attention_model_map(task_type, config.label_kind)
        if source_model_prefix is not None:
            source_model_map = {
                name: info
                for name, info in source_model_map.items()
                if name.startswith(source_model_prefix)
            }
        if not source_model_map:
            continue

        for source_model_name in source_model_map:
            xgb_model_name = attention_xgb_model_name(source_model_name)
            model_info = get_attention_xgb_model_info(config.label_kind, task_type)

            for emb_name in discovered_embeddings:
                if emb_name_filter is not None and emb_name != emb_name_filter:
                    continue

                feature_run_dir = features_dir_obj / f"{emb_name}__{task_name}__{source_model_name}"
                run_dir = output_dir_obj / f"{emb_name}__{task_name}__{xgb_model_name}"
                if is_run_complete(run_dir):
                    log(f"[{emb_name} | {task_name} | {xgb_model_name}] Already trained — skipping.")
                    continue
                if not (feature_run_dir / "train_features.npz").exists() or not (feature_run_dir / "test_features.npz").exists():
                    continue

                log(f"\nTask: {task_name} | embedding: {emb_name} | source model: {source_model_name}")
                X_train, y_train, target_mask_train = load_feature_split(feature_run_dir / "train_features.npz")
                if (feature_run_dir / "val_features.npz").exists():
                    X_val, y_val, target_mask_val = load_feature_split(feature_run_dir / "val_features.npz")
                    use_validation = len(X_val) > 0
                else:
                    X_val = y_val = target_mask_val = None
                    use_validation = False
                X_test, y_test, target_mask_test = load_feature_split(feature_run_dir / "test_features.npz")

                if len(X_train) == 0 or len(X_test) == 0:
                    log("  Skipping: empty train/test feature arrays.")
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                trained = train_model(
                    model_info,
                    X_train,
                    y_train,
                    X_val=X_val if use_validation else None,
                    y_val=y_val if use_validation else None,
                    target_mask=target_mask_train,
                    target_mask_val=target_mask_val if use_validation else None,
                )

                pred_train = trained.predict(X_train)
                pred_val = trained.predict(X_val) if use_validation else None
                pred_test = trained.predict(X_test)

                make_predictions_file(y_train, pred_train, run_dir, "train", target_mask_train)
                if use_validation:
                    make_predictions_file(y_val, pred_val, run_dir, "val", target_mask_val)
                make_predictions_file(y_test, pred_test, run_dir, "test", target_mask_test)
                model_file = save_xgb_model(trained, run_dir)

                result_key = f"{emb_name}__{xgb_model_name}"
                result_entry = make_result_entry(emb_name, xgb_model_name, model_info, run_dir, model_file)
                update_card_locked(Path(card_path), task_name, result_key, result_entry)
                log(f"  Saved → {run_dir.name}")

    log(f"\nDone. Attention XGBoost results written to {card_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("features_dir", help="Directory containing exported attention-pooled features")
    parser.add_argument("output_dir", help="Directory to write XGBoost predictions and models")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding.")
    parser.add_argument(
        "--source_model_prefix",
        default=None,
        help="Use only exported attention features whose source model starts with this prefix.",
    )
    args = parser.parse_args()
    main(
        args.card_path,
        args.features_dir,
        args.output_dir,
        emb_name_filter=args.emb_name,
        source_model_prefix=args.source_model_prefix,
    )
