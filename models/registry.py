"""Model registries keyed by benchmark task type and label kind.

This file is organized by model family first:

1. Linear / SGD baselines
2. MLP baselines
3. XGBoost baselines
4. Final registry assembly by task type / label kind

That keeps each family's parameters and wrappers together, while preserving the
same public registry objects used by the training pipeline.
"""

from __future__ import annotations

import os
import json
import torch

from models.torch_standard_wrappers import (
    LinearClassifierTorch,
    LinearRegressorTorch,
    MLPClassifierTorch,
    MLPRegressorTorch,
)
from models.torch_structured_wrappers import (
    ECMultiHeadClassifierTorch,
    ResidueBinaryClassifierTorch,
)
from models.xgb_wrappers import (
    XGBClassifier,
    XGBFlatECBlockClassifier,
    XGBFlatECClassifier,
    XGBRegressor,
    XGBResidueBinaryClassifier,
)


def _make_model_entry(
    wrap,
    params: dict,
    *,
    residue_aware: bool | None = None,
    loader_batch_size: int | None = None,
    concat_residue_batch: bool = False,
    flat_ec_labels: bool = False,
) -> dict:
    """Build one registry entry while keeping the stored shape consistent."""
    entry = {
        "wrap": wrap,
        "params": dict(params),
    }
    if residue_aware is not None:
        entry["residue_aware"] = residue_aware
    if loader_batch_size is not None:
        entry["loader_batch_size"] = loader_batch_size
    if concat_residue_batch:
        entry["concat_residue_batch"] = True
    if flat_ec_labels:
        entry["flat_ec_labels"] = True
    return entry


# ---------------------------------------------------------------------------
# Linear / SGD family
# ---------------------------------------------------------------------------

def _standard_linear_classifier_entry() -> dict:
    return _make_model_entry(
        LinearClassifierTorch,
        {
            "lr": 0.01,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _standard_linear_regressor_entry() -> dict:
    return _make_model_entry(
        LinearRegressorTorch,
        {
            "lr": 0.01,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _ec_linear_classifier_entry() -> dict:
    return _make_model_entry(
        ECMultiHeadClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.01,
            "epochs": 200,
            "optimizer_name": "sgd",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=False,
    )


def _residue_linear_classifier_entry() -> dict:
    return _make_model_entry(
        ResidueBinaryClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.01,
            "epochs": 30,
            "optimizer_name": "sgd",
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 1e-3,
        },
        residue_aware=True,
        loader_batch_size=128,
    )


# ---------------------------------------------------------------------------
# MLP family
# ---------------------------------------------------------------------------

def _standard_mlp_classifier_entry() -> dict:
    return _make_model_entry(
        MLPClassifierTorch,
        {
            "hidden_dims": (512, 256),
            "lr": 1e-3,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _standard_mlp_regressor_entry() -> dict:
    return _make_model_entry(
        MLPRegressorTorch,
        {
            "hidden_dims": (512, 256),
            "lr": 1e-3,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _ec_mlp_classifier_entry() -> dict:
    return _make_model_entry(
        ECMultiHeadClassifierTorch,
        {
            "hidden_dims": (512, 256),
            "lr": 1e-3,
            "epochs": 200,
            "optimizer_name": "adam",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=False,
    )


def _residue_mlp_classifier_entry() -> dict:
    return _make_model_entry(
        ResidueBinaryClassifierTorch,
        {
            "hidden_dims": (256,),
            "lr": 1e-3,
            "epochs": 40,
            "optimizer_name": "adam",
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 1e-3,
        },
        residue_aware=True,
        loader_batch_size=96,
    )


# ---------------------------------------------------------------------------
# XGBoost family
# ---------------------------------------------------------------------------

def _standard_xgb_classifier_entry() -> dict:
    return _make_model_entry(
        XGBClassifier,
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda",
        },
    )


def _standard_xgb_regressor_entry() -> dict:
    return _make_model_entry(
        XGBRegressor,
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda",
        },
    )


def _ec_xgb_classifier_entry() -> dict:
    return _make_model_entry(
        XGBFlatECBlockClassifier,
        {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.1,
            "tree_method": "hist",
            "max_bin": 64,
            "block_size": 256,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        },
        residue_aware=False,
        flat_ec_labels=True,
    )


def _residue_xgb_classifier_entry() -> dict:
    return _make_model_entry(
        XGBResidueBinaryClassifier,
        {
            "n_estimators": 40,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.5,
            "colsample_bytree": 0.25,
            "max_bin": 64,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 10,
        },
        residue_aware=True,
        loader_batch_size=80,
        concat_residue_batch=True,
    )


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "classification": {
        "mlp_classification": _standard_mlp_classifier_entry(),
        "linear_classification": _standard_linear_classifier_entry(),
    },
    "regression": {
        "mlp_regressor": _standard_mlp_regressor_entry(),
        "linear_regressor": _standard_linear_regressor_entry(),
    },
}

if XGBClassifier is not None:
    MODEL_MAP["classification"]["xgb_classification"] = _standard_xgb_classifier_entry()

if XGBRegressor is not None:
    MODEL_MAP["regression"]["xgb_regression"] = _standard_xgb_regressor_entry()


STRUCTURED_MODEL_MAP = {
    "multilabel_ec": {
        "mlp_classification": _ec_mlp_classifier_entry(),
        "linear_classification": _ec_linear_classifier_entry(),
    },
    "residue_binary_vector": {
        "mlp_classification": _residue_mlp_classifier_entry(),
        "linear_classification": _residue_linear_classifier_entry(),
    },
}

if XGBClassifier is not None:
    STRUCTURED_MODEL_MAP["multilabel_ec"]["xgb_classification"] = _ec_xgb_classifier_entry()
    STRUCTURED_MODEL_MAP["residue_binary_vector"]["xgb_classification"] = _residue_xgb_classifier_entry()


hpo_file = os.environ.get("HPO_CONFIG_PATH")
if hpo_file and os.path.exists(hpo_file):
    print(f"\n[REGISTRY] Successfully found HPO configs at: {hpo_file}")
    with open(hpo_file, "r") as f:
        hpo_list = json.load(f)
        
    for i, params in enumerate(hpo_list):
        idx = params.get("config_index", i)
        run_id = f"mlp_hpo_{idx:03d}" 
        params["is_hpo"] = True
        
        # Inject directly into regression map
        MODEL_MAP["regression"][run_id] = {
            "wrap": MLPRegressorTorch,
            "params": params,
            "residue_aware": True 
        }
else:
    if hpo_file:
        print(f"\n[REGISTRY ERROR] HPO_CONFIG_PATH was set but file not found: {hpo_file}")