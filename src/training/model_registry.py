"""Model registries keyed by benchmark task type and label kind.

This file is organized by model family first:

1. Linear / SGD baselines
2. MLP baselines
3. XGBoost baselines
4. Final registry assembly by task type / label kind
5. Attention-model registries grouped by model family

That keeps each family's parameters and wrappers together, while preserving the
same public registry objects used by the training pipeline.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch.utils.data import DataLoader

from .attention_wrappers import (
    AttentionBinaryClassifierTorch,
    AttentionECMultiHeadClassifierTorch,
    AttentionRegressorTorch,
    AttentionResidueBinaryClassifierTorch,
)
from .torch_standard_wrappers import (
    LinearClassifierTorch,
    LinearRegressorTorch,
    MLPClassifierTorch,
    MLPRegressorTorch,
)
from .torch_structured_wrappers import (
    ECMultiHeadClassifierTorch,
    ResidueBinaryClassifierTorch,
)
from .xgb_wrappers import (
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
            "lr": 0.001,
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
            "lr": 0.001,
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
            "optimizer_name": "adamw",
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
            "optimizer_name": "adamw",
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
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "batch_size": 256,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _standard_mlp_regressor_entry() -> dict:
    return _make_model_entry(
        MLPRegressorTorch,
        {
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "batch_size": 256,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
    )


def _ec_mlp_classifier_entry() -> dict:
    return _make_model_entry(
        ECMultiHeadClassifierTorch,
        {
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "batch_size": 256,
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
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
        loader_batch_size=256,
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
            # Pin nthread to the actual cpus-per-task allocation rather than -1
            # (auto-detect all host cores), since SLURM cgroups can confine the
            # process to fewer cores than the host reports.
            "n_jobs": 4,
            # CPU is plenty fast for this streaming QuantileDMatrix path (the
            # per-batch sums dominate; binning to <=256 bins is cheap), and
            # keeps the residue path independent of GPU vector-leaf support
            # across xgboost versions.
            "device": "cpu",
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


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_model_map(task_type: str, label_kind: str) -> dict[str, dict]:
    """Return the model registry appropriate for the task's label structure."""
    if label_kind in STRUCTURED_MODEL_MAP:
        return STRUCTURED_MODEL_MAP[label_kind]
    return MODEL_MAP[task_type]


def train_model(model_info: dict, X_train, y_train, X_val=None, y_val=None, **extra_fit_kwargs):
    """Instantiate and fit a model wrapper using only supported keyword args."""
    model = model_info["wrap"](**model_info["params"])
    fit_signature = inspect.signature(model.fit)
    fit_kwargs = {}
    if X_val is not None and y_val is not None and "X_val" in fit_signature.parameters:
        fit_kwargs["X_val"] = X_val
        fit_kwargs["y_val"] = y_val
    for key, value in extra_fit_kwargs.items():
        if value is not None and key in fit_signature.parameters:
            fit_kwargs[key] = value
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_model_loader(
    model_info: dict,
    train_loader: DataLoader,
    batch_kind: str,
    val_loader: DataLoader | None = None,
    **extra_fit_kwargs,
):
    """Instantiate and fit a model wrapper that trains from DataLoaders."""
    model = model_info["wrap"](**model_info["params"])
    fit_loader_signature = inspect.signature(model.fit_loader)
    fit_kwargs = {}
    if val_loader is not None and "val_dataloader" in fit_loader_signature.parameters:
        fit_kwargs["val_dataloader"] = val_loader
    for key, value in extra_fit_kwargs.items():
        if value is not None and key in fit_loader_signature.parameters:
            fit_kwargs[key] = value
    model.fit_loader(train_loader, batch_kind=batch_kind, **fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Attention-model registries grouped by model family
# ---------------------------------------------------------------------------

def _make_attention_entry(wrap, params: dict) -> dict[str, Any]:
    """Build one attention registry entry."""
    return {
        "wrap": wrap,
        "params": dict(params),
        "residue_aware": True,
    }


def _attention_linear_classifier_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionBinaryClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.001,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_linear_regressor_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionRegressorTorch,
        {
            "hidden_dims": (),
            "lr": 0.001,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_ec_linear_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionECMultiHeadClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.001,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_residue_linear_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionResidueBinaryClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.001,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_mlp_classifier_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionBinaryClassifierTorch,
        {
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_mlp_regressor_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionRegressorTorch,
        {
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_ec_mlp_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionECMultiHeadClassifierTorch,
        {
            "hidden_dims": (2048,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "batch_size": 256,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


def _attention_residue_mlp_entry() -> dict[str, Any]:
    return _make_attention_entry(
        AttentionResidueBinaryClassifierTorch,
        {
            "hidden_dims": (256,),
            "lr": 1e-3,
            "epochs": 100,
            "optimizer_name": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.1,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
    )


ATTENTION_MODEL_MAP = {
    "classification": {
        "attention_mlp_classification": _attention_mlp_classifier_entry(),
        "attention_linear_classification": _attention_linear_classifier_entry(),
    },
    "regression": {
        "attention_mlp_regressor": _attention_mlp_regressor_entry(),
        "attention_linear_regressor": _attention_linear_regressor_entry(),
    },
}


ATTENTION_STRUCTURED_MODEL_MAP = {
    "multilabel_ec": {
        "attention_mlp_classification": _attention_ec_mlp_entry(),
        "attention_linear_classification": _attention_ec_linear_entry(),
    },
    "residue_binary_vector": {
        "attention_mlp_classification": _attention_residue_mlp_entry(),
        "attention_linear_classification": _attention_residue_linear_entry(),
    },
}


def get_attention_model_map(task_type: str, label_kind: str) -> dict[str, dict[str, Any]]:
    """Return the attention registry appropriate for the task label structure."""
    if label_kind in ATTENTION_STRUCTURED_MODEL_MAP:
        return ATTENTION_STRUCTURED_MODEL_MAP[label_kind]
    return ATTENTION_MODEL_MAP[task_type]


__all__ = [
    "MODEL_MAP",
    "STRUCTURED_MODEL_MAP",
    "get_model_map",
    "train_model",
    "train_model_loader",
    "ATTENTION_MODEL_MAP",
    "ATTENTION_STRUCTURED_MODEL_MAP",
    "get_attention_model_map",
]
