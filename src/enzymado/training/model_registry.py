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

import collections.abc
import inspect
import typing

import torch
import torch.utils.data

import enzymado.data_loading as data_loading
import enzymado.types

from .attention_wrappers import (
    AttentionBinaryClassifierTorch,
    AttentionECMultiHeadClassifierTorch,
    AttentionRegressorTorch,
    AttentionResidueBinaryClassifierTorch,
)
from .torch_standard_wrappers import (
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
    XGBRegressor,
    XGBResidueBinaryClassifier,
)


def _make_model_entry(
    wrap: collections.abc.Callable[..., object],
    params: dict[str, object],
    *,
    residue_aware: bool | None = None,
    loader_batch_size: int | None = None,
    concat_residue_batch: bool = False,
    flat_ec_labels: bool = False,
) -> enzymado.types.ModelInfo:
    """Build one registry entry while keeping the stored shape consistent."""
    entry: enzymado.types.ModelInfo = {
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


# Linear / SGD family


# The scalar linear head is the MLP wrapper without a hidden layer, trained in
# minibatches like the MLP. The earlier full-batch head took one optimizer step
# per epoch and stopped at 200 steps, before it had converged. Weight decay
# matches the MLP: across 0 to 1 the mean validation score moved by under 0.002.
_LINEAR_WEIGHT_DECAY = 0.1
_SCALAR_LINEAR_PARAMS: dict[str, object] = {
    "hidden_dims": (),
    "lr": 1e-3,
    "epochs": 200,
    "optimizer_name": "adamw",
    "weight_decay": _LINEAR_WEIGHT_DECAY,
    "dropout": 0.0,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,
}


def _standard_linear_classifier_entry() -> enzymado.types.ModelInfo:
    return _make_model_entry(
        MLPClassifierTorch,
        {**_SCALAR_LINEAR_PARAMS, "batch_size": 256},
        residue_aware=True,
    )


def _standard_linear_regressor_entry() -> enzymado.types.ModelInfo:
    return _make_model_entry(
        MLPRegressorTorch,
        {**_SCALAR_LINEAR_PARAMS, "batch_size": 256},
        residue_aware=True,
    )


def _ec_linear_classifier_entry() -> enzymado.types.ModelInfo:
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


def _residue_linear_classifier_entry() -> enzymado.types.ModelInfo:
    return _make_model_entry(
        ResidueBinaryClassifierTorch,
        {
            "hidden_dims": (),
            "lr": 0.001,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        residue_aware=True,
        loader_batch_size=128,
    )


# MLP family


def _standard_mlp_classifier_entry() -> enzymado.types.ModelInfo:
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


def _standard_mlp_regressor_entry() -> enzymado.types.ModelInfo:
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


def _ec_mlp_classifier_entry() -> enzymado.types.ModelInfo:
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


def _residue_mlp_classifier_entry() -> enzymado.types.ModelInfo:
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


# XGBoost family


def _standard_xgb_classifier_entry() -> enzymado.types.ModelInfo:
    return _make_model_entry(
        XGBClassifier,
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda",
            # Select on the metric the leaderboard reports, matching the torch
            # baselines, instead of spending the whole boosting budget blind.
            "eval_metric": "aucpr",
            "early_stopping_rounds": 20,
        },
    )


def _standard_xgb_regressor_entry() -> enzymado.types.ModelInfo:
    return _make_model_entry(
        XGBRegressor,
        {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "device": "cuda",
            # Regression boosting selects on RMSE under this protocol.
            "eval_metric": "rmse",
            "early_stopping_rounds": 20,
        },
    )


def _ec_xgb_classifier_entry() -> enzymado.types.ModelInfo:
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


def _residue_xgb_classifier_entry() -> enzymado.types.ModelInfo:
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
            # Spill the binned matrix to on-disk cache whenever the embedding
            # store is lazy (too large to hold in RAM), so the biggest residue
            # tasks (e.g. binding_site at dim=2560) no longer OOM building the
            # QuantileDMatrix. Small tasks keep the fast in-RAM path.
            "external_memory": "auto",
        },
        residue_aware=True,
        loader_batch_size=80,
        concat_residue_batch=True,
    )


# Registry assembly

MODEL_MAP: enzymado.types.ModelMap = {
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


STRUCTURED_MODEL_MAP: enzymado.types.ModelMap = {
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
    STRUCTURED_MODEL_MAP["multilabel_ec"]["xgb_classification"] = (
        _ec_xgb_classifier_entry()
    )
    STRUCTURED_MODEL_MAP["residue_binary_vector"]["xgb_classification"] = (
        _residue_xgb_classifier_entry()
    )


# Training helpers


def get_model_map(
    task_type: str,
    label_kind: str,
) -> enzymado.types.ModelRegistry:
    """Return the model registry appropriate for the task's label structure."""
    if label_kind in STRUCTURED_MODEL_MAP:
        return STRUCTURED_MODEL_MAP[label_kind]
    return MODEL_MAP[task_type]


def train_model(
    model_info: enzymado.types.ModelInfo,
    x_train: enzymado.types.Array,
    y_train: enzymado.types.ArrayPayload,
    x_val: enzymado.types.Array | None = None,
    y_val: enzymado.types.OptionalArrayPayload = None,
    **extra_fit_kwargs: object,
) -> object:
    """Instantiate and fit a model wrapper using only supported keyword args."""
    model = model_info["wrap"](**model_info["params"])
    fit = typing.cast(
        collections.abc.Callable[..., object],
        getattr(model, "fit"),
    )
    fit_signature = inspect.signature(fit)
    fit_kwargs: dict[str, object] = {}
    if x_val is not None and y_val is not None and "x_val" in fit_signature.parameters:
        fit_kwargs["x_val"] = x_val
        fit_kwargs["y_val"] = y_val
    # Gradient-boosting estimators take held-out data as sklearn's eval_set
    # rather than x_val/y_val, so they need their own hand-off; without it
    # early_stopping_rounds is silently inert.
    if (
        x_val is not None
        and y_val is not None
        and "eval_set" in fit_signature.parameters
        and "x_val" not in fit_signature.parameters
    ):
        fit_kwargs["eval_set"] = [(x_val, y_val)]
    for key, value in extra_fit_kwargs.items():
        if value is not None and key in fit_signature.parameters:
            fit_kwargs[key] = value
    fit(x_train, y_train, **fit_kwargs)
    return model


def train_model_loader(
    model_info: enzymado.types.ModelInfo,
    train_loader: torch.utils.data.DataLoader[data_loading.Batch],
    batch_kind: str,
    val_loader: torch.utils.data.DataLoader[data_loading.Batch] | None = None,
    **extra_fit_kwargs: object,
) -> object:
    """Instantiate and fit a model wrapper that trains from DataLoaders."""
    model = model_info["wrap"](**model_info["params"])
    fit_loader = typing.cast(
        collections.abc.Callable[..., object],
        getattr(model, "fit_loader"),
    )
    fit_loader_signature = inspect.signature(fit_loader)
    fit_kwargs: dict[str, object] = {}
    if val_loader is not None and "val_dataloader" in fit_loader_signature.parameters:
        fit_kwargs["val_dataloader"] = val_loader
    for key, value in extra_fit_kwargs.items():
        if value is not None and key in fit_loader_signature.parameters:
            fit_kwargs[key] = value
    fit_loader(train_loader, batch_kind=batch_kind, **fit_kwargs)
    return model


# Attention-model registries grouped by model family
#
# Pooled (global) attention entries train exactly like their mean-pool
# counterparts apart from the attention block. The attention route loads 16
# proteins per loader batch (long residue tensors), so gradients are accumulated
# to reach the same effective batch: 16 x 16 = 256 for every head.
_ATTENTION_LOADER_BATCH = 16
_MEAN_POOL_BATCH = 256
_ACCUMULATE_TO_MEAN_POOL_BATCH = _MEAN_POOL_BATCH // _ATTENTION_LOADER_BATCH
_ATTENTION_BLOCK_LR = 1e-3


def _make_attention_entry(
    wrap: collections.abc.Callable[..., object],
    params: dict[str, object],
) -> enzymado.types.ModelInfo:
    """Build one attention registry entry."""
    return {
        "wrap": wrap,
        "params": dict(params),
        "residue_aware": True,
    }


def _attention_linear_classifier_entry() -> enzymado.types.ModelInfo:
    return _make_attention_entry(
        AttentionBinaryClassifierTorch,
        {
            **_SCALAR_LINEAR_PARAMS,
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_linear_regressor_entry() -> enzymado.types.ModelInfo:
    return _make_attention_entry(
        AttentionRegressorTorch,
        {
            **_SCALAR_LINEAR_PARAMS,
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_ec_linear_entry() -> enzymado.types.ModelInfo:
    return _make_attention_entry(
        AttentionECMultiHeadClassifierTorch,
        {
            "hidden_dims": (),
            # The head keeps the mean-pool EC linear rate. The attention block
            # diverges at 0.01 (validation loss in the thousands from epoch 1),
            # so it trains at 1e-3 like every other attention block.
            "lr": 0.01,
            "attention_block_lr": _ATTENTION_BLOCK_LR,
            "epochs": 200,
            "optimizer_name": "adamw",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_residue_linear_entry() -> enzymado.types.ModelInfo:
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


def _attention_mlp_classifier_entry() -> enzymado.types.ModelInfo:
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
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_mlp_regressor_entry() -> enzymado.types.ModelInfo:
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
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_ec_mlp_entry() -> enzymado.types.ModelInfo:
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
            "gradient_accumulation_steps": _ACCUMULATE_TO_MEAN_POOL_BATCH,
        },
    )


def _attention_residue_mlp_entry() -> enzymado.types.ModelInfo:
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


ATTENTION_MODEL_MAP: enzymado.types.ModelMap = {
    "classification": {
        "attention_mlp_classification": _attention_mlp_classifier_entry(),
        "attention_linear_classification": _attention_linear_classifier_entry(),
    },
    "regression": {
        "attention_mlp_regressor": _attention_mlp_regressor_entry(),
        "attention_linear_regressor": _attention_linear_regressor_entry(),
    },
}


ATTENTION_STRUCTURED_MODEL_MAP: enzymado.types.ModelMap = {
    "multilabel_ec": {
        "attention_mlp_classification": _attention_ec_mlp_entry(),
        "attention_linear_classification": _attention_ec_linear_entry(),
    },
    "residue_binary_vector": {
        "attention_mlp_classification": _attention_residue_mlp_entry(),
        "attention_linear_classification": _attention_residue_linear_entry(),
    },
}


def get_attention_model_map(
    task_type: str, label_kind: str
) -> enzymado.types.ModelRegistry:
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
