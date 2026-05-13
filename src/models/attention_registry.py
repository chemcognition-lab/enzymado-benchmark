"""Attention-model registries grouped by model family."""

from __future__ import annotations

from typing import Any

from training.attention_wrappers import (
    AttentionBinaryClassifierTorch,
    AttentionECMultiHeadClassifierTorch,
    AttentionRegressorTorch,
    AttentionResidueBinaryClassifierTorch,
)


def _make_attention_entry(wrap, params: dict) -> dict[str, Any]:
    """Build one attention registry entry."""
    return {
        "wrap": wrap,
        "params": dict(params),
        "residue_aware": True,
    }


# ---------------------------------------------------------------------------
# Linear / SGD family
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MLP family
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------

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
    "ATTENTION_MODEL_MAP",
    "ATTENTION_STRUCTURED_MODEL_MAP",
    "get_attention_model_map",
]
