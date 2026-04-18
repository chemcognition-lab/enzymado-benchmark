"""Model registries keyed by benchmark task type and label kind."""

from __future__ import annotations

import torch

from models.torch_wrappers import (
    ECMultiHeadClassifierTorch,
    LinearClassifierTorch,
    LinearRegressorTorch,
    MLPClassifierTorch,
    MLPRegressorTorch,
    ResidueBinaryClassifierTorch,
)
from models.xgb_wrappers import (
    XGBClassifier,
    XGBFlatECClassifier,
    XGBRegressor,
    XGBResidueBinaryClassifier,
)


def _linear_classifier_params():
    return {
        "wrap": LinearClassifierTorch,
        "params": {
            "lr": 0.01,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        "residue_aware": True,
    }


def _linear_regressor_params():
    return {
        "wrap": LinearRegressorTorch,
        "params": {
            "lr": 0.01,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        "residue_aware": True,
    }


MODEL_MAP = {
    "classification": {
        "mlp_classification": {
            "wrap": MLPClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "linear_classification": _linear_classifier_params(),
        "sgd_classification": _linear_classifier_params(),
    },
    "regression": {
        "mlp_regressor": {
            "wrap": MLPRegressorTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "linear_regressor": _linear_regressor_params(),
        "sgd_regressor": _linear_regressor_params(),
    },
}

if XGBClassifier is not None:
    MODEL_MAP["classification"]["xgb_classification"] = {
        "wrap": XGBClassifier,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"},
    }

if XGBRegressor is not None:
    MODEL_MAP["regression"]["xgb_regression"] = {
        "wrap": XGBRegressor,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "device": "cuda"},
    }


STRUCTURED_MODEL_MAP = {
    "multilabel_ec": {
        "mlp_classification": {
            "wrap": ECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": False,
        },
        "linear_classification": {
            "wrap": ECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": False,
        },
        "sgd_classification": {
            "wrap": ECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": False,
        },
    },
    "residue_binary_vector": {
        "mlp_classification": {
            "wrap": ResidueBinaryClassifierTorch,
            "params": {
                "hidden_dims": (256,),
                "lr": 1e-3,
                "epochs": 40,
                "optimizer_name": "adam",
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 1e-3,
            },
            "residue_aware": True,
            "loader_batch_size": 96,
        },
        "linear_classification": {
            "wrap": ResidueBinaryClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 30,
                "optimizer_name": "sgd",
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 1e-3,
            },
            "residue_aware": True,
            "loader_batch_size": 128,
        },
        "sgd_classification": {
            "wrap": ResidueBinaryClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 30,
                "optimizer_name": "sgd",
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 1e-3,
            },
            "residue_aware": True,
            "loader_batch_size": 128,
        },
    },
}

if XGBClassifier is not None:
    STRUCTURED_MODEL_MAP["residue_binary_vector"]["xgb_classification"] = {
        "wrap": XGBResidueBinaryClassifier,
        "params": {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 10,
        },
        "residue_aware": True,
        "concat_residue_batch": True,
        "loader_batch_size": 96,
    }
    STRUCTURED_MODEL_MAP["multilabel_ec"]["xgb_classification"] = {
        "wrap": XGBFlatECClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.2,
            "tree_method": "hist",
            "max_bin": 128,
            "n_jobs": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": 0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        },
        "residue_aware": False,
        "flat_ec_labels": True,
    }
