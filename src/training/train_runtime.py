"""Training-time model selection and fitting helpers."""

from __future__ import annotations

import inspect

from torch.utils.data import DataLoader

from models.registry import MODEL_MAP, STRUCTURED_MODEL_MAP


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


__all__ = [
    "get_model_map",
    "train_model",
    "train_model_loader",
]
