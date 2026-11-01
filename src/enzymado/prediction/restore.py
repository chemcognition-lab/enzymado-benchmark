"""Restore fitted wrappers from published runs and attach probability artifacts."""

from __future__ import annotations

import lzma
from pathlib import Path
import sys

import joblib
import numpy as np
import safetensors.numpy
import safetensors.torch
import torch

from enzymado import data_loading, models, training
from enzymado.artifacts import serialization

_LEGACY_ROOTS = {
    "data_loading": data_loading,
    "models": models,
    "training": training,
}


def install_legacy_module_aliases() -> None:
    """Map pre-reorganization module names onto the canonical enzymado package.

    Wrappers pickled before ``src/`` was consolidated reference ``training.*``,
    ``models.*`` and ``data_loading.*``. Real, versioned checkpoints in this
    repository require the aliases, so unpickling them needs the old names to
    resolve. This registers aliases only; it adds no importable packages.
    """
    for legacy_name, module in _LEGACY_ROOTS.items():
        sys.modules.setdefault(legacy_name, module)
        for attribute in dir(module):
            candidate = getattr(module, attribute)
            name = getattr(candidate, "__name__", "")
            if name.startswith("enzymado."):
                sys.modules.setdefault(f"{legacy_name}.{attribute}", candidate)


def _model_weights_path(run_dir: Path) -> Path | None:
    """Return the safetensors file holding final module weights, if present."""
    for path in sorted(run_dir.glob("*.safetensors")):
        if not path.name.endswith("_predictions.safetensors"):
            return path
    return None


def restore_fitted_wrapper(run_dir: Path, *, device: torch.device | None = None):
    """Load the fitted wrapper a published run left behind.

    XGBoost runs pickle the whole estimator. Torch and attention runs pickle the
    wrapper (carrying fitted state such as ``seq_mean_``) and separately persist
    the final module weights as safetensors, which are written after training
    restores the best epoch. The safetensors file therefore wins over whatever
    epoch the periodic checkpoint happened to capture.
    """
    install_legacy_module_aliases()

    joblib_path = run_dir / "model.joblib"
    if joblib_path.exists():
        return joblib.load(joblib_path)

    checkpoint_path = run_dir / "training_checkpoint.bin"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{run_dir} has no restorable wrapper checkpoint.")
    wrapper = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    target_device = device or getattr(wrapper, "device", None) or torch.device("cpu")
    target_device = torch.device(target_device)
    wrapper.device = target_device
    # Move every fitted tensor attribute, not an enumerated subset: wrappers
    # carry task-dependent scalers (sequence, ligand extras, feature scalers)
    # and a missing one only surfaces at predict time as a device mismatch.
    for attribute, value in list(vars(wrapper).items()):
        if torch.is_tensor(value):
            setattr(wrapper, attribute, value.to(target_device))

    if wrapper.model is None:
        raise ValueError(f"{run_dir} checkpoint carries no built model.")
    wrapper.model = wrapper.model.to(target_device)
    weights_path = _model_weights_path(run_dir)
    if weights_path is not None:
        safetensors.torch.load_model(wrapper.model, weights_path)
        wrapper.model = wrapper.model.to(target_device)
    return wrapper


def predict_probabilities(wrapper, *, dataloader=None, features=None, batch_kind=None):
    """Return probabilities from a restored wrapper for loader or array routes."""
    if dataloader is not None:
        return wrapper.predict_proba_loader(dataloader, batch_kind=batch_kind)
    scores = wrapper.predict_proba(features)
    if isinstance(scores, dict):
        return scores
    scores = np.asarray(scores)
    # scikit-learn style estimators return one column per class.
    return scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores


def attach_probabilities(artifact_path: Path, proba) -> Path:
    """Add quantized probabilities to an artifact, preserving every stored tensor.

    Published hard labels are copied through untouched rather than regenerated,
    so adding probabilities cannot perturb the predictions a benchmark already
    reports.
    """
    with lzma.open(artifact_path, "rb") as handle:
        tensors = safetensors.numpy.load(handle.read())

    quantized = serialization.quantize_probabilities(proba)
    if isinstance(quantized, dict):
        additions = {f"proba_{key}": value for key, value in quantized.items()}
    else:
        additions = {"proba": np.asarray(quantized)}

    for key, value in additions.items():
        reference = tensors.get(key.replace("proba", "predictions", 1))
        if reference is not None and value.shape != reference.shape:
            raise ValueError(
                f"{artifact_path.name}: probability block {key} has shape "
                f"{value.shape}, expected {reference.shape}."
            )
    payload = {**tensors, **additions}
    blob = safetensors.numpy.save(payload)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with lzma.open(temporary, "wb") as handle:
        handle.write(blob)
    temporary.replace(artifact_path)
    return artifact_path


__all__ = [
    "attach_probabilities",
    "install_legacy_module_aliases",
    "predict_probabilities",
    "restore_fitted_wrapper",
]
