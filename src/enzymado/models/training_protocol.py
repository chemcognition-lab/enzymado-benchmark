"""Selection policy, numerical checks and checkpoint compatibility."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch

PROTOCOL_VERSION = 3


def selection_protocol(wrapper_class, params: dict | None = None) -> dict:
    """Describe checkpoint selection independently of fitted state."""
    if wrapper_class.__name__.startswith("XGB"):
        hp = params or {}
        metric = hp.get("eval_metric", "library_default")
        return {
            "version": PROTOCOL_VERSION,
            "metric": metric,
            "mode": "max" if metric in {"auc", "aucpr"} else "min",
            "early_stopping_rounds": hp.get("early_stopping_rounds"),
            "initialization": "not_applicable",
        }
    metric = getattr(wrapper_class, "validation_metric", None)
    return {
        "version": PROTOCOL_VERSION,
        "metric": metric or "validation_loss",
        "objective": f"negative_{metric}" if metric else "validation_loss",
        "mode": "min",
        "initialization": "reference_only",
        "eligible_epochs": "trained_with_defined_metric",
        "undefined_metric": "ineligible_epoch",
        "non_finite": "fail_run",
    }


def resume_contract(wrapper, checkpoint_path: Path | None, dataloader=None) -> dict:
    """Capture constructor configuration and pipeline provenance before loading."""
    names = inspect.signature(type(wrapper).__init__).parameters
    params = {
        name: getattr(wrapper, name)
        for name in names
        if name not in {"self", "device"} and hasattr(wrapper, name)
    }
    manifest = checkpoint_path.parent / "run_manifest.json" if checkpoint_path else None
    return {
        "selection": selection_protocol(type(wrapper), params),
        "params": json.loads(json.dumps(params)),
        "device_type": str(getattr(wrapper, "device", "cpu")).split(":")[0],
        "loader": (
            {
                "batch_size": dataloader.batch_size,
                "drop_last": dataloader.drop_last,
                "num_workers": dataloader.num_workers,
                "sampler": type(dataloader.sampler).__name__,
                "dataset_size": len(dataloader.dataset),
            }
            if dataloader is not None
            else None
        ),
        "run_manifest_sha256": (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest is not None and manifest.exists()
            else None
        ),
    }


def backward_checked(loss: torch.Tensor) -> None:
    """Reject invalid losses before computing gradients."""
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("Non-finite training loss; optimizer not stepped.")
    loss.backward()


def check_optimizer_gradients(optimizer, args, kwargs) -> None:
    """Reject invalid gradients before an optimizer can corrupt its state."""
    gradients = [
        param.grad
        for group in optimizer.param_groups
        for param in group["params"]
        if param.grad is not None
    ]
    check_finite_tensors(gradients, "gradient")


def check_optimizer_parameters(optimizer, args, kwargs) -> None:
    """Reject invalid weights, including an overflow on the final update."""
    parameters = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    check_finite_tensors(parameters, "parameter")


def check_finite_tensors(tensors, name: str) -> None:
    """Check tensors with a single host synchronization per device."""
    if not tensors:
        return
    finite = torch.stack([torch.isfinite(t).all() for t in tensors]).all()
    if not finite.item():
        raise FloatingPointError(f"Non-finite {name}; training aborted.")
