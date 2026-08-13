"""Shared utilities for model wrappers."""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation import metrics_registry


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_random_generators(seed: int) -> None:
    """Seed the RNGs used by benchmark model initialization and training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def masked_mean_pool(out: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool (batch, seq_len, dim) -> (batch, dim) using a boolean mask."""
    if mask is None:
        return out.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def make_optimizer(parameters, optimizer_name: str, lr: float, weight_decay: float = 0.0):
    """Instantiate the optimizer requested by a lightweight wrapper config."""
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_name '{optimizer_name}'.")


def fit_feature_scaler(X, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple per-feature standardizer for 2D or padded 3D inputs."""
    X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 2:
        observed = X_np
    elif X_np.ndim == 3:
        if mask is None:
            observed = X_np.reshape(-1, X_np.shape[-1])
        else:
            observed = X_np[np.asarray(mask, dtype=bool)]
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {X_np.shape}")
    mean = observed.mean(axis=0).astype(np.float32, copy=False)
    scale = observed.std(axis=0).astype(np.float32, copy=False)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def apply_feature_scaler(X, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a fitted per-feature standardizer."""
    return (np.asarray(X, dtype=np.float32) - mean) / scale


def mean_losses(losses: list[float]) -> float:
    return sum(losses) / max(len(losses), 1)


def pad_1d_array_list(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    """Pad variable-length 1D outputs into a rectangular NumPy array."""
    if not arrays:
        return np.zeros((0, 0), dtype=dtype)
    max_length = max(array.shape[0] for array in arrays)
    batch = np.zeros((len(arrays), max_length), dtype=dtype)
    for index, array in enumerate(arrays):
        batch[index, : array.shape[0]] = array.astype(dtype, copy=False)
    return batch


def batch_to_input(batch, batch_kind: str, device: torch.device):
    """Extract (X_t, mask_t, y_np) from a collated batch for residue-aware training."""
    if batch_kind == "protein_global":
        X_t = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    elif batch_kind == "mutation_delta_global":
        X_t = torch.from_numpy(np.asarray(batch.protein_delta_emb, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    elif batch_kind == "protein_ligand_global":
        raise ValueError(
            f"batch_kind='protein_ligand_global' is not supported by standard MLP/Linear wrappers "
            f"because they cannot combine protein and ligand features. "
            f"Use AttentionRegressorTorch or AttentionBinaryClassifierTorch instead."
        )
    else:
        raise ValueError(f"Unrecognised batch_kind='{batch_kind}'.")
    y_np = np.asarray(batch.target, dtype=np.float32)
    return X_t, mask_t, y_np


def prepare_inputs(
    X: np.ndarray, mask: np.ndarray | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return (X_tensor, mask_tensor, input_dim) on the given device."""
    X = np.asarray(X, dtype=np.float32)
    mask_t = None
    if X.ndim == 3:
        input_dim = X.shape[2]
        if mask is not None:
            mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
    elif X.ndim == 2:
        input_dim = X.shape[1]
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
    return torch.from_numpy(X).to(device), mask_t, input_dim


# Worst attainable value per checkpoint-selection metric (both higher-is-better).
_WORST_VALIDATION_SCORE = {"kendall_tau": -1.0, "auprc": 0.0}


def validation_score(
    y_true,
    pred,
    metric_name: str,
    target_mask=None,
) -> float:
    """Calculate the checkpoint-selection score from validation predictions."""
    match metric_name:
        case "auprc":
            task_type = "classification"
        case "kendall_tau":
            task_type = "regression"
        case _:
            raise ValueError(f"Unsupported validation metric: {metric_name}.")

    if isinstance(y_true, dict):
        scores = []
        for level_name, level_y in y_true.items():
            level_mask = (
                target_mask[level_name] if isinstance(target_mask, dict) else None
            )
            scores.append(
                metrics_registry.evaluate_metric(
                    level_y,
                    pred[level_name],
                    task_type,
                    metric_name,
                    level_mask,
                )
            )
        finite_scores = [value for value in scores if math.isfinite(value)]
        score = (
            float(np.mean(finite_scores))
            if finite_scores
            else float("nan")
        )
    else:
        score = metrics_registry.evaluate_metric(
            y_true,
            pred,
            task_type,
            metric_name,
            target_mask,
        )
    if not math.isfinite(score):
        # Both metrics are undefined on degenerate predictions (Kendall tau is
        # NaN when every prediction is identical, which a freshly initialised
        # attention model routinely produces at the epoch-0 baseline). Score it
        # as the worst possible value instead of aborting training: it stays a
        # valid checkpoint candidate that any real epoch beats.
        return _WORST_VALIDATION_SCORE[metric_name]
    return score


def _atomic_torch_save(payload, path: str | Path) -> None:
    """Write a torch-serialized payload atomically to avoid partial checkpoints."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path_obj)


def save_torch_wrapper_checkpoint(wrapper, path: str | Path) -> None:
    """Persist a full wrapper object so interrupted runs can resume later."""
    _atomic_torch_save(wrapper, path)


def maybe_resume_torch_wrapper(wrapper, path: str | Path | None) -> int:
    """Load a previously saved wrapper checkpoint into the current wrapper instance."""
    if path is None:
        return 0

    path_obj = Path(path)
    if not path_obj.exists():
        return 0

    loaded = torch.load(path_obj, map_location="cpu", weights_only=False)
    if not isinstance(loaded, wrapper.__class__):
        raise TypeError(
            f"Checkpoint at {path_obj} contains {type(loaded).__name__}, "
            f"expected {wrapper.__class__.__name__}."
        )

    current_device = getattr(wrapper, "device", DEVICE)
    wrapper.__dict__.update(loaded.__dict__)
    wrapper.device = torch.device(current_device)
    if getattr(wrapper, "model", None) is not None:
        wrapper.model = wrapper.model.to(wrapper.device)
    # Checkpoints are loaded to CPU; move any other restored tensor attributes
    # (e.g. normalization scalers) back to the active device so forward passes
    # don't mix cuda/cpu tensors after a resume.
    for attr_name, attr_value in wrapper.__dict__.items():
        if isinstance(attr_value, torch.Tensor):
            wrapper.__dict__[attr_name] = attr_value.to(wrapper.device)
    return int(getattr(wrapper, "current_epoch_", 0))


def make_torch_checkpoint_callback(wrapper, path: str | Path | None, checkpoint_every: int = 1):
    """Build an epoch-end callback that updates wrapper progress and saves checkpoints."""
    if path is None:
        return None

    checkpoint_every = max(1, int(checkpoint_every))

    def _callback(
        epoch: int,
        val_score,
        best_epoch,
        best_val_score,
        best_state,
    ) -> None:
        wrapper.current_epoch_ = int(epoch)
        wrapper.best_epoch_ = best_epoch
        wrapper.best_val_score_ = best_val_score
        wrapper.best_model_state_ = copy.deepcopy(best_state) if best_state is not None else None
        if not hasattr(wrapper, "training_log_") or wrapper.training_log_ is None:
            wrapper.training_log_ = []
        wrapper.training_log_.append(
            {
                "epoch": int(epoch),
                "validation_metric": wrapper.validation_metric_,
                "val_score": val_score,
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
            }
        )
        if epoch % checkpoint_every == 0:
            save_torch_wrapper_checkpoint(wrapper, path)

    return _callback


def run_training_with_early_stopping(
    model: nn.Module,
    epochs: int,
    train_epoch_fn,
    val_score_fn=None,
    metric_name: str = "validation_score",
    patience: int | None = None,
    min_delta: float = 0.0,
    validation_interval: int = 1,
    log_every: int = 10,
    start_epoch: int = 0,
    initial_best_state: dict | None = None,
    initial_best_epoch: int | None = None,
    initial_best_val_score: float | None = None,
    epoch_callback=None,
) -> tuple[int | None, float | None]:
    """Run a training loop and optionally restore the best validation checkpoint."""
    validation_interval = max(1, int(validation_interval))
    if val_score_fn is None or patience is None or patience <= 0:
        for ep in range(start_epoch, epochs):
            t0 = time.time()
            train_epoch_fn()
            if (ep + 1) % log_every == 0 or ep == 0:
                elapsed = time.time() - t0
                print(
                    f"        epoch {ep + 1}/{epochs}  ({elapsed:.1f}s/epoch)",
                    flush=True,
                )
            if epoch_callback is not None:
                epoch_callback(ep + 1, None, None, None, None)
        return None, None

    best_state = (
        copy.deepcopy(initial_best_state)
        if initial_best_state is not None
        else copy.deepcopy(model.state_dict())
    )
    best_epoch: int | None = initial_best_epoch
    best_val_score = (
        float(initial_best_val_score)
        if initial_best_val_score is not None
        else float("-inf")
    )
    epochs_without_improvement = 0

    if initial_best_val_score is None:
        baseline_epoch = int(start_epoch)
        best_val_score = float(val_score_fn())
        if not math.isfinite(best_val_score):
            raise RuntimeError(
                f"Training produced a non-finite validation {metric_name} "
                f"at epoch {baseline_epoch}: {best_val_score}."
            )
        best_epoch = baseline_epoch
        best_state = copy.deepcopy(model.state_dict())
        print(
            f"        epoch {baseline_epoch}/{epochs}  "
            f"val_{metric_name}={best_val_score:.4f}  (baseline)",
            flush=True,
        )
        if epoch_callback is not None:
            epoch_callback(
                baseline_epoch,
                best_val_score,
                best_epoch,
                best_val_score,
                best_state,
            )

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_epoch_fn()
        epoch_number = epoch + 1
        should_validate = (
            epoch_number % validation_interval == 0 or epoch_number == epochs
        )
        if not should_validate:
            if epoch == 0 or epoch_number % log_every == 0:
                epoch_time = time.time() - t0
                print(
                    f"        epoch {epoch_number}/{epochs}  "
                    f"({epoch_time:.1f}s training; validation every "
                    f"{validation_interval} epochs)",
                    flush=True,
                )
            if epoch_callback is not None:
                epoch_callback(
                    epoch_number,
                    None,
                    best_epoch,
                    best_val_score,
                    best_state,
                )
            continue

        val_score = float(val_score_fn())
        if not math.isfinite(val_score):
            raise RuntimeError(
                f"Training produced a non-finite validation "
                f"{metric_name}: {val_score}."
            )
        epoch_time = time.time() - t0
        if val_score > (best_val_score + min_delta):
            best_val_score = val_score
            best_epoch = epoch_number
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epoch_number - int(best_epoch)

        if epoch_number % log_every == 0 or epoch == 0 or validation_interval > 1:
            print(
                f"        epoch {epoch_number}/{epochs}  "
                f"val_{metric_name}={val_score:.4f}  "
                f"best={best_val_score:.4f}  "
                f"no_improve={epochs_without_improvement}  ({epoch_time:.1f}s/epoch)",
                flush=True,
            )

        if epoch_callback is not None:
            epoch_callback(
                epoch_number,
                val_score,
                best_epoch,
                best_val_score,
                best_state,
            )
        if epochs_without_improvement >= patience:
            print(
                f"        early stopping at epoch {epoch_number}, "
                f"best_epoch={best_epoch}, "
                f"best_val_{metric_name}={best_val_score:.4f}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_score


def run_wrapper_training(
    wrapper,
    train_epoch_fn,
    val_score_fn,
    start_epoch,
    checkpoint_path,
) -> None:
    """Run metric-based early stopping for a Torch wrapper."""
    wrapper.best_epoch_, wrapper.best_val_score_ = run_training_with_early_stopping(
        wrapper.model,
        wrapper.epochs,
        train_epoch_fn,
        val_score_fn=val_score_fn,
        metric_name=wrapper.validation_metric_,
        patience=wrapper.early_stopping_patience,
        min_delta=wrapper.early_stopping_min_delta,
        validation_interval=getattr(wrapper, "validation_interval", 1),
        start_epoch=start_epoch,
        initial_best_state=wrapper.best_model_state_,
        initial_best_epoch=wrapper.best_epoch_,
        initial_best_val_score=wrapper.best_val_score_,
        epoch_callback=make_torch_checkpoint_callback(wrapper, checkpoint_path),
    )


def compute_ec_batch_loss(logits_by_level, y_t, target_mask_t, criterion):
    """Masked multi-head BCE loss for hierarchical EC targets."""
    loss_terms = []
    for level_name, logits in logits_by_level.items():
        level_loss = criterion(logits.float(), y_t[level_name])
        mask = target_mask_t[level_name].float()
        if mask.ndim == 1:
            mask = mask[:, None].expand_as(level_loss)
        denom = mask.sum()
        if denom.item() == 0:
            continue
        loss_terms.append((level_loss * mask).sum() / denom)
    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()
