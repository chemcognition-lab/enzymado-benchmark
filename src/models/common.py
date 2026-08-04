"""Shared utilities for model wrappers."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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

    def _callback(epoch: int, best_epoch, best_val_loss, best_state) -> None:
        wrapper.current_epoch_ = int(epoch)
        wrapper.best_epoch_ = best_epoch
        wrapper.best_val_loss_ = best_val_loss
        wrapper.best_model_state_ = copy.deepcopy(best_state) if best_state is not None else None
        if not hasattr(wrapper, "training_log_") or wrapper.training_log_ is None:
            wrapper.training_log_ = []
        wrapper.training_log_.append(
            {
                "epoch": int(epoch),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        )
        if epoch % checkpoint_every == 0:
            save_torch_wrapper_checkpoint(wrapper, path)

    return _callback


def run_training_with_early_stopping(
    model: nn.Module,
    epochs: int,
    train_epoch_fn,
    val_loss_fn=None,
    patience: int | None = None,
    min_delta: float = 0.0,
    log_every: int = 10,
    start_epoch: int = 0,
    initial_best_state: dict | None = None,
    initial_best_epoch: int | None = None,
    initial_best_val_loss: float | None = None,
    epoch_callback=None,
) -> tuple[int | None, float | None]:
    """Run a training loop and optionally restore the best validation checkpoint."""
    import time as _time

    if val_loss_fn is None or patience is None or patience <= 0:
        for ep in range(start_epoch, epochs):
            t0 = _time.time()
            train_epoch_fn()
            if (ep + 1) % log_every == 0 or ep == 0:
                print(f"        epoch {ep+1}/{epochs}  ({_time.time()-t0:.1f}s/epoch)", flush=True)
            if epoch_callback is not None:
                epoch_callback(ep + 1, None, None, None)
        return None, None

    best_state = copy.deepcopy(initial_best_state) if initial_best_state is not None else copy.deepcopy(model.state_dict())
    best_epoch: int | None = initial_best_epoch
    best_val_loss = float(initial_best_val_loss) if initial_best_val_loss is not None else float("inf")
    epochs_without_improvement = 0

    # Make the untrained (or currently resumed) model a real early-stopping
    # candidate. Without this baseline, one catastrophic optimizer step could
    # become the best checkpoint simply because every later step was worse.
    if initial_best_val_loss is None:
        baseline_epoch = int(start_epoch)
        best_val_loss = float(val_loss_fn())
        if not math.isfinite(best_val_loss):
            raise RuntimeError(
                f"Training produced a non-finite validation loss at epoch "
                f"{baseline_epoch}: {best_val_loss}."
            )
        best_epoch = baseline_epoch
        best_state = copy.deepcopy(model.state_dict())
        print(
            f"        epoch {baseline_epoch}/{epochs}  "
            f"val_loss={best_val_loss:.4f}  (baseline)",
            flush=True,
        )
        if epoch_callback is not None:
            epoch_callback(baseline_epoch, best_epoch, best_val_loss, best_state)

    for epoch in range(start_epoch, epochs):
        t0 = _time.time()
        train_epoch_fn()
        val_loss = float(val_loss_fn())
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Training produced a non-finite validation loss: {val_loss}.")
        epoch_time = _time.time() - t0

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"        epoch {epoch+1}/{epochs}  "
                f"val_loss={val_loss:.4f}  best={best_val_loss:.4f}  "
                f"no_improve={epochs_without_improvement}  ({epoch_time:.1f}s/epoch)",
                flush=True,
            )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if epoch_callback is not None:
                epoch_callback(epoch + 1, best_epoch, best_val_loss, best_state)
            continue

        epochs_without_improvement += 1
        if epoch_callback is not None:
            epoch_callback(epoch + 1, best_epoch, best_val_loss, best_state)
        if epochs_without_improvement >= patience:
            print(
                f"        early stopping at epoch {epoch+1}, "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_loss


def run_wrapper_training(wrapper, train_epoch_fn, val_loss_fn, start_epoch, checkpoint_path):
    """Run the standard early-stopping training loop against any wrapper exposing
    .model/.epochs/.early_stopping_patience/.early_stopping_min_delta/.best_model_state_/
    .best_epoch_/.best_val_loss_, mutating .best_epoch_/.best_val_loss_ in place."""
    wrapper.best_epoch_, wrapper.best_val_loss_ = run_training_with_early_stopping(
        wrapper.model,
        wrapper.epochs,
        train_epoch_fn,
        val_loss_fn=val_loss_fn,
        patience=wrapper.early_stopping_patience,
        min_delta=wrapper.early_stopping_min_delta,
        start_epoch=start_epoch,
        initial_best_state=wrapper.best_model_state_,
        initial_best_epoch=wrapper.best_epoch_,
        initial_best_val_loss=wrapper.best_val_loss_,
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

