"""Shared utilities for model wrappers."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mean_pool(out: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool (batch, seq_len, dim) -> (batch, dim) using a boolean mask."""
    if mask is None:
        return out.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def make_optimizer(parameters, optimizer_name: str, lr: float):
    """Instantiate the optimizer requested by a lightweight wrapper config."""
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer_name '{optimizer_name}'.")


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
    else:
        raise ValueError(f"fit_loader does not support batch_kind='{batch_kind}'")
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


def run_training_with_early_stopping(
    model: nn.Module,
    epochs: int,
    train_epoch_fn,
    val_loss_fn=None,
    patience: int | None = None,
    min_delta: float = 0.0,
    log_every: int = 10,
) -> tuple[int | None, float | None]:
    """Run a training loop and optionally restore the best validation checkpoint."""
    import time as _time

    if val_loss_fn is None or patience is None or patience <= 0:
        for ep in range(epochs):
            t0 = _time.time()
            train_epoch_fn()
            if (ep + 1) % log_every == 0 or ep == 0:
                print(f"        epoch {ep+1}/{epochs}  ({_time.time()-t0:.1f}s/epoch)", flush=True)
        return None, None

    best_state = copy.deepcopy(model.state_dict())
    best_epoch: int | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = _time.time()
        train_epoch_fn()
        val_loss = float(val_loss_fn())
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
            continue

        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(
                f"        early stopping at epoch {epoch+1}, "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_loss


# Backward-compatible aliases for modules that still import the underscored names.
_masked_mean_pool = masked_mean_pool
_make_optimizer = make_optimizer
_pad_1d_array_list = pad_1d_array_list
_batch_to_input = batch_to_input
_prepare_inputs = prepare_inputs
_run_training_with_early_stopping = run_training_with_early_stopping
