"""Shared utilities for model wrappers."""

from __future__ import annotations

import collections.abc
import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import sys
import time
import typing

from jaxtyping import Bool, Float
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import tqdm

from enzymado.evaluation import metrics_registry
from enzymado.models import training_protocol

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_CHECKPOINT_SCHEMA_VERSION = 3

type ModelState = dict[str, torch.Tensor]
type TrainEpoch = collections.abc.Callable[[], object]
type ValidationLoss = collections.abc.Callable[[], float | None]
type EpochCallback = collections.abc.Callable[
    [int, int | None, float | None, ModelState | None],
    None,
]


def seed_random_generators(
    seed: int,
    *,
    deterministic_algorithms: bool = False,  # noqa: FBT002
) -> None:
    """Seed benchmark RNGs and explicitly configure PyTorch determinism."""
    seed = int(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(mode=deterministic_algorithms)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def masked_mean_pool(
    out: Float[Tensor, "batch seq dim"],
    mask: Bool[Tensor, "batch seq"] | None,
) -> Float[Tensor, "batch dim"]:
    """Mean-pool residue embeddings, ignoring padded positions."""
    if mask is None:
        return out.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def make_optimizer(
    parameters, optimizer_name: str, lr: float, weight_decay: float = 0.0
):
    """Instantiate the optimizer requested by a lightweight wrapper config."""
    match optimizer_name:
        case "sgd":
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        case "adam":
            optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        case "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        case _:
            raise ValueError(f"Unsupported optimizer_name '{optimizer_name}'.")
    optimizer.register_step_pre_hook(training_protocol.check_optimizer_gradients)
    optimizer.register_step_post_hook(training_protocol.check_optimizer_parameters)
    return optimizer


def fit_feature_scaler(x, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple per-feature standardizer for 2D or padded 3D inputs."""
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim == 2:
        observed = x_np
    elif x_np.ndim == 3:
        if mask is None:
            observed = x_np.reshape(-1, x_np.shape[-1])
        else:
            observed = x_np[np.asarray(mask, dtype=bool)]
    else:
        raise ValueError(f"x must be 2D or 3D, got shape {x_np.shape}")
    mean = observed.mean(axis=0).astype(np.float32, copy=False)
    scale = observed.std(axis=0).astype(np.float32, copy=False)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def apply_feature_scaler(x, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a fitted per-feature standardizer."""
    return (np.asarray(x, dtype=np.float32) - mean) / scale


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
    """Extract (x_t, mask_t, y_np) from a collated batch for residue-aware training."""
    if batch_kind == "protein_global":
        x_t = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(
            device
        )
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    elif batch_kind == "mutation_delta_global":
        x_t = torch.from_numpy(
            np.asarray(batch.protein_delta_emb, dtype=np.float32)
        ).to(device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
    elif batch_kind == "protein_ligand_global":
        raise ValueError(
            "batch_kind='protein_ligand_global' is not supported by standard "
            "MLP/Linear wrappers "
            "because they cannot combine protein and ligand features. "
            "Use AttentionRegressorTorch or AttentionBinaryClassifierTorch instead."
        )
    else:
        raise ValueError(f"Unrecognised batch_kind='{batch_kind}'.")
    y_np = np.asarray(batch.target, dtype=np.float32)
    return x_t, mask_t, y_np


def prepare_inputs(
    x: np.ndarray, mask: np.ndarray | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return (x_tensor, mask_tensor, input_dim) on the given device."""
    x = np.asarray(x, dtype=np.float32)
    mask_t = None
    if x.ndim == 3:
        input_dim = x.shape[2]
        if mask is not None:
            mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
    elif x.ndim == 2:
        input_dim = x.shape[1]
    else:
        raise ValueError(f"x must be 2D or 3D, got shape {x.shape}")
    return torch.from_numpy(x).to(device), mask_t, input_dim


def validation_objective(
    y_true,
    scores,
    metric_name: str,
    target_mask=None,
) -> float | None:
    """Minimize a negated metric; constant Kendall predictions are ineligible."""
    y_array = np.asarray(y_true, dtype=np.float64).ravel()
    score_array = np.asarray(scores, dtype=np.float64).ravel()
    if target_mask is not None:
        mask = np.asarray(target_mask, dtype=bool).ravel()
        y_array, score_array = y_array[mask], score_array[mask]
    if not np.isfinite(y_array).all() or not np.isfinite(score_array).all():
        raise FloatingPointError("Non-finite validation labels or predictions.")
    if y_array.size < 2 or np.unique(y_array).size < 2:
        raise ValueError("Validation labels do not support the selection metric.")
    if metric_name == "auprc":
        return -metrics_registry.auprc_from_scores(y_array, score_array)
    if metric_name != "kendall_tau":
        raise ValueError(f"Unsupported validation metric: {metric_name}.")
    if np.unique(score_array).size < 2:
        return None
    score = metrics_registry.regression_kendall_tau(y_array, score_array)
    if not math.isfinite(score):
        raise FloatingPointError("Non-finite validation metric.")
    return -float(score)


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


def _restore_rng_state(wrapper, dataloader=None) -> None:
    """Restore stochastic state captured by a versioned training checkpoint."""
    resume_state = getattr(wrapper, "resume_state_", None)
    if resume_state is None:
        return

    schema_version = getattr(wrapper, "checkpoint_schema_version_", None)
    if schema_version != TORCH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported PyTorch checkpoint schema {schema_version!r}; "
            f"expected {TORCH_CHECKPOINT_SCHEMA_VERSION}."
        )
    generator_state = resume_state["dataloader_generator_state"]

    random.setstate(resume_state["python_rng_state"])
    np.random.set_state(resume_state["numpy_rng_state"])
    torch.set_rng_state(resume_state["torch_rng_state"])
    cuda_rng_state = resume_state["cuda_rng_state_all"]
    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state)
    if generator_state is not None:
        dataloader.generator.set_state(generator_state)


def maybe_resume_torch_wrapper(
    wrapper,
    path: str | Path | None,
    *,
    dataloader=None,
) -> int:
    """Load a previously saved wrapper checkpoint into the current wrapper instance."""
    path_obj = Path(path) if path is not None else None
    wrapper.training_contract_ = training_protocol.resume_contract(
        wrapper, path_obj, dataloader
    )
    if path_obj is None:
        return 0

    if not path_obj.exists():
        return 0

    loaded = torch.load(path_obj, map_location="cpu", weights_only=False)
    if not isinstance(loaded, wrapper.__class__):
        raise TypeError(
            f"Checkpoint at {path_obj} contains {type(loaded).__name__}, "
            f"expected {wrapper.__class__.__name__}."
        )

    if (
        getattr(loaded, "checkpoint_schema_version_", None)
        != TORCH_CHECKPOINT_SCHEMA_VERSION
        or getattr(loaded, "training_contract_", None) != wrapper.training_contract_
        or getattr(loaded, "resume_state_", None) is None
    ):
        raise ValueError(
            "Incompatible training checkpoint: protocol, configuration or provenance "
            "changed. Start a new run directory; old weights remain usable "
            "for inference."
        )

    current_device = getattr(wrapper, "device", DEVICE)
    wrapper.__dict__.update(loaded.__dict__)
    wrapper.device = torch.device(current_device)
    if getattr(wrapper, "model", None) is not None:
        wrapper.model = wrapper.model.to(wrapper.device)
    for attr_name, attr_value in wrapper.__dict__.items():
        if isinstance(attr_value, torch.Tensor):
            wrapper.__dict__[attr_name] = attr_value.to(wrapper.device)
    _restore_rng_state(wrapper, dataloader)
    return int(getattr(wrapper, "current_epoch_", 0))


def restore_torch_training_state(wrapper, optimizer, scheduler=None) -> None:
    """Restore optimizer and optional scheduler state after their construction."""
    resume_state = getattr(wrapper, "resume_state_", None)
    if resume_state is None:
        return

    optimizer.load_state_dict(resume_state["optimizer_state_dict"])
    scheduler_state = resume_state["scheduler_state_dict"]
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    wrapper.resume_state_ = None


def _capture_torch_training_state(optimizer, scheduler, dataloader) -> dict:
    """Capture every stochastic and optimizer state required for exact resume."""
    generator = getattr(dataloader, "generator", None)
    return {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "dataloader_generator_state": (
            generator.get_state() if generator is not None else None
        ),
    }


def make_torch_checkpoint_callback(
    wrapper,
    path: str | Path | None,
    *,
    optimizer,
    scheduler=None,
    dataloader=None,
    checkpoint_every: int = 1,
):
    """Build an epoch-end callback that updates state and saves checkpoints."""
    checkpoint_every = int(checkpoint_every)
    if not hasattr(wrapper, "training_contract_"):
        wrapper.training_contract_ = training_protocol.resume_contract(
            wrapper, Path(path) if path is not None else None, dataloader
        )

    def _callback(epoch: int, best_epoch, best_val_loss, best_state) -> None:
        wrapper.current_epoch_ = int(epoch)
        wrapper.best_epoch_ = best_epoch
        wrapper.best_val_loss_ = best_val_loss
        wrapper.best_model_state_ = (
            copy.deepcopy(best_state) if best_state is not None else None
        )
        if not hasattr(wrapper, "training_log_") or wrapper.training_log_ is None:
            wrapper.training_log_ = []
        if epoch == 0 and wrapper.training_log_:
            return
        wrapper.training_log_.append(
            {
                "epoch": int(epoch),
                "best_epoch": best_epoch,
                "selection_objective": training_protocol.selection_protocol(
                    type(wrapper)
                )["objective"],
                "validation_score": getattr(wrapper, "validation_score_", None),
                "best_validation_score": best_val_loss,
                "baseline_validation_score": getattr(wrapper, "baseline_score_", None),
                "improved_over_initialization": (
                    best_val_loss
                    < wrapper.baseline_score_ - wrapper.early_stopping_min_delta
                    if best_epoch is not None
                    and getattr(wrapper, "baseline_score_", None) is not None
                    else None
                ),
                "status": (
                    "baseline"
                    if epoch == 0
                    else "trained_without_validation"
                    if not hasattr(wrapper, "baseline_score_")
                    else "undefined_validation_metric"
                    if wrapper.validation_score_ is None
                    else "trained"
                ),
            }
        )
        if path is not None and epoch % checkpoint_every == 0:
            wrapper.checkpoint_schema_version_ = TORCH_CHECKPOINT_SCHEMA_VERSION
            wrapper.resume_state_ = _capture_torch_training_state(
                optimizer,
                scheduler,
                dataloader,
            )
            try:
                save_torch_wrapper_checkpoint(wrapper, path)
            finally:
                wrapper.resume_state_ = None

    return _callback


def make_validation_callback(wrapper, checkpoint_path):
    """Record validation values and preserve initialization as a reference."""

    def observe(epoch: int, score: float | None) -> None:
        wrapper.validation_score_ = score
        if epoch != 0:
            return
        wrapper.baseline_score_ = score
        if checkpoint_path is not None:
            _atomic_torch_save(
                wrapper.model.state_dict(),
                Path(checkpoint_path).parent / "initial_model.pt",
            )

    return observe


def _epoch_progress_bar(start_epoch: int, epochs: int) -> tqdm.tqdm:
    """Build the shared epoch progress bar for the torch training loops."""
    return tqdm.tqdm(
        range(start_epoch, epochs),
        total=epochs,
        initial=start_epoch,
        desc="        epochs",
        unit="epoch",
        file=sys.stdout,
        mininterval=5.0,
        leave=False,
    )


@dataclass
class _EarlyStoppingState:
    """Mutable validation-loss state carried across epochs and checkpoints."""

    best_model: ModelState
    best_epoch: int | None
    best_val_loss: float
    epochs_without_improvement: int
    # Patience is measured against this, not against best_val_loss. Using one
    # value for both made min_delta silently discard better checkpoints: a run
    # improving by less than min_delta each epoch kept the older, worse model.
    patience_reference: float = float("inf")


def _copy_model_state(model: nn.Module) -> ModelState:
    """Copy a module state dict for restoration or checkpoint callbacks."""
    return typing.cast(ModelState, copy.deepcopy(model.state_dict()))


def _run_without_validation(
    epochs: int,
    train_epoch_fn: TrainEpoch,
    log_every: int,
    start_epoch: int,
    epoch_callback: EpochCallback | None,
) -> None:
    """Run all requested epochs when validation early stopping is disabled."""
    progress = _epoch_progress_bar(start_epoch, epochs)
    try:
        for epoch in progress:
            started_at = time.time()
            train_epoch_fn()
            epoch_time = time.time() - started_at
            progress.set_postfix({"s/epoch": f"{epoch_time:.1f}"})
            if (epoch + 1) % log_every == 0 or epoch == 0:
                progress.write(
                    f"        epoch {epoch + 1}/{epochs}  ({epoch_time:.1f}s/epoch)"
                )
            if epoch_callback is not None:
                epoch_callback(epoch + 1, None, None, None)
    finally:
        progress.close()


def _initialize_early_stopping(
    model: nn.Module,
    epochs: int,
    start_epoch: int,
    val_loss_fn: ValidationLoss,
    initial_best_state: ModelState | None,
    initial_best_epoch: int | None,
    initial_best_val_loss: float | None,
    epoch_callback: EpochCallback | None,
    validation_callback,
) -> _EarlyStoppingState:
    """Restore resumed state or evaluate the untrained validation baseline."""
    best_model = (
        copy.deepcopy(initial_best_state)
        if initial_best_state is not None
        else _copy_model_state(model)
    )
    best_val_loss = (
        float(initial_best_val_loss)
        if initial_best_val_loss is not None
        else float("inf")
    )
    epochs_without_improvement = (
        max(0, int(start_epoch) - int(initial_best_epoch))
        if initial_best_val_loss is not None and initial_best_epoch is not None
        else start_epoch
    )
    state = _EarlyStoppingState(
        best_model=best_model,
        best_epoch=initial_best_epoch,
        best_val_loss=best_val_loss,
        epochs_without_improvement=epochs_without_improvement,
        # A resumed run carries its patience reference over, so the counter
        # keeps measuring against the same value it did before the restart.
        patience_reference=best_val_loss,
    )
    if initial_best_val_loss is not None:
        return state

    if start_epoch != 0:
        return state
    baseline_score = val_loss_fn()
    if baseline_score is not None and not math.isfinite(baseline_score):
        raise FloatingPointError("Non-finite validation objective at initialization.")
    if validation_callback is not None:
        validation_callback(0, baseline_score)
    print(f"        epoch 0/{epochs}  baseline_objective={baseline_score}", flush=True)
    if epoch_callback is not None:
        epoch_callback(0, None, None, None)
    return state


def _log_validation_epoch(
    progress: tqdm.tqdm,
    epoch: int,
    epochs: int,
    epoch_time: float,
    val_loss: float,
    state: _EarlyStoppingState,
    log_every: int,
) -> None:
    """Update the progress display for one validation epoch."""
    progress.set_postfix(
        {
            "validation_objective": f"{val_loss:.4f}",
            "best": f"{state.best_val_loss:.4f}",
            "no_improve": state.epochs_without_improvement,
            "s/epoch": f"{epoch_time:.1f}",
        }
    )
    if (epoch + 1) % log_every == 0 or epoch == 0:
        progress.write(
            f"        epoch {epoch + 1}/{epochs}  "
            f"validation_objective={val_loss:.4f}  best={state.best_val_loss:.4f}  "
            f"no_improve={state.epochs_without_improvement}  "
            f"({epoch_time:.1f}s/epoch)"
        )


def _notify_epoch(
    epoch_callback: EpochCallback | None,
    epoch: int,
    state: _EarlyStoppingState,
) -> None:
    """Send the current early-stopping state to an optional checkpoint hook."""
    if epoch_callback is not None:
        epoch_callback(
            epoch,
            state.best_epoch,
            state.best_val_loss if state.best_epoch is not None else None,
            state.best_model if state.best_epoch is not None else None,
        )


def run_training_with_early_stopping(
    model: nn.Module,
    epochs: int,
    train_epoch_fn: TrainEpoch,
    val_loss_fn: ValidationLoss | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    log_every: int = 10,
    start_epoch: int = 0,
    initial_best_state: ModelState | None = None,
    initial_best_epoch: int | None = None,
    initial_best_val_loss: float | None = None,
    epoch_callback: EpochCallback | None = None,
    validation_callback=None,
) -> tuple[int | None, float | None]:
    """Select a trained epoch and retain initialization only as a reference."""
    if initial_best_epoch is not None and initial_best_epoch < 1:
        raise ValueError("An initialization checkpoint is not a trained candidate.")
    if val_loss_fn is None or patience is None or patience <= 0:
        _run_without_validation(
            epochs,
            train_epoch_fn,
            log_every,
            start_epoch,
            epoch_callback,
        )
        return None, None

    state = _initialize_early_stopping(
        model,
        epochs,
        start_epoch,
        val_loss_fn,
        initial_best_state,
        initial_best_epoch,
        initial_best_val_loss,
        epoch_callback,
        validation_callback,
    )
    if state.best_epoch is not None and state.epochs_without_improvement >= patience:
        model.load_state_dict(state.best_model)
        return state.best_epoch, state.best_val_loss
    progress = _epoch_progress_bar(start_epoch, epochs)
    try:
        for epoch in progress:
            started_at = time.time()
            train_epoch_fn()
            score = val_loss_fn()
            if score is not None and not math.isfinite(score):
                raise RuntimeError(
                    f"Training produced a non-finite validation objective: {score}."
                )
            if validation_callback is not None:
                validation_callback(epoch + 1, score)
            val_loss = float(score) if score is not None else float("inf")
            epoch_time = time.time() - started_at
            _log_validation_epoch(
                progress,
                epoch,
                epochs,
                epoch_time,
                val_loss,
                state,
                log_every,
            )

            # Keep the best epoch outright, and count patience separately.
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.best_epoch = epoch + 1
                state.best_model = _copy_model_state(model)
            if val_loss < state.patience_reference - min_delta:
                state.patience_reference = val_loss
                state.epochs_without_improvement = 0
            else:
                state.epochs_without_improvement += 1
            _notify_epoch(epoch_callback, epoch + 1, state)
            if state.epochs_without_improvement >= patience:
                print(
                    f"        early stopping at epoch {epoch + 1}, "
                    f"best_epoch={state.best_epoch}, "
                    f"best_validation_objective={state.best_val_loss:.4f}",
                    flush=True,
                )
                break
    finally:
        progress.close()

    if state.best_epoch is None:
        raise ValueError("No trained epoch has a defined validation metric.")
    model.load_state_dict(state.best_model)
    return state.best_epoch, state.best_val_loss


def run_wrapper_training(
    wrapper,
    train_epoch_fn,
    val_loss_fn,
    start_epoch,
    checkpoint_path,
    *,
    optimizer,
    scheduler=None,
    dataloader=None,
):
    """Run the shared validation-loss training loop for a configured wrapper."""
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
        validation_callback=make_validation_callback(wrapper, checkpoint_path),
        epoch_callback=make_torch_checkpoint_callback(
            wrapper,
            checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
        ),
    )


class StreamingAUPRC:
    """Accumulate validation probabilities into the leaderboard's AUPRC histogram.

    Memory stays at 2 * RANKING_BINS counters however large the validation set,
    and the value equals metrics_registry.auprc_from_scores on all scores at once.
    """

    def __init__(self, device: torch.device) -> None:
        self.totals = torch.zeros(
            2 * metrics_registry.RANKING_BINS, dtype=torch.int64, device=device
        )

    def update(self, probabilities: Tensor, labels: Tensor, mask: Tensor) -> None:
        mask = mask.bool()
        if mask.ndim < labels.ndim:
            mask = mask.reshape(mask.shape + (1,) * (labels.ndim - mask.ndim))
        mask = mask.expand_as(labels)
        scores = probabilities.double()[mask]
        labels = labels[mask]
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite validation probabilities.")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("AUPRC probabilities must lie in [0, 1].")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("AUPRC selection requires binary labels.")
        n_bins = metrics_registry.RANKING_BINS
        levels = torch.clamp(torch.round(scores * (n_bins - 1)).long(), 0, n_bins - 1)
        offsets = (labels == 1).long() * n_bins + levels
        self.totals += torch.bincount(offsets, minlength=2 * n_bins)

    def value(self) -> float:
        return metrics_registry.auprc_from_histogram(self.totals.cpu().numpy())


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
