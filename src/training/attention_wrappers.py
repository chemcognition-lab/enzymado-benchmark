"""Training wrappers for attention-pooled global models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.attention_blocks import (
    AttentionMultiHeadPredictor,
    AttentionPredictor,
    AttentionResiduePredictor,
    _seq_emb_to_tensor,
    attention_autocast,
    batch_to_attention_tensors,
)
from models.common import (
    DEVICE,
    compute_ec_batch_loss,
    make_optimizer,
    make_torch_checkpoint_callback,
    maybe_resume_torch_wrapper,
    mean_losses,
    pad_1d_array_list,
    run_training_with_early_stopping,
)


_SEQUENCE_VARIANCE_EPS = 1e-12
_DIAGNOSTIC_FLUSH_EVERY = 100
_BATCH_DIAGNOSTIC_FIELDS = (
    "epoch",
    "batch",
    "global_step",
    "loss",
    "max_abs_z",
    "grad_norm",
    "gradient_clipped",
    "problematic",
    "problem_reasons",
    "sample_ids",
)


class _BaseAttentionWrapper:
    """Shared training boilerplate for attention-based wrappers."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...],
        lr: float,
        epochs: int,
        optimizer_name: str,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        gradient_clip_norm: float | None = None,
        pre_attention_layer_norm: bool = False,
        record_batch_diagnostics: bool = False,
        diagnostic_loss_threshold: float | None = 100.0,
        diagnostic_z_threshold: float | None = 20.0,
        diagnostic_grad_norm_threshold: float | None = 1000.0,
        device: str | torch.device | None = None,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        self.gradient_clip_norm = gradient_clip_norm
        self.pre_attention_layer_norm = bool(pre_attention_layer_norm)
        self.record_batch_diagnostics = bool(record_batch_diagnostics)
        self.diagnostic_loss_threshold = diagnostic_loss_threshold
        self.diagnostic_z_threshold = diagnostic_z_threshold
        self.diagnostic_grad_norm_threshold = diagnostic_grad_norm_threshold
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.input_dim_ = None
        self.extra_dim_ = 0
        self.batch_kind_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.seq_mean_ = None
        self.seq_scale_ = None
        self.extra_mean_ = None
        self.extra_scale_ = None
        self.first_problematic_batch_ = None
        self.diagnostic_global_step_ = 0
        self._diagnostic_epoch_ = 0
        self._batch_diagnostics_path = None
        self._first_problematic_batch_path = None
        self._batch_diagnostic_buffer = []

    def _configure_batch_diagnostics(
        self,
        checkpoint_path: str | Path | None,
        start_epoch: int,
    ) -> None:
        """Point optional streaming diagnostics at the current run directory."""
        self._diagnostic_epoch_ = int(start_epoch)
        if not self.record_batch_diagnostics or checkpoint_path is None:
            self._batch_diagnostics_path = None
            self._first_problematic_batch_path = None
            return
        run_dir = Path(checkpoint_path).parent
        self._batch_diagnostics_path = run_dir / "batch_diagnostics.csv"
        self._first_problematic_batch_path = run_dir / "first_problematic_batch.json"

    def _next_diagnostic_epoch(self) -> int:
        self._diagnostic_epoch_ += 1
        return self._diagnostic_epoch_

    @staticmethod
    def _max_abs_valid_z(seq_t: torch.Tensor, mask_t: torch.Tensor | None) -> float:
        observed = seq_t[mask_t] if mask_t is not None else seq_t
        if observed.numel() == 0:
            return 0.0
        return float(observed.detach().abs().max().item())

    def _prepare_gradients(self) -> float | None:
        """Measure the pre-clipping norm and apply optional clipping once."""
        if self.gradient_clip_norm is None and not self.record_batch_diagnostics:
            return None
        max_norm = (
            float(self.gradient_clip_norm)
            if self.gradient_clip_norm is not None
            else float("inf")
        )
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        return float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

    def _problem_reasons(
        self,
        *,
        loss: float,
        max_abs_z: float,
        grad_norm: float | None,
    ) -> list[str]:
        reasons = []
        if not np.isfinite(loss):
            reasons.append("non_finite_loss")
        elif self.diagnostic_loss_threshold is not None and loss >= self.diagnostic_loss_threshold:
            reasons.append("loss_threshold")
        if not np.isfinite(max_abs_z):
            reasons.append("non_finite_z")
        elif self.diagnostic_z_threshold is not None and max_abs_z >= self.diagnostic_z_threshold:
            reasons.append("z_threshold")
        if grad_norm is not None:
            if not np.isfinite(grad_norm):
                reasons.append("non_finite_grad_norm")
            elif (
                self.diagnostic_grad_norm_threshold is not None
                and grad_norm >= self.diagnostic_grad_norm_threshold
            ):
                reasons.append("grad_norm_threshold")
        return reasons

    def _record_batch_diagnostic(
        self,
        *,
        epoch: int,
        batch_index: int,
        batch,
        loss: float,
        max_abs_z: float,
        grad_norm: float | None,
    ) -> None:
        if not self.record_batch_diagnostics:
            return
        self.diagnostic_global_step_ += 1
        reasons = self._problem_reasons(
            loss=loss,
            max_abs_z=max_abs_z,
            grad_norm=grad_norm,
        )
        row = {
            "epoch": int(epoch),
            "batch": int(batch_index),
            "global_step": int(self.diagnostic_global_step_),
            "loss": loss,
            "max_abs_z": max_abs_z,
            "grad_norm": grad_norm,
            "gradient_clipped": bool(
                self.gradient_clip_norm is not None
                and grad_norm is not None
                and grad_norm > self.gradient_clip_norm
            ),
            "problematic": bool(reasons),
            "problem_reasons": ";".join(reasons),
            "sample_ids": ";".join(str(value) for value in getattr(batch, "sample_id", [])),
        }
        self._batch_diagnostic_buffer.append(row)
        if len(self._batch_diagnostic_buffer) >= _DIAGNOSTIC_FLUSH_EVERY:
            self._flush_batch_diagnostics()
        if reasons and self.first_problematic_batch_ is None:
            self.first_problematic_batch_ = dict(row)
            if self._first_problematic_batch_path is not None:
                self._first_problematic_batch_path.write_text(
                    json.dumps(row, indent=4) + "\n",
                    encoding="utf-8",
                )

    def _flush_batch_diagnostics(self) -> None:
        """Append buffered rows in one write to reduce filesystem overhead."""
        if not self._batch_diagnostic_buffer or self._batch_diagnostics_path is None:
            return
        write_header = (
            not self._batch_diagnostics_path.exists()
            or self._batch_diagnostics_path.stat().st_size == 0
        )
        with self._batch_diagnostics_path.open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=_BATCH_DIAGNOSTIC_FIELDS,
                lineterminator="\n",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(self._batch_diagnostic_buffer)
        self._batch_diagnostic_buffer.clear()

    def _infer_dims(self, dataloader, batch_kind: str) -> tuple[int, int]:
        first_batch = next(iter(dataloader))
        seq_t, _, extra_t = batch_to_attention_tensors(first_batch, batch_kind, self.device)
        input_dim = int(seq_t.shape[2])
        extra_dim = int(extra_t.shape[1]) if extra_t is not None else 0
        return input_dim, extra_dim

    def _fit_extra_scaler(self, dataloader, batch_kind: str) -> None:
        """Fit train-only standardization for ligand/extra features."""
        extras = []
        for batch in dataloader:
            _, _, extra_t = batch_to_attention_tensors(batch, batch_kind, torch.device("cpu"))
            if extra_t is not None:
                extras.append(extra_t.float().cpu().numpy())
        if not extras:
            self.extra_mean_ = None
            self.extra_scale_ = None
            return
        extra_np = np.concatenate(extras, axis=0).astype(np.float32, copy=False)
        mean = extra_np.mean(axis=0, keepdims=True)
        scale = extra_np.std(axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        self.extra_mean_ = torch.from_numpy(mean.astype(np.float32, copy=False)).to(self.device)
        self.extra_scale_ = torch.from_numpy(scale.astype(np.float32, copy=False)).to(self.device)

    def _fit_sequence_scaler_from_batches(self, dataloader, tensor_fn) -> None:
        """Fit train-only per-channel standardization over valid residue embeddings."""
        mean = None
        m2 = None
        count = 0
        cpu = torch.device("cpu")
        for batch in dataloader:
            seq_t, mask_t = tensor_fn(batch, cpu)
            mask_t = mask_t.bool()
            observed = seq_t[mask_t].to(dtype=torch.float64)
            if observed.numel() == 0:
                continue
            batch_count = int(observed.shape[0])
            batch_var, batch_mean = torch.var_mean(observed, dim=0, correction=0)
            batch_m2 = batch_var * batch_count
            if mean is None or m2 is None:
                mean = batch_mean
                m2 = batch_m2
                count = batch_count
                continue

            total_count = count + batch_count
            delta = batch_mean - mean
            mean = mean + delta * (batch_count / total_count)
            m2 = (
                m2
                + batch_m2
                + delta.square() * (count * batch_count / total_count)
            )
            count = total_count

        if count == 0 or mean is None or m2 is None:
            self.seq_mean_ = None
            self.seq_scale_ = None
            return

        var = m2 / count
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        degenerate = var <= _SEQUENCE_VARIANCE_EPS
        scale = torch.sqrt(torch.clamp(var, min=0.0))
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        scale[degenerate] = 1.0
        self.seq_mean_ = mean.reshape(1, 1, -1).to(
            device=self.device, dtype=torch.float32
        )
        self.seq_scale_ = scale.reshape(1, 1, -1).to(
            device=self.device, dtype=torch.float32
        )

    def _fit_sequence_scaler(self, dataloader, batch_kind: str) -> None:
        """Fit sequence embedding normalization for global attention tasks."""

        def _tensor_fn(batch, device):
            seq_t, mask_t, _ = batch_to_attention_tensors(batch, batch_kind, device)
            return seq_t, mask_t

        self._fit_sequence_scaler_from_batches(dataloader, _tensor_fn)

    def _normalize_sequence(self, seq_t: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, "seq_mean_", None)
        scale = getattr(self, "seq_scale_", None)
        if mean is None or scale is None:
            return seq_t
        return (seq_t.float() - mean) / scale

    def _normalize_extra(self, extra_t: torch.Tensor | None) -> torch.Tensor | None:
        if extra_t is None or self.extra_mean_ is None or self.extra_scale_ is None:
            return extra_t
        return (extra_t.float() - self.extra_mean_) / self.extra_scale_

    def _maybe_fit_pooled_scalers(self, dataloader, batch_kind: str) -> None:
        """Fit sequence (+ extra) scalers once, skipping when already known.

        Called *after* ``maybe_resume_torch_wrapper``: a resumed checkpoint
        already carries ``seq_mean_``/``seq_scale_`` (and extra scalers), so the
        expensive full-dataset scaler passes are not re-paid on every restart.
        The extra-feature pass is skipped entirely when the task has no extra
        (ligand) features, which is a full no-op pass for protein-only tasks.
        """
        if self.seq_mean_ is not None:
            return
        self._fit_sequence_scaler(dataloader, batch_kind)
        if self.extra_dim_ > 0:
            self._fit_extra_scaler(dataloader, batch_kind)

    def _run_early_stopping(self, train_epoch, val_loss, val_dataloader, start_epoch, checkpoint_path):
        """Shared early-stopping loop plumbing used by every attention wrapper."""
        try:
            self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
                self.model,
                self.epochs,
                train_epoch,
                val_loss_fn=val_loss if val_dataloader is not None else None,
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                start_epoch=start_epoch,
                initial_best_state=self.best_model_state_,
                initial_best_epoch=self.best_epoch_,
                initial_best_val_loss=self.best_val_loss_,
                epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
            )
        finally:
            self._flush_batch_diagnostics()
        return self

    def build_checkpoint(self) -> dict[str, Any]:
        """Serialize the trained wrapper and pooled model weights."""
        if self.model is None:
            raise ValueError("Model must be fit before building a checkpoint.")
        return {
            "wrapper_class": self.__class__.__name__,
            "hidden_dims": list(self.hidden_dims),
            "lr": self.lr,
            "epochs": self.epochs,
            "optimizer_name": self.optimizer_name,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout_rate": self.attention_dropout_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "gradient_clip_norm": self.gradient_clip_norm,
            "pre_attention_layer_norm": self.pre_attention_layer_norm,
            "record_batch_diagnostics": self.record_batch_diagnostics,
            "diagnostic_loss_threshold": self.diagnostic_loss_threshold,
            "diagnostic_z_threshold": self.diagnostic_z_threshold,
            "diagnostic_grad_norm_threshold": self.diagnostic_grad_norm_threshold,
            "first_problematic_batch": self.first_problematic_batch_,
            "input_dim": self.input_dim_,
            "extra_dim": self.extra_dim_,
            "batch_kind": self.batch_kind_,
            "best_epoch": self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
            "seq_mean": self.seq_mean_.detach().cpu() if self.seq_mean_ is not None else None,
            "seq_scale": self.seq_scale_.detach().cpu() if self.seq_scale_ is not None else None,
            "extra_mean": self.extra_mean_.detach().cpu() if self.extra_mean_ is not None else None,
            "extra_scale": self.extra_scale_.detach().cpu() if self.extra_scale_ is not None else None,
            "model_state_dict": self.model.state_dict(),
            "pooler_state_dict": self.model.pooler.state_dict(),
        }


class _PooledAttentionWrapper(_BaseAttentionWrapper):
    """Shared fit/predict for single-output attention-pooled global models.

    Subclasses only supply the loss (`_make_criterion`) and how logits map to a
    prediction row (`_activate_predictions`); regressor vs binary classifier are
    otherwise identical.
    """

    def _make_criterion(self):
        raise NotImplementedError

    def _activate_predictions(self, logits: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self.model = AttentionPredictor(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            extra_dim=extra_dim,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
            pre_attention_layer_norm=self.pre_attention_layer_norm,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        self._maybe_fit_pooled_scalers(dataloader, batch_kind)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = self._make_criterion()

        def forward_logits(normalized_seq_t, mask_t, normalized_extra_t):
            with attention_autocast(self.device):
                return self.model(
                    normalized_seq_t, mask_t, normalized_extra_t
                ).float()

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            for batch_index, batch in enumerate(dataloader, start=1):
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                max_abs_z = (
                    self._max_abs_valid_z(seq_t, mask_t)
                    if self.record_batch_diagnostics
                    else 0.0
                )
                optimizer.zero_grad()
                loss = criterion(forward_logits(seq_t, mask_t, extra_t), y_t)
                loss.backward()
                grad_norm = self._prepare_gradients()
                self._record_batch_diagnostic(
                    epoch=diagnostic_epoch,
                    batch_index=batch_index,
                    batch=batch,
                    loss=float(loss.detach().item()),
                    max_abs_z=max_abs_z,
                    grad_norm=grad_norm,
                )
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                    seq_t = self._normalize_sequence(seq_t)
                    extra_t = self._normalize_extra(extra_t)
                    losses.append(float(criterion(forward_logits(seq_t, mask_t, extra_t), y_t).item()))
            return mean_losses(losses)

        return self._run_early_stopping(train_epoch, val_loss, val_dataloader, start_epoch, checkpoint_path)

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                with attention_autocast(self.device):
                    logits = self.model(seq_t, mask_t, extra_t).float()
                preds.append(self._activate_predictions(logits))
        return np.concatenate(preds).astype(np.float32, copy=False)


class AttentionRegressorTorch(_PooledAttentionWrapper):
    """Attention-pooled regressor for global tasks."""

    def _make_criterion(self):
        return nn.MSELoss()

    def _activate_predictions(self, logits: torch.Tensor) -> np.ndarray:
        return logits.squeeze(1).cpu().numpy()


class AttentionBinaryClassifierTorch(_PooledAttentionWrapper):
    """Attention-pooled binary classifier for global tasks."""

    def _make_criterion(self):
        return nn.BCEWithLogitsLoss()

    def _activate_predictions(self, logits: torch.Tensor) -> np.ndarray:
        return (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(np.float32)


class AttentionECMultiHeadClassifierTorch(_BaseAttentionWrapper):
    """Attention-pooled multi-head classifier for EC prediction."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (2048,),
        lr: float = 1e-3,
        epochs: int = 100,
        optimizer_name: str = "adamw",
        batch_size: int = 256,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        gradient_clip_norm: float | None = None,
        pre_attention_layer_norm: bool = False,
        record_batch_diagnostics: bool = False,
        diagnostic_loss_threshold: float | None = 100.0,
        diagnostic_z_threshold: float | None = 20.0,
        diagnostic_grad_norm_threshold: float | None = 1000.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            lr=lr,
            epochs=epochs,
            optimizer_name=optimizer_name,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            gradient_clip_norm=gradient_clip_norm,
            pre_attention_layer_norm=pre_attention_layer_norm,
            record_batch_diagnostics=record_batch_diagnostics,
            diagnostic_loss_threshold=diagnostic_loss_threshold,
            diagnostic_z_threshold=diagnostic_z_threshold,
            diagnostic_grad_norm_threshold=diagnostic_grad_norm_threshold,
            device=device,
        )
        self.batch_size = batch_size
        self.output_dims_ = None

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        first_batch = next(iter(dataloader))
        output_dims = {
            level_name: int(np.asarray(target, dtype=np.float32).shape[1])
            for level_name, target in first_batch.target.items()
        }
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self.output_dims_ = dict(output_dims)
        self.model = AttentionMultiHeadPredictor(
            input_dim,
            self.hidden_dims,
            output_dims,
            extra_dim=extra_dim,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
            pre_attention_layer_norm=self.pre_attention_layer_norm,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        self._maybe_fit_pooled_scalers(dataloader, batch_kind)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def forward_logits(normalized_seq_t, mask_t, normalized_extra_t):
            with attention_autocast(self.device):
                return self.model(normalized_seq_t, mask_t, normalized_extra_t)

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            for batch_index, batch in enumerate(dataloader, start=1):
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = {
                    level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(self.device)
                    for level_name, target in batch.target.items()
                }
                target_mask_t = {
                    level_name: torch.from_numpy(np.asarray(mask_arr, dtype=bool)).to(self.device)
                    for level_name, mask_arr in batch.target_mask.items()
                }
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                max_abs_z = (
                    self._max_abs_valid_z(seq_t, mask_t)
                    if self.record_batch_diagnostics
                    else 0.0
                )
                optimizer.zero_grad()
                logits_by_level = forward_logits(seq_t, mask_t, extra_t)
                loss = compute_ec_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                if loss is None:
                    continue
                loss.backward()
                grad_norm = self._prepare_gradients()
                self._record_batch_diagnostic(
                    epoch=diagnostic_epoch,
                    batch_index=batch_index,
                    batch=batch,
                    loss=float(loss.detach().item()),
                    max_abs_z=max_abs_z,
                    grad_norm=grad_norm,
                )
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = {
                        level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(self.device)
                        for level_name, target in batch.target.items()
                    }
                    target_mask_t = {
                        level_name: torch.from_numpy(np.asarray(mask_arr, dtype=bool)).to(self.device)
                        for level_name, mask_arr in batch.target_mask.items()
                    }
                    seq_t = self._normalize_sequence(seq_t)
                    extra_t = self._normalize_extra(extra_t)
                    logits_by_level = forward_logits(seq_t, mask_t, extra_t)
                    loss = compute_ec_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                    if loss is not None:
                        losses.append(float(loss.item()))
            return mean_losses(losses)

        return self._run_early_stopping(train_epoch, val_loss, val_dataloader, start_epoch, checkpoint_path)

    def predict_loader(self, dataloader, batch_kind: str) -> dict[str, np.ndarray]:
        if self.model is None or self.output_dims_ is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        preds_by_level = {level_name: [] for level_name in self.output_dims_}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                with attention_autocast(self.device):
                    logits_by_level = self.model(seq_t, mask_t, extra_t)
                for level_name, logits in logits_by_level.items():
                    preds_by_level[level_name].append(
                        (torch.sigmoid(logits.float()) >= 0.5).cpu().numpy().astype(np.float32)
                    )
        return {
            level_name: np.concatenate(chunks, axis=0)
            if chunks else np.zeros((0, self.output_dims_[level_name]), dtype=np.float32)
            for level_name, chunks in preds_by_level.items()
        }

    def build_checkpoint(self) -> dict[str, Any]:
        checkpoint = super().build_checkpoint()
        checkpoint["output_dims"] = dict(self.output_dims_ or {})
        checkpoint["batch_size"] = self.batch_size
        return checkpoint


class AttentionResidueBinaryClassifierTorch(_BaseAttentionWrapper):
    """Per-residue binary classifier with self-attention over residues (no pooling)."""

    @staticmethod
    def _batch_to_input(batch, device: torch.device):
        X_t = _seq_emb_to_tensor(batch.protein_emb, device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        y_t = torch.from_numpy(np.asarray(batch.target_per_residue, dtype=np.float32)).to(device)
        target_mask_t = torch.from_numpy(np.asarray(batch.target_mask, dtype=bool)).to(device)
        return X_t, mask_t, y_t, target_mask_t

    def _fit_residue_sequence_scaler(self, dataloader) -> None:
        """Fit sequence embedding normalization for per-residue attention tasks."""

        def _tensor_fn(batch, device):
            X_t, mask_t, _, _ = self._batch_to_input(batch, device)
            return X_t, mask_t

        self._fit_sequence_scaler_from_batches(dataloader, _tensor_fn)

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        if batch_kind != "protein_residue":
            raise ValueError(
                f"AttentionResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'"
            )
        first_batch = next(iter(dataloader))
        X_t, _, _, _ = self._batch_to_input(first_batch, self.device)
        input_dim = int(X_t.shape[2])
        self.input_dim_ = input_dim
        self.extra_dim_ = 0
        self.batch_kind_ = batch_kind
        self.model = AttentionResiduePredictor(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        if self.seq_mean_ is None:
            self._fit_residue_sequence_scaler(dataloader)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits.float(), y_t.float())
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def forward_logits(normalized_X_t, mask_t):
            with attention_autocast(self.device):
                return self.model(normalized_X_t, mask_t).squeeze(-1)

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            for batch_index, batch in enumerate(dataloader, start=1):
                X_t, mask_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                X_t = self._normalize_sequence(X_t)
                max_abs_z = (
                    self._max_abs_valid_z(X_t, mask_t)
                    if self.record_batch_diagnostics
                    else 0.0
                )
                logits = forward_logits(X_t, mask_t)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._prepare_gradients()
                self._record_batch_diagnostic(
                    epoch=diagnostic_epoch,
                    batch_index=batch_index,
                    batch=batch,
                    loss=float(loss.detach().item()),
                    max_abs_z=max_abs_z,
                    grad_norm=grad_norm,
                )
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                    X_t = self._normalize_sequence(X_t)
                    logits = forward_logits(X_t, mask_t)
                    losses.append(float(compute_batch_loss(logits, y_t, target_mask_t).item()))
            return mean_losses(losses)

        return self._run_early_stopping(train_epoch, val_loss, val_dataloader, start_epoch, checkpoint_path)

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(
                f"AttentionResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'"
            )
        if self.model is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        pred_rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_t, mask_t, _, target_mask_t = self._batch_to_input(batch, self.device)
                X_t = self._normalize_sequence(X_t)
                with attention_autocast(self.device):
                    logits = self.model(X_t, mask_t).squeeze(-1).float()
                batch_pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
                batch_mask = target_mask_t.cpu().numpy().astype(bool)
                for pred_row, mask_row in zip(batch_pred, batch_mask):
                    pred_rows.append(pred_row[mask_row])
        return pad_1d_array_list(pred_rows, dtype=np.float32)

    def build_checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model must be fit before building a checkpoint.")
        return {
            "wrapper_class": self.__class__.__name__,
            "hidden_dims": list(self.hidden_dims),
            "lr": self.lr,
            "epochs": self.epochs,
            "optimizer_name": self.optimizer_name,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout_rate": self.attention_dropout_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "gradient_clip_norm": self.gradient_clip_norm,
            "pre_attention_layer_norm": self.pre_attention_layer_norm,
            "record_batch_diagnostics": self.record_batch_diagnostics,
            "diagnostic_loss_threshold": self.diagnostic_loss_threshold,
            "diagnostic_z_threshold": self.diagnostic_z_threshold,
            "diagnostic_grad_norm_threshold": self.diagnostic_grad_norm_threshold,
            "first_problematic_batch": self.first_problematic_batch_,
            "input_dim": self.input_dim_,
            "extra_dim": self.extra_dim_,
            "batch_kind": self.batch_kind_,
            "best_epoch": self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
            "seq_mean": self.seq_mean_.detach().cpu() if self.seq_mean_ is not None else None,
            "seq_scale": self.seq_scale_.detach().cpu() if self.seq_scale_ is not None else None,
            "model_state_dict": self.model.state_dict(),
            "mixer_state_dict": self.model.mixer.state_dict(),
        }


def save_attention_checkpoint(wrapper, run_dir: Path, *, checkpoint_path: Path | None = None) -> Path:
    """Save the trained attention wrapper checkpoint to disk."""
    path = checkpoint_path or (run_dir / "model.pt")
    torch.save(wrapper.build_checkpoint(), path)
    return path


__all__ = [
    "AttentionRegressorTorch",
    "AttentionBinaryClassifierTorch",
    "AttentionECMultiHeadClassifierTorch",
    "AttentionResidueBinaryClassifierTorch",
    "save_attention_checkpoint",
]
