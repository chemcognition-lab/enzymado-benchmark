"""Normalization, diagnostics, and training state shared by attention wrappers."""

# Preserve positional boolean arguments in the established wrapper API.
# ruff: noqa: FBT001, FBT002

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from enzymado.models.attention_blocks import (
    MeanAggregation,
    batch_to_attention_tensors,
)
from enzymado.models.common import (
    DEVICE,
    make_torch_checkpoint_callback,
    make_validation_callback,
    run_training_with_early_stopping,
    score_validation_tensors,
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

    # Same rationale as BaseStandardWrapper: pooled attention tasks select on
    # the metric the leaderboard reports; the EC and residue attention routes
    # override this back to None and keep selecting on loss.
    validation_metric: str | None = None

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
        record_batch_diagnostics: bool = False,
        diagnostic_loss_threshold: float | None = 100.0,
        diagnostic_z_threshold: float | None = 20.0,
        diagnostic_grad_norm_threshold: float | None = 1000.0,
        device: str | torch.device | None = None,
        gradient_accumulation_steps: int = 1,
        attention_block_lr: float | None = None,
    ):
        if int(gradient_accumulation_steps) < 0:
            raise ValueError(
                "gradient_accumulation_steps must be >= 0 (0 = whole epoch)."
            )
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
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if attention_block_lr is not None and attention_block_lr <= 0:
            raise ValueError("attention_block_lr must be positive when provided.")
        self.attention_block_lr = attention_block_lr
        self.record_batch_diagnostics = bool(record_batch_diagnostics)
        self.diagnostic_loss_threshold = diagnostic_loss_threshold
        self.diagnostic_z_threshold = diagnostic_z_threshold
        self.diagnostic_grad_norm_threshold = diagnostic_grad_norm_threshold
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.validation_metric_ = self.validation_metric
        self.input_dim_ = None
        self.extra_dim_ = 0
        self.batch_kind_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None
        self.seq_mean_ = None
        self.seq_scale_ = None
        self.extra_mean_ = None
        self.extra_scale_ = None
        self.pooled_standardizer_fitted_ = False
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
        elif (
            self.diagnostic_loss_threshold is not None
            and loss >= self.diagnostic_loss_threshold
        ):
            reasons.append("loss_threshold")
        if not np.isfinite(max_abs_z):
            reasons.append("non_finite_z")
        elif (
            self.diagnostic_z_threshold is not None
            and max_abs_z >= self.diagnostic_z_threshold
        ):
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
            "sample_ids": ";".join(
                str(value) for value in getattr(batch, "sample_id", [])
            ),
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
        seq_t, _, extra_t = batch_to_attention_tensors(
            first_batch, batch_kind, self.device
        )
        input_dim = int(seq_t.shape[2])
        extra_dim = int(extra_t.shape[1]) if extra_t is not None else 0
        return input_dim, extra_dim

    def _fit_extra_scaler(self, dataloader, batch_kind: str) -> None:
        """Fit train-only standardization for ligand/extra features."""
        extras = []
        ligand_dim = 0
        for batch in dataloader:
            _, _, extra_t = batch_to_attention_tensors(
                batch, batch_kind, torch.device("cpu")
            )
            if extra_t is not None:
                extras.append(extra_t.float().cpu().numpy())
            if batch_kind == "protein_ligand_global":
                ligand_dim = np.asarray(batch.ligand_feat).shape[1]
        if not extras:
            self.extra_mean_ = None
            self.extra_scale_ = None
            return
        extra_np = np.concatenate(extras, axis=0).astype(np.float32, copy=False)
        mean = extra_np.mean(axis=0, keepdims=True)
        scale = extra_np.std(axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        # Ligand descriptors (leading columns) already lie in [0, 1]. A column
        # that is nearly constant in train would turn a test value into a huge
        # z-score, so they pass through; numeric context is still scaled.
        mean[:, :ligand_dim] = 0.0
        scale[:, :ligand_dim] = 1.0
        self.extra_mean_ = torch.from_numpy(mean.astype(np.float32, copy=False)).to(
            self.device
        )
        self.extra_scale_ = torch.from_numpy(scale.astype(np.float32, copy=False)).to(
            self.device
        )

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
            m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total_count)
            count = total_count

        if count == 0 or mean is None or m2 is None:
            self.seq_mean_ = None
            self.seq_scale_ = None
            return

        var = m2 / count
        scale = torch.sqrt(torch.clamp(var, min=0.0))
        scale[var <= _SEQUENCE_VARIANCE_EPS] = 1.0
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
        if self.seq_mean_ is None or self.seq_scale_ is None:
            return seq_t
        return (seq_t.float() - self.seq_mean_) / self.seq_scale_

    def _normalize_extra(self, extra_t: torch.Tensor | None) -> torch.Tensor | None:
        if extra_t is None or self.extra_mean_ is None or self.extra_scale_ is None:
            return extra_t
        return (extra_t.float() - self.extra_mean_) / self.extra_scale_

    def _optimizer_parameters(self):
        """Parameters for the optimizer, with the attention block at its own lr.

        ``attention_block_lr`` lets a head keep its mean-pool learning rate while
        the attention block, which has no mean-pool counterpart, trains at a rate
        it tolerates. Without it every parameter uses ``lr``.
        """
        if self.attention_block_lr is None:
            return self.model.parameters()
        block = list(self.model.pooler.parameters())
        block_ids = {id(param) for param in block}
        rest = [p for p in self.model.parameters() if id(p) not in block_ids]
        return [
            {"params": block, "lr": float(self.attention_block_lr)},
            {"params": rest},
        ]

    def _scale_gradients(self, factor: float) -> None:
        """Rescale accumulated gradients in place before an optimizer step."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(factor)

    def _fit_pooled_standardizer(self, dataloader, batch_kind: str) -> None:
        """Set the model's pooled-vector statistics from the train split.

        Uses the mean-pooled normalized sequence per protein, which is a per-
        feature affine map of the raw mean-pooled embedding, so standardizing it
        reproduces exactly the input scale of the published mean-pool route.
        Skipped when the model already carries fitted statistics (resume).
        """
        if self.pooled_standardizer_fitted_:
            return
        model = self.model
        mean_agg = MeanAggregation()
        total = None
        total_sq = None
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, _ = batch_to_attention_tensors(
                    batch, batch_kind, self.device
                )
                pooled = mean_agg(self._normalize_sequence(seq_t), ~mask_t).double()
                total = pooled.sum(0) if total is None else total + pooled.sum(0)
                sq = pooled.pow(2).sum(0)
                total_sq = sq if total_sq is None else total_sq + sq
                count += int(pooled.shape[0])
        if not count:
            raise ValueError("Cannot fit the pooled standardizer on an empty split.")
        mean = total / count
        var = (total_sq / count - mean.pow(2)).clamp(min=0.0)
        scale = var.sqrt()
        scale[scale <= 1e-6] = 1.0
        model.pooled_mean.copy_(mean.float())
        model.pooled_scale.copy_(scale.float())
        self.pooled_standardizer_fitted_ = True

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

    def _selection_objective(self, logits_t, y_t):
        """Return the negated selection metric for concatenated validation output."""
        return score_validation_tensors(logits_t, y_t, self.validation_metric)

    def _run_early_stopping(
        self,
        train_epoch,
        val_loss,
        val_dataloader,
        start_epoch,
        checkpoint_path,
        optimizer,
        dataloader,
    ):
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
                validation_callback=make_validation_callback(self, checkpoint_path),
                epoch_callback=make_torch_checkpoint_callback(
                    self,
                    checkpoint_path,
                    optimizer=optimizer,
                    dataloader=dataloader,
                ),
            )
        finally:
            self._flush_batch_diagnostics()
        return self


__all__ = ["_BaseAttentionWrapper"]
