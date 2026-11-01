"""Hierarchical EC classification with attention-pooled embeddings."""

# Preserve positional boolean arguments in the established wrapper API.
# ruff: noqa: FBT001, FBT002

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from enzymado.models import training_protocol
from enzymado.models.attention_blocks import (
    AttentionMultiHeadPredictor,
    attention_autocast,
    batch_to_attention_tensors,
)
from enzymado.models.common import (
    StreamingAUPRC,
    compute_ec_batch_loss,
    make_optimizer,
    maybe_resume_torch_wrapper,
    restore_torch_training_state,
)
from enzymado.training.attention_wrapper_base import _BaseAttentionWrapper

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


class AttentionECMultiHeadClassifierImplementation(_BaseAttentionWrapper):
    """Attention-pooled multi-head classifier for EC prediction.

    Checkpoints are selected on validation AUPRC pooled over the four EC levels,
    like the mean-pool EC head.
    """

    validation_metric = "auprc"

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
        record_batch_diagnostics: bool = False,
        diagnostic_loss_threshold: float | None = 100.0,
        diagnostic_z_threshold: float | None = 20.0,
        diagnostic_grad_norm_threshold: float | None = 1000.0,
        device: str | torch.device | None = None,
        gradient_accumulation_steps: int = 1,
        attention_block_lr: float | None = None,
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
            record_batch_diagnostics=record_batch_diagnostics,
            diagnostic_loss_threshold=diagnostic_loss_threshold,
            diagnostic_z_threshold=diagnostic_z_threshold,
            diagnostic_grad_norm_threshold=diagnostic_grad_norm_threshold,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            attention_block_lr=attention_block_lr,
        )
        self.batch_size = batch_size
        self.output_dims_ = None

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
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
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        self._maybe_fit_pooled_scalers(dataloader, batch_kind)
        self._fit_pooled_standardizer(dataloader, batch_kind)
        optimizer = make_optimizer(
            self._optimizer_parameters(),
            self.optimizer_name,
            self.lr,
            self.weight_decay,
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def forward_logits(normalized_seq_t, mask_t, normalized_extra_t):
            with attention_autocast(self.device):
                return self.model(normalized_seq_t, mask_t, normalized_extra_t)

        # 1 steps once per loader batch (the published behaviour); k > 1 sums k
        # loader batches into one step and 0 the whole epoch. The EC loss is a
        # masked mean per level, so the mean of micro-batch losses equals the
        # full-batch loss when micro-batches carry the same number of valid labels
        # per level, and is a close approximation otherwise.
        accumulation = int(getattr(self, "gradient_accumulation_steps", 1))

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            pending = 0
            last = None
            optimizer.zero_grad()

            def step(batch_index, batch, loss, max_abs_z):
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
                optimizer.zero_grad()

            for batch_index, batch in enumerate(dataloader, start=1):
                seq_t, mask_t, extra_t = batch_to_attention_tensors(
                    batch, batch_kind, self.device
                )
                y_t = {
                    level_name: torch.from_numpy(
                        np.asarray(target, dtype=np.float32)
                    ).to(self.device)
                    for level_name, target in batch.target.items()
                }
                target_mask_t = {
                    level_name: torch.from_numpy(np.asarray(mask_arr, dtype=bool)).to(
                        self.device
                    )
                    for level_name, mask_arr in batch.target_mask.items()
                }
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                max_abs_z = (
                    self._max_abs_valid_z(seq_t, mask_t)
                    if self.record_batch_diagnostics
                    else 0.0
                )
                logits_by_level = forward_logits(seq_t, mask_t, extra_t)
                loss = compute_ec_batch_loss(
                    logits_by_level, y_t, target_mask_t, criterion
                )
                if loss is None:
                    continue
                if accumulation == 1:
                    training_protocol.backward_checked(loss)
                    step(batch_index, batch, loss, max_abs_z)
                    continue
                training_protocol.backward_checked(loss)
                pending += 1
                last = (batch_index, batch, loss, max_abs_z)
                if pending == accumulation:
                    self._scale_gradients(1.0 / pending)
                    step(*last)
                    pending = 0
            if pending:
                self._scale_gradients(1.0 / pending)
                step(*last)

        def val_loss():
            self.model.eval()
            auprc = StreamingAUPRC(self.device)
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(
                        batch, batch_kind, self.device
                    )
                    y_t = {
                        level_name: torch.from_numpy(
                            np.asarray(target, dtype=np.float32)
                        ).to(self.device)
                        for level_name, target in batch.target.items()
                    }
                    target_mask_t = {
                        level_name: torch.from_numpy(
                            np.asarray(mask_arr, dtype=bool)
                        ).to(self.device)
                        for level_name, mask_arr in batch.target_mask.items()
                    }
                    seq_t = self._normalize_sequence(seq_t)
                    extra_t = self._normalize_extra(extra_t)
                    for level_name, logits in forward_logits(
                        seq_t, mask_t, extra_t
                    ).items():
                        auprc.update(
                            torch.sigmoid(logits.float()),
                            y_t[level_name],
                            target_mask_t[level_name],
                        )
            return -auprc.value()

        return self._run_early_stopping(
            train_epoch,
            val_loss,
            val_dataloader,
            start_epoch,
            checkpoint_path,
            optimizer,
            dataloader,
        )

    def _predict_loader(
        self, dataloader, batch_kind: str, *, proba: bool
    ) -> dict[str, np.ndarray]:
        if self.model is None or self.output_dims_ is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        preds_by_level = {level_name: [] for level_name in self.output_dims_}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(
                    batch, batch_kind, self.device
                )
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                with attention_autocast(self.device):
                    logits_by_level = self.model(seq_t, mask_t, extra_t)
                for level_name, logits in logits_by_level.items():
                    scores = torch.sigmoid(logits.float())
                    preds_by_level[level_name].append(
                        (scores if proba else (scores >= 0.5))
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
        return {
            level_name: np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, self.output_dims_[level_name]), dtype=np.float32)
            for level_name, chunks in preds_by_level.items()
        }

    def predict_loader(self, dataloader, batch_kind: str) -> dict[str, np.ndarray]:
        return self._predict_loader(dataloader, batch_kind, proba=False)

    def predict_proba_loader(
        self, dataloader, batch_kind: str
    ) -> dict[str, np.ndarray]:
        """Return per-level sigmoid probabilities instead of hard labels."""
        return self._predict_loader(dataloader, batch_kind, proba=True)


__all__ = ["AttentionECMultiHeadClassifierImplementation"]
