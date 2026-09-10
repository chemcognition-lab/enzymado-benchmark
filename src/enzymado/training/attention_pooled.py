"""Single-output attention adapters for global regression and classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from enzymado.models.attention_blocks import (
    AttentionPredictor,
    attention_autocast,
    batch_to_attention_tensors,
)
from enzymado.models.common import (
    make_optimizer,
    maybe_resume_torch_wrapper,
    mean_losses,
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

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
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
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        self._maybe_fit_pooled_scalers(dataloader, batch_kind)
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay
        )
        restore_torch_training_state(self, optimizer)
        criterion = self._make_criterion()

        def forward_logits(normalized_seq_t, mask_t, normalized_extra_t):
            with attention_autocast(self.device):
                return self.model(normalized_seq_t, mask_t, normalized_extra_t).float()

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            for batch_index, batch in enumerate(dataloader, start=1):
                seq_t, mask_t, extra_t = batch_to_attention_tensors(
                    batch, batch_kind, self.device
                )
                y_t = (
                    torch.from_numpy(np.asarray(batch.target, dtype=np.float32))
                    .unsqueeze(1)
                    .to(self.device)
                )
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
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(
                        batch, batch_kind, self.device
                    )
                    y_t = (
                        torch.from_numpy(np.asarray(batch.target, dtype=np.float32))
                        .unsqueeze(1)
                        .to(self.device)
                    )
                    seq_t = self._normalize_sequence(seq_t)
                    extra_t = self._normalize_extra(extra_t)
                    losses.append(
                        float(
                            criterion(
                                forward_logits(seq_t, mask_t, extra_t), y_t
                            ).item()
                        )
                    )
            return mean_losses(losses)

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
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        self.model.eval()
        activate = self._activate_probabilities if proba else self._activate_predictions
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(
                    batch, batch_kind, self.device
                )
                seq_t = self._normalize_sequence(seq_t)
                extra_t = self._normalize_extra(extra_t)
                with attention_autocast(self.device):
                    logits = self.model(seq_t, mask_t, extra_t).float()
                preds.append(activate(logits))
        return np.concatenate(preds).astype(np.float32, copy=False)

    def _activate_probabilities(self, logits: torch.Tensor) -> np.ndarray:
        """Default to the hard-label activation for non-classification wrappers."""
        return self._activate_predictions(logits)

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        return self._predict_loader(dataloader, batch_kind, proba=False)

    def predict_proba_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        """Return sigmoid probabilities instead of thresholded hard labels."""
        return self._predict_loader(dataloader, batch_kind, proba=True)


class AttentionRegressorImplementation(_PooledAttentionWrapper):
    """Attention-pooled regressor for global tasks."""

    def _make_criterion(self):
        return nn.MSELoss()

    def _activate_predictions(self, logits: torch.Tensor) -> np.ndarray:
        return logits.squeeze(1).cpu().numpy()


class AttentionBinaryClassifierImplementation(_PooledAttentionWrapper):
    """Attention-pooled binary classifier for global tasks."""

    def _make_criterion(self):
        return nn.BCEWithLogitsLoss()

    def _activate_predictions(self, logits: torch.Tensor) -> np.ndarray:
        return (
            (torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(np.float32)
        )

    def _activate_probabilities(self, logits: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)


__all__ = [
    "_PooledAttentionWrapper",
    "AttentionRegressorImplementation",
    "AttentionBinaryClassifierImplementation",
]
