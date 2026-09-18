"""Per-residue binary classification with self-attention."""

# Keep established tensor names such as x_t.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from enzymado.models.attention_blocks import (
    AttentionResiduePredictor,
    _seq_emb_to_tensor,
    attention_autocast,
)
from enzymado.models.common import (
    make_optimizer,
    maybe_resume_torch_wrapper,
    mean_losses,
    pad_1d_array_list,
    restore_torch_training_state,
)
import enzymado.models.training_protocol
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


class AttentionResidueBinaryClassifierImplementation(_BaseAttentionWrapper):
    """Per-residue binary classifier with self-attention over residues (no pooling)."""

    @staticmethod
    def _batch_to_input(batch, device: torch.device):
        x_t = _seq_emb_to_tensor(batch.protein_emb, device)
        mask_t = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        y_t = torch.from_numpy(
            np.asarray(batch.target_per_residue, dtype=np.float32)
        ).to(device)
        target_mask_t = torch.from_numpy(np.asarray(batch.target_mask, dtype=bool)).to(
            device
        )
        return x_t, mask_t, y_t, target_mask_t

    def _fit_residue_sequence_scaler(self, dataloader) -> None:
        """Fit sequence embedding normalization for per-residue attention tasks."""

        def _tensor_fn(batch, device):
            x_t, mask_t, _, _ = self._batch_to_input(batch, device)
            return x_t, mask_t

        self._fit_sequence_scaler_from_batches(dataloader, _tensor_fn)

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
        if batch_kind != "protein_residue":
            raise ValueError(
                "AttentionResidueBinaryClassifierTorch does not support "
                f"batch_kind='{batch_kind}'"
            )
        first_batch = next(iter(dataloader))
        x_t, _, _, _ = self._batch_to_input(first_batch, self.device)
        input_dim = int(x_t.shape[2])
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
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        self._configure_batch_diagnostics(checkpoint_path, start_epoch)
        if self.seq_mean_ is None:
            self._fit_residue_sequence_scaler(dataloader)
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits.float(), y_t.float())
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def forward_logits(normalized_x_t, mask_t):
            with attention_autocast(self.device):
                return self.model(normalized_x_t, mask_t).squeeze(-1)

        def train_epoch():
            self.model.train()
            diagnostic_epoch = self._next_diagnostic_epoch()
            for batch_index, batch in enumerate(dataloader, start=1):
                x_t, mask_t, y_t, target_mask_t = self._batch_to_input(
                    batch, self.device
                )
                x_t = self._normalize_sequence(x_t)
                max_abs_z = (
                    self._max_abs_valid_z(x_t, mask_t)
                    if self.record_batch_diagnostics
                    else 0.0
                )
                logits = forward_logits(x_t, mask_t)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                enzymado.models.training_protocol.backward_checked(loss)
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
                    x_t, mask_t, y_t, target_mask_t = self._batch_to_input(
                        batch, self.device
                    )
                    x_t = self._normalize_sequence(x_t)
                    logits = forward_logits(x_t, mask_t)
                    losses.append(
                        float(compute_batch_loss(logits, y_t, target_mask_t).item())
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
        if batch_kind != "protein_residue":
            raise ValueError(
                "AttentionResidueBinaryClassifierTorch does not support "
                f"batch_kind='{batch_kind}'"
            )
        if self.model is None:
            raise ValueError("Model must be fit before calling predict_loader().")
        pred_rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x_t, mask_t, _, target_mask_t = self._batch_to_input(batch, self.device)
                x_t = self._normalize_sequence(x_t)
                with attention_autocast(self.device):
                    logits = self.model(x_t, mask_t).squeeze(-1).float()
                scores = torch.sigmoid(logits)
                batch_pred = (
                    (scores if proba else (scores >= 0.5))
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                batch_mask = target_mask_t.cpu().numpy().astype(bool)
                for pred_row, mask_row in zip(batch_pred, batch_mask):
                    pred_rows.append(pred_row[mask_row])
        return pad_1d_array_list(pred_rows, dtype=np.float32)

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        return self._predict_loader(dataloader, batch_kind, proba=False)

    def predict_proba_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        """Return sigmoid probabilities instead of thresholded hard labels."""
        return self._predict_loader(dataloader, batch_kind, proba=True)


__all__ = ["AttentionResidueBinaryClassifierImplementation"]
