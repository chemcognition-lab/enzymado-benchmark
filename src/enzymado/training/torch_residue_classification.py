"""Per-residue binary classification with PyTorch embeddings."""

# Keep established tensor names such as x_t.

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from enzymado.models import training_protocol
from enzymado.models.common import (
    DEVICE,
    make_optimizer,
    maybe_resume_torch_wrapper,
    mean_losses,
    pad_1d_array_list,
    restore_torch_training_state,
    run_wrapper_training,
)
from enzymado.models.nn_blocks import MLPModel


class ResidueBinaryClassifierImplementation:
    """Binary classifier over per-residue embeddings with padding-aware BCE loss."""

    def __init__(
        self,
        hidden_dims=(),
        lr=1e-3,
        epochs=100,
        optimizer_name="adamw",
        weight_decay=0.0,
        dropout=0.0,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        device=None,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None

    @staticmethod
    def _protein_emb_to_tensor(protein_emb, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(protein_emb):
            return protein_emb.to(device)
        array = np.asarray(protein_emb)
        return torch.from_numpy(array.astype(np.float32, copy=False)).to(device)

    @staticmethod
    def _autocast_context(device: torch.device):
        if device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    @classmethod
    def _batch_to_input(cls, batch, device: torch.device):
        x_t = cls._protein_emb_to_tensor(batch.protein_emb, device)
        y_t = torch.from_numpy(
            np.asarray(batch.target_per_residue, dtype=np.float32)
        ).to(device)
        target_mask_t = torch.from_numpy(np.asarray(batch.target_mask, dtype=bool)).to(
            device
        )
        return x_t, y_t, target_mask_t

    def fit_loader(
        self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None
    ):
        if batch_kind != "protein_residue":
            raise ValueError(
                "ResidueBinaryClassifierTorch does not support "
                f"batch_kind='{batch_kind}'"
            )

        first_batch = next(iter(dataloader))
        x_t, _, _ = self._batch_to_input(first_batch, self.device)
        input_dim = x_t.shape[2]
        self.model = MLPModel(
            input_dim, self.hidden_dims, output_dim=1, dropout=self.dropout
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(
            self, checkpoint_path, dataloader=dataloader
        )
        optimizer = make_optimizer(
            self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay
        )
        restore_torch_training_state(self, optimizer)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits.float(), y_t.float())
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def forward_batch(x_t):
            with self._autocast_context(self.device):
                return self.model(x_t).squeeze(-1)

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                x_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                logits = forward_batch(x_t)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                training_protocol.backward_checked(loss)
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                    logits = forward_batch(x_t)
                    loss = compute_batch_loss(logits, y_t, target_mask_t)
                    losses.append(float(loss.item()))
            return mean_losses(losses)

        run_wrapper_training(
            self,
            train_epoch,
            val_loss if val_dataloader is not None else None,
            start_epoch,
            checkpoint_path,
            optimizer=optimizer,
            dataloader=dataloader,
        )
        return self

    def _predict_loader(
        self, dataloader, batch_kind: str, *, proba: bool
    ) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(
                "ResidueBinaryClassifierTorch does not support "
                f"batch_kind='{batch_kind}'"
            )

        pred_rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x_t, _, target_mask_t = self._batch_to_input(batch, self.device)
                with self._autocast_context(self.device):
                    logits = self.model(x_t).squeeze(-1)
                # The hard-label branch keeps the sigmoid at autocast precision
                # (bf16) so it reproduces the published predictions. The
                # probability branch promotes to float32: bf16 carries only 8
                # mantissa bits and would degrade the ranking AUROC/AUPRC measure.
                batch_pred = (
                    torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
                    if proba
                    else (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
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


__all__ = ["ResidueBinaryClassifierImplementation"]
