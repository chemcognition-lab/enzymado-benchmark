"""Torch wrappers for structured prediction tasks."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from models.common import (
    DEVICE,
    make_optimizer,
    make_torch_checkpoint_callback,
    maybe_resume_torch_wrapper,
    pad_1d_array_list,
    run_training_with_early_stopping,
)
from models.nn_blocks import MLPModel, MultiHeadMLPModel


class ECMultiHeadClassifierTorch:
    """Shared-backbone classifier for hierarchical EC targets on pooled inputs."""

    def __init__(
        self,
        hidden_dims=(2048,),
        lr=1e-3,
        epochs=100,
        optimizer_name="adamw",
        weight_decay=0.0,
        dropout=0.0,
        batch_size=256,
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
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.output_dims = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.current_epoch_ = 0
        self.best_model_state_ = None

    @staticmethod
    def _targets_to_tensors(y, target_mask, device: torch.device):
        y_t = {
            level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)
            for level_name, target in y.items()
        }
        target_mask_t = {
            level_name: torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
            for level_name, mask in target_mask.items()
        }
        return y_t, target_mask_t

    @staticmethod
    def _compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion):
        loss_terms = []
        for level_name, logits in logits_by_level.items():
            level_loss = criterion(logits, y_t[level_name])
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

    def _forward(self, X_t: torch.Tensor):
        if X_t.ndim != 2:
            raise ValueError(f"ECMultiHeadClassifierTorch expects pooled 2D inputs, got shape {tuple(X_t.shape)}")
        return self.model(X_t)

    def fit(self, X, y, target_mask, X_val=None, y_val=None, target_mask_val=None, checkpoint_path=None):
        if target_mask is None:
            raise ValueError("ECMultiHeadClassifierTorch.fit requires target_mask.")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        input_dim = X_np.shape[1]

        self.output_dims = {
            level_name: int(np.asarray(target, dtype=np.float32).shape[1])
            for level_name, target in y.items()
        }
        self.model = MultiHeadMLPModel(input_dim, self.hidden_dims, self.output_dims, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        val_payload = None
        if X_val is not None and y_val is not None and target_mask_val is not None:
            val_payload = (
                np.asarray(X_val, dtype=np.float32),
                {level_name: np.asarray(target, dtype=np.float32) for level_name, target in y_val.items()},
                {level_name: np.asarray(mask_arr, dtype=bool) for level_name, mask_arr in target_mask_val.items()},
            )

        def iter_minibatches(X_arr, y_arr, target_mask_arr):
            for start in range(0, X_arr.shape[0], self.batch_size):
                stop = min(start + self.batch_size, X_arr.shape[0])
                X_t = torch.from_numpy(X_arr[start:stop]).to(self.device)
                y_t, target_mask_t = self._targets_to_tensors(
                    {level_name: target[start:stop] for level_name, target in y_arr.items()},
                    {level_name: mask_values[start:stop] for level_name, mask_values in target_mask_arr.items()},
                    self.device,
                )
                yield X_t, y_t, target_mask_t

        def train_epoch():
            self.model.train()
            for X_t, y_t, target_mask_t in iter_minibatches(X_np, y, target_mask):
                logits_by_level = self._forward(X_t)
                loss = self._compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                X_val_np, y_val_np, target_mask_val_np = val_payload
                for X_t, y_t, target_mask_t in iter_minibatches(X_val_np, y_val_np, target_mask_val_np):
                    logits_by_level = self._forward(X_t)
                    loss = self._compute_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                    if loss is not None:
                        losses.append(float(loss.item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = run_training_with_early_stopping(
            self.model,
            self.epochs,
            train_epoch,
            val_loss_fn=val_loss if val_payload is not None else None,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            start_epoch=start_epoch,
            initial_best_state=self.best_model_state_,
            initial_best_epoch=self.best_epoch_,
            initial_best_val_loss=self.best_val_loss_,
            epoch_callback=make_torch_checkpoint_callback(self, checkpoint_path),
        )
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        if self.output_dims is None:
            raise ValueError("ECMultiHeadClassifierTorch must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        preds_by_level = {level_name: [] for level_name in self.output_dims}

        self.model.eval()
        with torch.no_grad():
            for start in range(0, X_np.shape[0], self.batch_size):
                stop = min(start + self.batch_size, X_np.shape[0])
                X_t = torch.from_numpy(X_np[start:stop]).to(self.device)
                logits_by_level = self._forward(X_t)
                for level_name, logits in logits_by_level.items():
                    preds_by_level[level_name].append(
                        (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
                    )

        return {
            level_name: np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, self.output_dims[level_name]), dtype=np.float32)
            for level_name, chunks in preds_by_level.items()
        }


class ResidueBinaryClassifierTorch:
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
        X_t = cls._protein_emb_to_tensor(batch.protein_emb, device)
        y_t = torch.from_numpy(np.asarray(batch.target_per_residue, dtype=np.float32)).to(device)
        target_mask_t = torch.from_numpy(np.asarray(batch.target_mask, dtype=bool)).to(device)
        return X_t, y_t, target_mask_t

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        if batch_kind != "protein_residue":
            raise ValueError(f"ResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'")

        first_batch = next(iter(dataloader))
        X_t, _, _ = self._batch_to_input(first_batch, self.device)
        input_dim = X_t.shape[2]
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim=1, dropout=self.dropout).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits.float(), y_t.float())
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def forward_batch(X_t):
            with self._autocast_context(self.device):
                return self.model(X_t).squeeze(-1)

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                logits = forward_batch(X_t)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                    logits = forward_batch(X_t)
                    loss = compute_batch_loss(logits, y_t, target_mask_t)
                    losses.append(float(loss.item()))
            return sum(losses) / max(len(losses), 1)

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
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(f"ResidueBinaryClassifierTorch does not support batch_kind='{batch_kind}'")

        pred_rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_t, _, target_mask_t = self._batch_to_input(batch, self.device)
                with self._autocast_context(self.device):
                    logits = self.model(X_t).squeeze(-1)
                batch_pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
                batch_mask = target_mask_t.cpu().numpy().astype(bool)
                for pred_row, mask_row in zip(batch_pred, batch_mask):
                    pred_rows.append(pred_row[mask_row])
        return pad_1d_array_list(pred_rows, dtype=np.float32)


__all__ = [
    "ECMultiHeadClassifierTorch",
    "ResidueBinaryClassifierTorch",
]
