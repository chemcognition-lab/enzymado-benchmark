"""Training wrappers for attention-pooled global models."""

from __future__ import annotations

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
        sum_t = None
        sumsq_t = None
        count = 0
        cpu = torch.device("cpu")
        for batch in dataloader:
            seq_t, mask_t = tensor_fn(batch, cpu)
            seq_t = seq_t.float()
            mask_t = mask_t.bool()
            observed = seq_t[mask_t]
            if observed.numel() == 0:
                continue
            batch_sum = observed.sum(dim=0)
            batch_sumsq = (observed * observed).sum(dim=0)
            sum_t = batch_sum if sum_t is None else sum_t + batch_sum
            sumsq_t = batch_sumsq if sumsq_t is None else sumsq_t + batch_sumsq
            count += int(observed.shape[0])

        if count == 0 or sum_t is None or sumsq_t is None:
            self.seq_mean_ = None
            self.seq_scale_ = None
            return

        mean = sum_t / float(count)
        var = (sumsq_t / float(count)) - (mean * mean)
        scale = torch.sqrt(torch.clamp(var, min=1e-12))
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        scale[scale == 0.0] = 1.0
        self.seq_mean_ = mean.reshape(1, 1, -1).to(self.device)
        self.seq_scale_ = scale.reshape(1, 1, -1).to(self.device)

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


class AttentionRegressorTorch(_BaseAttentionWrapper):
    """Attention-pooled regressor for global tasks."""

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self._fit_sequence_scaler(dataloader, batch_kind)
        self._fit_extra_scaler(dataloader, batch_kind)
        self.model = AttentionPredictor(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            extra_dim=extra_dim,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.MSELoss()

        def forward_logits(seq_t, mask_t, extra_t):
            seq_t = self._normalize_sequence(seq_t)
            extra_t = self._normalize_extra(extra_t)
            with attention_autocast(self.device):
                return self.model(seq_t, mask_t, extra_t).float()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(forward_logits(seq_t, mask_t, extra_t), y_t)
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(forward_logits(seq_t, mask_t, extra_t), y_t).item()))
            return mean_losses(losses)

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
                    out = self.model(seq_t, mask_t, extra_t).float()
                preds.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(preds).astype(np.float32, copy=False)


class AttentionBinaryClassifierTorch(_BaseAttentionWrapper):
    """Attention-pooled binary classifier for global tasks."""

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None, checkpoint_path=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self._fit_sequence_scaler(dataloader, batch_kind)
        self._fit_extra_scaler(dataloader, batch_kind)
        self.model = AttentionPredictor(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            extra_dim=extra_dim,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        def forward_logits(seq_t, mask_t, extra_t):
            seq_t = self._normalize_sequence(seq_t)
            extra_t = self._normalize_extra(extra_t)
            with attention_autocast(self.device):
                return self.model(seq_t, mask_t, extra_t).float()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(forward_logits(seq_t, mask_t, extra_t), y_t)
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(forward_logits(seq_t, mask_t, extra_t), y_t).item()))
            return mean_losses(losses)

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
                preds.append((torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(np.float32))
        return np.concatenate(preds).astype(np.float32, copy=False)


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
        self._fit_sequence_scaler(dataloader, batch_kind)
        self._fit_extra_scaler(dataloader, batch_kind)
        self.model = AttentionMultiHeadPredictor(
            input_dim,
            self.hidden_dims,
            output_dims,
            extra_dim=extra_dim,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def forward_logits(seq_t, mask_t, extra_t):
            seq_t = self._normalize_sequence(seq_t)
            extra_t = self._normalize_extra(extra_t)
            with attention_autocast(self.device):
                return self.model(seq_t, mask_t, extra_t)

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = {
                    level_name: torch.from_numpy(np.asarray(target, dtype=np.float32)).to(self.device)
                    for level_name, target in batch.target.items()
                }
                target_mask_t = {
                    level_name: torch.from_numpy(np.asarray(mask_arr, dtype=bool)).to(self.device)
                    for level_name, mask_arr in batch.target_mask.items()
                }
                logits_by_level = forward_logits(seq_t, mask_t, extra_t)
                loss = compute_ec_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
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
                    logits_by_level = forward_logits(seq_t, mask_t, extra_t)
                    loss = compute_ec_batch_loss(logits_by_level, y_t, target_mask_t, criterion)
                    if loss is not None:
                        losses.append(float(loss.item()))
            return mean_losses(losses)

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
        self._fit_residue_sequence_scaler(dataloader)
        self.model = AttentionResiduePredictor(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dropout=self.dropout,
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits, y_t, target_mask_t):
            loss = criterion(logits.float(), y_t.float())
            mask_f = target_mask_t.float()
            return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        def forward_logits(X_t, mask_t):
            X_t = self._normalize_sequence(X_t)
            with attention_autocast(self.device):
                return self.model(X_t, mask_t).squeeze(-1)

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                X_t, mask_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                logits = forward_logits(X_t, mask_t)
                loss = compute_batch_loss(logits, y_t, target_mask_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_t, mask_t, y_t, target_mask_t = self._batch_to_input(batch, self.device)
                    logits = forward_logits(X_t, mask_t)
                    losses.append(float(compute_batch_loss(logits, y_t, target_mask_t).item()))
            return mean_losses(losses)

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
