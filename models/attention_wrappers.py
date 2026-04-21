"""Training wrappers for attention-pooled global models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.attention_blocks import (
    AttentionAggregation,
    AttentionMultiHeadPredictor,
    AttentionPredictor,
    batch_to_attention_tensors,
)
from models.common import (
    _make_optimizer,
    _run_training_with_early_stopping,
    make_torch_checkpoint_callback,
    maybe_resume_torch_wrapper,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def _infer_dims(self, dataloader, batch_kind: str) -> tuple[int, int]:
        first_batch = next(iter(dataloader))
        seq_t, _, extra_t = batch_to_attention_tensors(first_batch, batch_kind, self.device)
        input_dim = int(seq_t.shape[2])
        extra_dim = int(extra_t.shape[1]) if extra_t is not None else 0
        return input_dim, extra_dim

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
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "input_dim": self.input_dim_,
            "extra_dim": self.extra_dim_,
            "batch_kind": self.batch_kind_,
            "best_epoch": self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
            "model_state_dict": self.model.state_dict(),
            "pooler_state_dict": self.model.pooler.state_dict(),
        }


class AttentionRegressorTorch(_BaseAttentionWrapper):
    """Attention-pooled regressor for global tasks."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(
            hidden_dims,
            lr,
            epochs,
            optimizer_name,
            num_attention_heads,
            attention_dropout_rate,
            early_stopping_patience,
            early_stopping_min_delta,
            device,
        )

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
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = _make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.MSELoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(seq_t, mask_t, extra_t), y_t)
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self.model(seq_t, mask_t, extra_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
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
                preds.append(self.model(seq_t, mask_t, extra_t).squeeze(1).cpu().numpy())
        return np.concatenate(preds).astype(np.float32, copy=False)


class AttentionBinaryClassifierTorch(_BaseAttentionWrapper):
    """Attention-pooled binary classifier for global tasks."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(
            hidden_dims,
            lr,
            epochs,
            optimizer_name,
            num_attention_heads,
            attention_dropout_rate,
            early_stopping_patience,
            early_stopping_min_delta,
            device,
        )

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
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = _make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.BCEWithLogitsLoss()

        def train_epoch():
            self.model.train()
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(seq_t, mask_t, extra_t), y_t)
                loss.backward()
                optimizer.step()

        def val_loss():
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                    y_t = torch.from_numpy(np.asarray(batch.target, dtype=np.float32)).unsqueeze(1).to(self.device)
                    losses.append(float(criterion(self.model(seq_t, mask_t, extra_t), y_t).item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
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
                logits = self.model(seq_t, mask_t, extra_t)
                preds.append((torch.sigmoid(logits).squeeze(1) >= 0.5).cpu().numpy().astype(np.float32))
        return np.concatenate(preds).astype(np.float32, copy=False)


class AttentionECMultiHeadClassifierTorch(_BaseAttentionWrapper):
    """Attention-pooled multi-head classifier for EC prediction."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        batch_size: int = 1024,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(
            hidden_dims,
            lr,
            epochs,
            optimizer_name,
            num_attention_heads,
            attention_dropout_rate,
            early_stopping_patience,
            early_stopping_min_delta,
            device,
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
        ).to(self.device)
        start_epoch = maybe_resume_torch_wrapper(self, checkpoint_path)
        optimizer = _make_optimizer(self.model.parameters(), self.optimizer_name, self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        def compute_batch_loss(logits_by_level, y_t, target_mask_t):
            loss_terms = []
            for level_name, logits in logits_by_level.items():
                level_loss = criterion(logits, y_t[level_name]).mean(dim=1)
                sample_mask = target_mask_t[level_name].float()
                denom = sample_mask.sum()
                if denom.item() == 0:
                    continue
                loss_terms.append((level_loss * sample_mask).sum() / denom)
            if not loss_terms:
                return None
            return torch.stack(loss_terms).mean()

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
                logits_by_level = self.model(seq_t, mask_t, extra_t)
                loss = compute_batch_loss(logits_by_level, y_t, target_mask_t)
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
                    logits_by_level = self.model(seq_t, mask_t, extra_t)
                    loss = compute_batch_loss(logits_by_level, y_t, target_mask_t)
                    if loss is not None:
                        losses.append(float(loss.item()))
            return sum(losses) / max(len(losses), 1)

        self.best_epoch_, self.best_val_loss_ = _run_training_with_early_stopping(
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
                logits_by_level = self.model(seq_t, mask_t, extra_t)
                for level_name, logits in logits_by_level.items():
                    preds_by_level[level_name].append(
                        (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.float32)
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


def save_attention_checkpoint(wrapper, run_dir: Path, *, checkpoint_path: Path | None = None) -> Path:
    """Save the trained attention wrapper checkpoint to disk."""
    path = checkpoint_path or (run_dir / "model.pt")
    torch.save(wrapper.build_checkpoint(), path)
    return path


__all__ = [
    "DEVICE",
    "AttentionRegressorTorch",
    "AttentionBinaryClassifierTorch",
    "AttentionECMultiHeadClassifierTorch",
    "save_attention_checkpoint",
]
