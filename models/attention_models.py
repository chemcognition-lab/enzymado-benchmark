from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.models import (
    MLPModel,
    SharedMLPBackbone,
    _make_optimizer,
    _run_training_with_early_stopping,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=1)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=1)
    weights = weights * mask.float()
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return weights / denom


def batch_to_attention_tensors(batch, batch_kind: str, device: torch.device):
    if batch_kind == "protein_global":
        seq = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(device)
        mask = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        extra = None
    elif batch_kind == "protein_ligand_global":
        seq = torch.from_numpy(np.asarray(batch.protein_emb, dtype=np.float32)).to(device)
        mask = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        extra = torch.from_numpy(np.asarray(batch.ligand_feat, dtype=np.float32)).to(device)
    elif batch_kind == "mutation_delta_global":
        seq = torch.from_numpy(np.asarray(batch.protein_delta_emb, dtype=np.float32)).to(device)
        mask = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        extra = None
    else:
        raise ValueError(f"Unsupported batch_kind '{batch_kind}' for attention models.")
    return seq, mask, extra


class ResidueAttentionPooler(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, seq_emb: torch.Tensor, mask: torch.Tensor | None = None):
        if seq_emb.ndim != 3:
            raise ValueError(f"Expected (batch, seq_len, dim), got shape {tuple(seq_emb.shape)}")
        logits = self.scorer(seq_emb).squeeze(-1)
        weights = _masked_softmax(logits, mask)
        pooled = torch.bmm(weights.unsqueeze(1), seq_emb).squeeze(1)
        return pooled, weights.unsqueeze(-1), logits.unsqueeze(-1)


class AttentionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, extra_dim: int = 0):
        super().__init__()
        self.pooler = ResidueAttentionPooler(input_dim)
        self.extra_dim = extra_dim
        self.head = MLPModel(input_dim + extra_dim, hidden_dims, output_dim)

    def pooled_features(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ):
        pooled, weights, logits = self.pooler(seq_emb, mask)
        if extra_features is not None:
            pooled = torch.cat([pooled, extra_features], dim=1)
        return pooled, weights, logits

    def forward(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled, _, _ = self.pooled_features(seq_emb, mask, extra_features)
        return self.head(pooled)


class AttentionMultiHeadPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dims: dict[str, int], extra_dim: int = 0):
        super().__init__()
        self.pooler = ResidueAttentionPooler(input_dim)
        self.extra_dim = extra_dim
        self.backbone = SharedMLPBackbone(input_dim + extra_dim, hidden_dims)
        self.heads = nn.ModuleDict({
            head_name: nn.Linear(self.backbone.output_dim, output_dim)
            for head_name, output_dim in output_dims.items()
        })
        self.output_dims = dict(output_dims)

    def pooled_features(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ):
        pooled, weights, logits = self.pooler(seq_emb, mask)
        if extra_features is not None:
            pooled = torch.cat([pooled, extra_features], dim=1)
        features = self.backbone(pooled)
        return features, weights, logits

    def forward(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features, _, _ = self.pooled_features(seq_emb, mask, extra_features)
        return {
            head_name: head(features)
            for head_name, head in self.heads.items()
        }


@dataclass
class AttentionExportBundle:
    sample_ids: list[str]
    features: np.ndarray


class _BaseAttentionWrapper:
    def __init__(
        self,
        hidden_dims: tuple[int, ...],
        lr: float,
        epochs: int,
        optimizer_name: str,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.lr = lr
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device) if device else DEVICE
        self.model = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.input_dim_ = None
        self.extra_dim_ = 0
        self.batch_kind_ = None

    def _infer_dims(self, dataloader, batch_kind: str) -> tuple[int, int]:
        first_batch = next(iter(dataloader))
        seq_t, _, extra_t = batch_to_attention_tensors(first_batch, batch_kind, self.device)
        input_dim = int(seq_t.shape[2])
        extra_dim = int(extra_t.shape[1]) if extra_t is not None else 0
        return input_dim, extra_dim

    def transform_loader(self, dataloader, batch_kind: str) -> AttentionExportBundle:
        if self.model is None:
            raise ValueError("Model must be fit before calling transform_loader().")
        self.model.eval()
        chunks: list[np.ndarray] = []
        sample_ids: list[str] = []
        with torch.no_grad():
            for batch in dataloader:
                seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, self.device)
                pooled, _, _ = self.model.pooled_features(seq_t, mask_t, extra_t)
                chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
                sample_ids.extend(list(batch.sample_id))
        features = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
        return AttentionExportBundle(sample_ids=sample_ids, features=features)

    def build_checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Model must be fit before building a checkpoint.")
        return {
            "wrapper_class": self.__class__.__name__,
            "hidden_dims": list(self.hidden_dims),
            "lr": self.lr,
            "epochs": self.epochs,
            "optimizer_name": self.optimizer_name,
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
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(hidden_dims, lr, epochs, optimizer_name, early_stopping_patience, early_stopping_min_delta, device)

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self.model = AttentionPredictor(input_dim, self.hidden_dims, output_dim=1, extra_dim=extra_dim).to(self.device)
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
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(hidden_dims, lr, epochs, optimizer_name, early_stopping_patience, early_stopping_min_delta, device)

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
        input_dim, extra_dim = self._infer_dims(dataloader, batch_kind)
        self.input_dim_ = input_dim
        self.extra_dim_ = extra_dim
        self.batch_kind_ = batch_kind
        self.model = AttentionPredictor(input_dim, self.hidden_dims, output_dim=1, extra_dim=extra_dim).to(self.device)
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
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        batch_size: int = 1024,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__(hidden_dims, lr, epochs, optimizer_name, early_stopping_patience, early_stopping_min_delta, device)
        self.batch_size = batch_size
        self.output_dims_ = None

    def fit_loader(self, dataloader, batch_kind: str, val_dataloader=None):
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
        self.model = AttentionMultiHeadPredictor(input_dim, self.hidden_dims, output_dims, extra_dim=extra_dim).to(self.device)
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


def save_attention_checkpoint(wrapper, run_dir: Path) -> Path:
    path = run_dir / "model.pt"
    torch.save(wrapper.build_checkpoint(), path)
    return path


def load_pooler_from_checkpoint(path: str | Path, device: str | torch.device | None = None):
    map_location = torch.device(device) if device else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    input_dim = int(checkpoint["input_dim"])
    pooler = ResidueAttentionPooler(input_dim)
    pooler.load_state_dict(checkpoint["pooler_state_dict"])
    pooler.to(map_location)
    pooler.eval()
    return pooler, checkpoint


def export_features_with_pooler(pooler: ResidueAttentionPooler, dataloader, batch_kind: str, device: str | torch.device | None = None):
    torch_device = torch.device(device) if device else DEVICE
    pooler = pooler.to(torch_device)
    pooler.eval()
    chunks: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in dataloader:
            seq_t, mask_t, extra_t = batch_to_attention_tensors(batch, batch_kind, torch_device)
            pooled, _, _ = pooler(seq_t, mask_t)
            if extra_t is not None:
                pooled = torch.cat([pooled, extra_t], dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            sample_ids.extend(list(batch.sample_id))
    features = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
    return AttentionExportBundle(sample_ids=sample_ids, features=features)


ATTENTION_MODEL_MAP = {
    "classification": {
        "attention_mlp_classification": {
            "wrap": AttentionBinaryClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "attention_sgd_classification": {
            "wrap": AttentionBinaryClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    },
    "regression": {
        "attention_mlp_regressor": {
            "wrap": AttentionRegressorTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "attention_sgd_regressor": {
            "wrap": AttentionRegressorTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    },
}


ATTENTION_STRUCTURED_MODEL_MAP = {
    "multilabel_ec": {
        "attention_mlp_classification": {
            "wrap": AttentionECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (512, 256),
                "lr": 1e-3,
                "epochs": 200,
                "optimizer_name": "adam",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
        "attention_sgd_classification": {
            "wrap": AttentionECMultiHeadClassifierTorch,
            "params": {
                "hidden_dims": (),
                "lr": 0.01,
                "epochs": 200,
                "optimizer_name": "sgd",
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
            },
            "residue_aware": True,
        },
    },
}


def get_attention_model_map(task_type: str, label_kind: str) -> dict[str, dict[str, Any]]:
    if label_kind in ATTENTION_STRUCTURED_MODEL_MAP:
        return ATTENTION_STRUCTURED_MODEL_MAP[label_kind]
    return ATTENTION_MODEL_MAP[task_type]
