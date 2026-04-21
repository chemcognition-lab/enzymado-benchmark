"""Neural building blocks for attention-pooled global models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.nn_blocks import MLPModel, SharedMLPBackbone


class MeanAggregation(nn.Module):
    """Mean-pool sequence embeddings while ignoring padded positions."""

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, seq_len, dim), got shape {tuple(x.shape)}")
        if key_padding_mask is None:
            return x.mean(dim=1)

        valid_mask = (~key_padding_mask).unsqueeze(-1).float()
        return (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)


class AttentionAggregation(nn.Module):
    """Aggregate residue embeddings with a learned cross-attention query."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.cross_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mean_agg = MeanAggregation()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ):
        avg_emb = self.mean_agg(x, key_padding_mask).unsqueeze(1)
        global_emb, _ = self.cross_attn_layer(
            query=avg_emb,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return global_emb.squeeze(1), None, None


def batch_to_attention_tensors(batch, batch_kind: str, device: torch.device):
    """Convert a collated batch into tensors used by attention wrappers."""
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


class AttentionPredictor(nn.Module):
    """Attention pooler plus an MLP prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        extra_dim: int = 0,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.pooler = AttentionAggregation(
            input_dim,
            num_heads=num_attention_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.extra_dim = extra_dim
        self.head = MLPModel(input_dim + extra_dim, hidden_dims, output_dim)

    def pooled_features(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
    ):
        key_padding_mask = ~mask if mask is not None else None
        pooled, weights, logits = self.pooler(seq_emb, key_padding_mask)
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
    """Attention pooler plus a shared backbone with one head per output group."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dims: dict[str, int],
        extra_dim: int = 0,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.pooler = AttentionAggregation(
            input_dim,
            num_heads=num_attention_heads,
            dropout_rate=attention_dropout_rate,
        )
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
        key_padding_mask = ~mask if mask is not None else None
        pooled, weights, logits = self.pooler(seq_emb, key_padding_mask)
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


__all__ = [
    "MeanAggregation",
    "AttentionAggregation",
    "batch_to_attention_tensors",
    "AttentionPredictor",
    "AttentionMultiHeadPredictor",
]
