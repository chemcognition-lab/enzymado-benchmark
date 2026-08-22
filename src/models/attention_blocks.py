"""Neural building blocks for attention-pooled global models."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from .nn_blocks import MLPModel, SharedMLPBackbone


def _seq_emb_to_tensor(protein_emb, device: torch.device) -> torch.Tensor:
    """Move a protein sequence embedding to the target device without going via NumPy for BF16.

    Accepts:
      - torch.Tensor (e.g. BF16 from the residue collator) → just transferred to device.
      - np.ndarray with dtype=int16 (BF16 raw bits) → reinterpreted as bfloat16.
      - any other numpy-convertible source → upcast to float32.
    """
    if torch.is_tensor(protein_emb):
        return protein_emb.to(device)
    array = np.asarray(protein_emb)
    if array.dtype == np.int16:
        return torch.from_numpy(array).view(torch.bfloat16).to(device)
    return torch.from_numpy(array.astype(np.float32, copy=False)).to(device)


def attention_autocast(device: torch.device):
    """BF16 autocast context for attention forward passes; loss must still be FP32."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


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
        pre_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pre_layer_norm = bool(pre_layer_norm)
        self.input_layer_norm = (
            nn.LayerNorm(embed_dim) if self.pre_layer_norm else nn.Identity()
        )
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
        attn_input = self.input_layer_norm(x)
        avg_emb = self.mean_agg(attn_input, key_padding_mask).unsqueeze(1)
        global_emb, _ = self.cross_attn_layer(
            query=avg_emb,
            key=attn_input,
            value=attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return global_emb.squeeze(1), None, None


def batch_to_attention_tensors(batch, batch_kind: str, device: torch.device):
    """Convert a collated batch into tensors used by attention wrappers."""
    if batch_kind == "protein_global":
        seq = _seq_emb_to_tensor(batch.protein_emb, device)
        mask = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        context_feat = getattr(batch, "context_feat", None)
        extra = (
            torch.from_numpy(np.asarray(context_feat, dtype=np.float32)).to(device)
            if context_feat is not None
            else None
        )
    elif batch_kind == "protein_ligand_global":
        seq = _seq_emb_to_tensor(batch.protein_emb, device)
        mask = torch.from_numpy(np.asarray(batch.protein_mask, dtype=bool)).to(device)
        extra_parts = [
            torch.from_numpy(np.asarray(batch.ligand_feat, dtype=np.float32)).to(device)
        ]
        context_feat = getattr(batch, "context_feat", None)
        if context_feat is not None:
            extra_parts.append(
                torch.from_numpy(
                    np.asarray(context_feat, dtype=np.float32)
                ).to(device)
            )
        extra = torch.cat(extra_parts, dim=1)
    elif batch_kind == "mutation_delta_global":
        seq = _seq_emb_to_tensor(batch.protein_delta_emb, device)
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
        dropout: float = 0.0,
        pre_attention_layer_norm: bool = False,
    ):
        super().__init__()
        self.pooler = AttentionAggregation(
            input_dim,
            num_heads=num_attention_heads,
            dropout_rate=attention_dropout_rate,
            pre_layer_norm=pre_attention_layer_norm,
        )
        self.extra_dim = extra_dim
        self.head = MLPModel(input_dim + extra_dim, hidden_dims, output_dim, dropout=dropout)

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
        dropout: float = 0.0,
        pre_attention_layer_norm: bool = False,
    ):
        super().__init__()
        self.pooler = AttentionAggregation(
            input_dim,
            num_heads=num_attention_heads,
            dropout_rate=attention_dropout_rate,
            pre_layer_norm=pre_attention_layer_norm,
        )
        self.extra_dim = extra_dim
        self.backbone = SharedMLPBackbone(input_dim + extra_dim, hidden_dims, dropout=dropout)
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


class AttentionResidueMixer(nn.Module):
    """Self-attention over residues with LayerNorm + residual; no pooling."""

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
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.layer_norm(x)
        attended, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return x + attended


class AttentionResiduePredictor(nn.Module):
    """Self-attention over residues plus a per-residue MLP head (no pooling)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        num_attention_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mixer = AttentionResidueMixer(
            input_dim,
            num_heads=num_attention_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.head = MLPModel(input_dim, hidden_dims, output_dim, dropout=dropout)

    def forward(
        self,
        seq_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = ~mask if mask is not None else None
        mixed = self.mixer(seq_emb, key_padding_mask)
        return self.head(mixed)


__all__ = [
    "MeanAggregation",
    "AttentionAggregation",
    "attention_autocast",
    "batch_to_attention_tensors",
    "AttentionPredictor",
    "AttentionMultiHeadPredictor",
    "AttentionResidueMixer",
    "AttentionResiduePredictor",
]
