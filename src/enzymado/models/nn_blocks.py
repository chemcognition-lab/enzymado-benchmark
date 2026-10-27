"""Reusable neural network building blocks.

Example:
    model = nn_blocks.MLPModel(960, (512, 512), 1, activation="swiglu")
"""

from __future__ import annotations

import collections.abc

import jaxtyping
import torch
from torch import nn
from torch.nn import functional


class LinearModel(nn.Module):
    """Single linear layer used by the linear baseline wrappers."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "*batch in_dim"]
    ) -> jaxtyping.Float[torch.Tensor, "*batch out_dim"]:
        return self.linear(x)


class SwiGLU(nn.Module):
    """Gate the second half of a projection with SiLU of its first half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return functional.silu(gate) * value


class MLPModel(nn.Module):
    """Feed-forward MLP used for pooled or residue-wise embedding inputs.

    Each hidden dimension specifies one layer. SwiGLU uses the original HPO's
    two-thirds width, rounded up to a multiple of 32, with a doubled projection.
    Layer normalization precedes each hidden projection, as in origin/hpo.
    Defaults retain the original SiLU stack and its state-dict keys exactly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: collections.abc.Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
        *,
        activation: str = "silu",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        activation = activation.lower()
        if activation not in {"relu", "silu", "swiglu"}:
            raise ValueError(f"Unsupported activation: {activation}")
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            if normalize:
                layers.append(nn.LayerNorm(in_dim))
            if activation == "swiglu":
                hidden_dim = ((int(hidden_dim * (2.0 / 3.0)) + 31) // 32) * 32
                layers.append(nn.Linear(in_dim, hidden_dim * 2))
                layers.append(SwiGLU())
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU() if activation == "relu" else nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "*batch in_dim"]
    ) -> jaxtyping.Float[torch.Tensor, "*batch out_dim"]:
        return self.net(x)


class SharedMLPBackbone(nn.Module):
    """Shared hidden stack for multi-head EC prediction models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: collections.abc.Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Identity() if not layers else nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "*batch in_dim"]
    ) -> jaxtyping.Float[torch.Tensor, "*batch out_dim"]:
        return self.net(x)


class MultiHeadMLPModel(nn.Module):
    """Shared MLP backbone followed by one linear head per target group."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: collections.abc.Sequence[int],
        output_dims: collections.abc.Mapping[str, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = SharedMLPBackbone(input_dim, hidden_dims, dropout=dropout)
        self.heads = nn.ModuleDict(
            {
                head_name: nn.Linear(self.backbone.output_dim, output_dim)
                for head_name, output_dim in output_dims.items()
            }
        )
        self.output_dims = dict(output_dims)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "*batch in_dim"]
    ) -> dict[str, jaxtyping.Float[torch.Tensor, "*batch head_dim"]]:
        features = self.backbone(x)
        return {head_name: head(features) for head_name, head in self.heads.items()}
