"""Reusable neural network building blocks."""

from __future__ import annotations

import collections.abc

from jaxtyping import Float
from torch import Tensor
import torch.nn as nn


class LinearModel(nn.Module):
    """Single linear layer used by the linear baseline wrappers."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> Float[Tensor, "*batch out_dim"]:
        return self.linear(x)


class MLPModel(nn.Module):
    """Feed-forward MLP used for pooled or residue-wise embedding inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: collections.abc.Sequence[int],
        output_dim: int,
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
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> Float[Tensor, "*batch out_dim"]:
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
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> Float[Tensor, "*batch out_dim"]:
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
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> dict[str, Float[Tensor, "*batch head_dim"]]:
        features = self.backbone(x)
        return {head_name: head(features) for head_name, head in self.heads.items()}
