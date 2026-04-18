"""Reusable neural network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Single linear layer used by the linear baseline wrappers."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


class MLPModel(nn.Module):
    """Feed-forward MLP used for pooled or residue-wise embedding inputs."""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class SharedMLPBackbone(nn.Module):
    """Shared hidden stack for multi-head EC prediction models."""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.net = nn.Identity() if not layers else nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, X):
        return self.net(X)


class MultiHeadMLPModel(nn.Module):
    """Shared MLP backbone followed by one linear head per target group."""

    def __init__(self, input_dim, hidden_dims, output_dims):
        super().__init__()
        self.backbone = SharedMLPBackbone(input_dim, hidden_dims)
        self.heads = nn.ModuleDict(
            {
                head_name: nn.Linear(self.backbone.output_dim, output_dim)
                for head_name, output_dim in output_dims.items()
            }
        )
        self.output_dims = dict(output_dims)

    def forward(self, X):
        features = self.backbone(X)
        return {head_name: head(features) for head_name, head in self.heads.items()}


# Backward-compatible alias.
SGDModel = LinearModel
