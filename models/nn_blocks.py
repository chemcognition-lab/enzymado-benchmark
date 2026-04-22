"""Reusable neural network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """Single linear layer used by the linear baseline wrappers."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

def width_correction_fn(expansion_ratio: float, input_dim: int, multiple_of: int) -> int:
    target_dim = int(expansion_ratio * input_dim)
    return ((target_dim + multiple_of - 1) // multiple_of) * multiple_of

class MLPModel(nn.Module):
    """Feed-forward MLP used for pooled or residue-wise embedding inputs."""
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_layers: int = 1,
        expansion_ratio: float = 4.0, 
        activation: str = "relu", 
        dropout_prob: float = 0.0, 
        normalize: bool = False,
        multiple_of: int = 32
    ):
        super().__init__()
        
        # Adjust param count for GLU activations
        activation = activation.lower()
        if activation in ["relu", "silu"]:
            gated = False
            hidden_dim = width_correction_fn(expansion_ratio, input_dim, multiple_of)
        elif activation == "swiglu":
            gated = True
            # 2/3 factor to normalize GLU parameter count
            gated_expansion_ratio = expansion_ratio * (2.0 / 3.0)
            hidden_dim = width_correction_fn(gated_expansion_ratio, input_dim, multiple_of)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Hidden dimension
        proj_dim_out = hidden_dim * 2 if gated else hidden_dim
        
        layers = []
        in_features = input_dim
        
        for _ in range(num_layers):
            # Normalization
            if normalize:
                layers.append(nn.LayerNorm(in_features))

            layers.append(nn.Linear(in_features, proj_dim_out))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "swiglu":
                layers.append(SwiGLU())
                
            # Dropout
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))

            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
