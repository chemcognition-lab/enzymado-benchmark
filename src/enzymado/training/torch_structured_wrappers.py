"""Stable public entrypoints for structured PyTorch wrappers."""

from enzymado.training.torch_ec_classification import (
    ECMultiHeadClassifierImplementation,
)
from enzymado.training.torch_residue_classification import (
    ResidueBinaryClassifierImplementation,
)


class ECMultiHeadClassifierTorch(ECMultiHeadClassifierImplementation):
    """Shared-backbone classifier for hierarchical EC targets on pooled inputs."""

class ResidueBinaryClassifierTorch(ResidueBinaryClassifierImplementation):
    """Binary classifier over per-residue embeddings with padding-aware BCE loss."""


__all__ = [
    "ECMultiHeadClassifierTorch",
    "ResidueBinaryClassifierTorch",
]
