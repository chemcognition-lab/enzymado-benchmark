"""Stable public entrypoints for attention-based PyTorch wrappers."""

from enzymado.training.attention_ec_classification import (
    AttentionECMultiHeadClassifierImplementation,
)
from enzymado.training.attention_pooled import (
    AttentionBinaryClassifierImplementation,
    AttentionRegressorImplementation,
    _PooledAttentionWrapper,
)
from enzymado.training.attention_residue_classification import (
    AttentionResidueBinaryClassifierImplementation,
)


class AttentionRegressorTorch(AttentionRegressorImplementation):
    """Attention-pooled regressor for global tasks."""


class AttentionBinaryClassifierTorch(AttentionBinaryClassifierImplementation):
    """Attention-pooled binary classifier for global tasks."""


class AttentionECMultiHeadClassifierTorch(AttentionECMultiHeadClassifierImplementation):
    """Attention-pooled multi-head classifier for EC prediction."""


class AttentionResidueBinaryClassifierTorch(
    AttentionResidueBinaryClassifierImplementation
):
    """Per-residue binary classifier with self-attention over residues."""


__all__ = [
    "_PooledAttentionWrapper",
    "AttentionRegressorTorch",
    "AttentionBinaryClassifierTorch",
    "AttentionECMultiHeadClassifierTorch",
    "AttentionResidueBinaryClassifierTorch",
]
