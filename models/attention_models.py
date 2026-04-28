"""Compatibility facade for attention model blocks, wrappers, and registries."""

from models.attention_blocks import (
    AttentionAggregation,
    AttentionMultiHeadPredictor,
    AttentionPredictor,
    AttentionResidueMixer,
    AttentionResiduePredictor,
    MeanAggregation,
    batch_to_attention_tensors,
)
from models.attention_registry import (
    ATTENTION_MODEL_MAP,
    ATTENTION_STRUCTURED_MODEL_MAP,
    get_attention_model_map,
)
from models.attention_wrappers import (
    AttentionBinaryClassifierTorch,
    AttentionECMultiHeadClassifierTorch,
    AttentionRegressorTorch,
    AttentionResidueBinaryClassifierTorch,
    DEVICE,
    save_attention_checkpoint,
)

__all__ = [
    "DEVICE",
    "MeanAggregation",
    "AttentionAggregation",
    "batch_to_attention_tensors",
    "AttentionPredictor",
    "AttentionMultiHeadPredictor",
    "AttentionResidueMixer",
    "AttentionResiduePredictor",
    "AttentionRegressorTorch",
    "AttentionBinaryClassifierTorch",
    "AttentionECMultiHeadClassifierTorch",
    "AttentionResidueBinaryClassifierTorch",
    "save_attention_checkpoint",
    "ATTENTION_MODEL_MAP",
    "ATTENTION_STRUCTURED_MODEL_MAP",
    "get_attention_model_map",
]
