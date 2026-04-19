"""Compatibility facade for Torch model wrappers.

The wrapper implementations now live in focused modules:

- `torch_standard_wrappers.py` for pooled classification/regression baselines
- `torch_structured_wrappers.py` for EC and residue-level tasks

Imports from `models.torch_wrappers` continue to work so the training
pipeline, tests, and older notebooks do not need to change all at once.
"""

from models.torch_standard_wrappers import (
    LinearClassifierTorch,
    LinearRegressorTorch,
    MLPClassifierTorch,
    MLPRegressorTorch,
    SGDClassifierTorch,
    SGDRegressorTorch,
)
from models.torch_structured_wrappers import (
    ECMultiHeadClassifierTorch,
    ResidueBinaryClassifierTorch,
)

__all__ = [
    "LinearRegressorTorch",
    "LinearClassifierTorch",
    "SGDRegressorTorch",
    "SGDClassifierTorch",
    "MLPRegressorTorch",
    "MLPClassifierTorch",
    "ECMultiHeadClassifierTorch",
    "ResidueBinaryClassifierTorch",
]
