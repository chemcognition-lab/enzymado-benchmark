"""Stable public entrypoints for standard PyTorch wrappers."""

from enzymado.training.torch_standard_classification import (
    LinearClassifierImplementation,
    MLPClassifierImplementation,
)
from enzymado.training.torch_standard_regression import (
    LinearRegressorImplementation,
    MLPRegressorImplementation,
)


class LinearRegressorTorch(LinearRegressorImplementation):
    """Linear regressor baseline for 2D pooled or 3D residue-aware inputs."""

class LinearClassifierTorch(LinearClassifierImplementation):
    """Linear classifier baseline for pooled or residue-aware embedding inputs."""


class MLPRegressorTorch(MLPRegressorImplementation):
    """MLP regressor for pooled or residue-aware embedding inputs."""


class MLPClassifierTorch(MLPClassifierImplementation):
    """MLP classifier for pooled or residue-aware embedding inputs."""


__all__ = [
    "LinearRegressorTorch",
    "LinearClassifierTorch",
    "MLPRegressorTorch",
    "MLPClassifierTorch",
]
