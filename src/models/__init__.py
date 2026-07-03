"""Public model exports used by the benchmark training pipeline.

models/ is a dependency-free leaf package: it exposes only stateless building
blocks (optimizer/training-loop helpers, nn building blocks). Trainable model
wrapper classes live in training/*_wrappers.py and are imported from there
directly, not re-exported here, so models/ never depends on training/.
"""

from .common import (
    DEVICE,
    batch_to_input,
    make_optimizer,
    masked_mean_pool,
    pad_1d_array_list,
    prepare_inputs,
    run_training_with_early_stopping,
)
from .nn_blocks import (
    LinearModel,
    MLPModel,
    MultiHeadMLPModel,
    SharedMLPBackbone,
)

__all__ = [
    "DEVICE",
    "masked_mean_pool",
    "make_optimizer",
    "pad_1d_array_list",
    "batch_to_input",
    "prepare_inputs",
    "run_training_with_early_stopping",
    "LinearModel",
    "MLPModel",
    "SharedMLPBackbone",
    "MultiHeadMLPModel",
]
