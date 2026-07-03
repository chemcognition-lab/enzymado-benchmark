"""Training pipelines, wrappers, and output helpers."""

from data_loading import LoaderTaskData, PooledTaskData
from training.residue import (
    ResidueAwareLoaders,
    ResidueAwareOutputs,
    build_residue_aware_loaders,
    collect_residue_aware_outputs,
    train_residue_aware_model,
)
from training.train import (
    ATTENTION_ROUTE,
    GLOBAL_ROUTE,
    RESIDUE_ROUTE,
    TrainingRouteSpec,
    _dataset_sizes,
    _select_global_arrays,
    _select_training_route,
    prepare_attention,
    train_attention,
    train_global,
    train_residue,
)

__all__ = [
    "ATTENTION_ROUTE",
    "GLOBAL_ROUTE",
    "RESIDUE_ROUTE",
    "LoaderTaskData",
    "PooledTaskData",
    "ResidueAwareLoaders",
    "ResidueAwareOutputs",
    "TrainingRouteSpec",
    "_dataset_sizes",
    "_select_global_arrays",
    "_select_training_route",
    "build_residue_aware_loaders",
    "collect_residue_aware_outputs",
    "prepare_attention",
    "train_attention",
    "train_global",
    "train_residue",
    "train_residue_aware_model",
]
