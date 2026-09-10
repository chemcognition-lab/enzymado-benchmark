"""Shared type contracts for benchmark models, arrays, and registries."""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
from pathlib import Path
import typing

import numpy as np
import numpy.typing as npt
import torch

type NDArray = npt.NDArray[np.generic]
type Array = NDArray | torch.Tensor
type ArrayPayload = Array | dict[str, Array]
type OptionalArrayPayload = ArrayPayload | None
type TargetPayload = tuple[ArrayPayload, OptionalArrayPayload]
type MetricTask = typing.Literal["regression", "classification"]
type MetricFn = collections.abc.Callable[[NDArray, NDArray, NDArray | None], float]
type AggregateMetricFn = collections.abc.Callable[[NDArray], NDArray]


class MetricStatistic(enum.StrEnum):
    """Sufficient statistic used to bootstrap a metric efficiently."""

    RAW = "raw"
    ERROR_SUMS = "error_sums"
    CONFUSION = "confusion"


class MetricInput(enum.StrEnum):
    """Prediction representation consumed by a metric."""

    PREDICTIONS = "predictions"
    PROBABILITIES_OR_PREDICTIONS = "probabilities_or_predictions"


@dataclasses.dataclass(frozen=True, slots=True)
class MetricSpec:
    """Complete declaration of one benchmark metric."""

    name: str
    task_type: MetricTask
    higher_is_better: bool
    statistic: MetricStatistic
    # Only metrics that are recomputed from raw arrays need a function: the
    # ``RAW`` bootstrap path calls it per replicate, and ``evaluate_predictions``
    # calls it for the task types it serves. ``CONFUSION`` metrics are derived
    # from per-row counts and score histograms instead, so declaring a function
    # for them would advertise an implementation production never runs.
    fn: MetricFn | None = None
    input_kind: MetricInput = MetricInput.PREDICTIONS
    aggregate_fn: AggregateMetricFn | None = None
    score_aggregate_fn: AggregateMetricFn | None = None
    is_primary: bool = False
    rating_weight: float = 1.0


class ModelInfo(typing.TypedDict):
    """Configuration stored for one model implementation in the registry."""

    wrap: collections.abc.Callable[..., object]
    params: dict[str, object]
    residue_aware: typing.NotRequired[bool]
    loader_batch_size: typing.NotRequired[int]
    loader_num_workers: typing.NotRequired[int]
    loader_seed: typing.NotRequired[int]
    concat_residue_batch: typing.NotRequired[bool]
    flat_ec_labels: typing.NotRequired[bool]


class EmbeddingInfo(typing.TypedDict):
    """Embedding manifest entry consumed by benchmark orchestration."""

    path: str | Path
    dim: int | None


type ModelRegistry = dict[str, ModelInfo]
type ModelMap = dict[str, ModelRegistry]
type EmbeddingMap = dict[str, EmbeddingInfo]


class UniProtMatch(typing.TypedDict):
    """Local or remote UniProt match used during property preparation."""

    accession: str | None
    ec_numbers: list[str]


class EnzymeMetadata(typing.TypedDict):
    """Normalized enzyme metadata stored for one protein sequence."""

    accession: str | None
    ec_number: str | None
    is_enzyme: bool
    match_source: str


type SequenceLookupResult = tuple[dict[str, str], dict[str, str]]
type EnzymeLookupResult = tuple[dict[str, EnzymeMetadata], dict[str, str]]
type SwissProtSequenceIndex = dict[str, list[UniProtMatch]]


class ArrayPredictor(typing.Protocol):
    """Model capable of predicting from one array-like feature payload."""

    def predict(self, features: Array) -> ArrayPayload:
        """Return predictions for in-memory features."""

        ...


class LoaderPredictor(typing.Protocol):
    """Model capable of predicting batches supplied by a data loader."""

    def predict_loader(self, dataloader: object, *, batch_kind: str) -> ArrayPayload:
        """Return predictions for every sample yielded by ``dataloader``."""

        ...


