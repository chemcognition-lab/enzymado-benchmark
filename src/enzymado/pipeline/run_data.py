"""Prepare task data and select its benchmark training route."""

import dataclasses
from pathlib import Path

from enzymado.data_loading import data_registry, task_data
from enzymado.training import train


@dataclasses.dataclass(frozen=True)
class PreparedTask:
    """Prepared data and its selected training route."""

    data: task_data.PooledTaskData | task_data.LoaderTaskData
    route: train.TrainingRouteSpec


@dataclasses.dataclass(frozen=True)
class TaskSizes:
    """Split sizes and whether validation is available."""

    n_train: int
    n_val: int
    n_test: int
    use_validation: bool


def prepare(
    *,
    config: data_registry.TaskConfig,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    residue_store_mode: str,
    embedding_dim: int | None,
    route_override: train.TrainingRouteSpec | None = None,
) -> PreparedTask:
    """Load one task and preserve the existing route-selection rules."""
    if route_override is train.ATTENTION_ROUTE:
        prepared = train.ATTENTION_ROUTE.prepare(  # type: ignore[misc]
            config,
            task_name,
            emb_path,
            split_variant,
            allowed_ids=allowed_ids,
            sequence_to_id=sequence_to_id,
            residue_store_mode=residue_store_mode,
            embedding_dim=embedding_dim,
        )
        route = train.ATTENTION_ROUTE
    else:
        prepared = task_data.load_task_data(
            task_name,
            emb_path,
            split_variant,
            allowed_ids=allowed_ids,
            sequence_to_id=sequence_to_id,
            residue_store_mode=residue_store_mode,
            embedding_dim=embedding_dim,
        )
        route = (
            train.GLOBAL_ROUTE
            if isinstance(prepared, task_data.PooledTaskData)
            else train.RESIDUE_ROUTE
        )

    return PreparedTask(data=prepared, route=route)


def summarize(
    data: task_data.PooledTaskData | task_data.LoaderTaskData,
) -> TaskSizes:
    """Calculate split sizes after the caller owns the prepared data."""
    n_train, n_val, n_test = train._dataset_sizes(data)
    return TaskSizes(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        use_validation=n_val > 0,
    )
