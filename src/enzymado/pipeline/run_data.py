"""Prepare task data and select its benchmark training route."""

import dataclasses
from pathlib import Path

import enzymado.data_loading.data_registry
import enzymado.data_loading.task_data
import enzymado.training.train


@dataclasses.dataclass(frozen=True)
class PreparedTask:
    """Prepared data and its selected training route."""

    data: (
        enzymado.data_loading.task_data.PooledTaskData
        | enzymado.data_loading.task_data.LoaderTaskData
    )
    route: enzymado.training.train.TrainingRouteSpec


@dataclasses.dataclass(frozen=True)
class TaskSizes:
    """Split sizes and whether validation is available."""

    n_train: int
    n_val: int
    n_test: int
    use_validation: bool


def prepare(
    *,
    config: enzymado.data_loading.data_registry.TaskConfig,
    task_name: str,
    emb_path: str | Path,
    split_variant: str,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    residue_store_mode: str,
    embedding_dim: int | None,
    route_override: enzymado.training.train.TrainingRouteSpec | None = None,
) -> PreparedTask:
    """Load one task and preserve the existing route-selection rules."""
    if route_override is enzymado.training.train.ATTENTION_ROUTE:
        prepared = enzymado.training.train.ATTENTION_ROUTE.prepare(  # type: ignore[misc]
            config,
            task_name,
            emb_path,
            split_variant,
            allowed_ids=allowed_ids,
            sequence_to_id=sequence_to_id,
            residue_store_mode=residue_store_mode,
            embedding_dim=embedding_dim,
        )
        route = enzymado.training.train.ATTENTION_ROUTE
    else:
        prepared = enzymado.data_loading.task_data.load_task_data(
            task_name,
            emb_path,
            split_variant,
            allowed_ids=allowed_ids,
            sequence_to_id=sequence_to_id,
            residue_store_mode=residue_store_mode,
            embedding_dim=embedding_dim,
        )
        route = (
            enzymado.training.train.GLOBAL_ROUTE
            if isinstance(prepared, enzymado.data_loading.task_data.PooledTaskData)
            else enzymado.training.train.RESIDUE_ROUTE
        )

    return PreparedTask(
        data=prepared,
        route=route,
    )


def summarize(
    data: (
        enzymado.data_loading.task_data.PooledTaskData
        | enzymado.data_loading.task_data.LoaderTaskData
    ),
) -> TaskSizes:
    """Calculate split sizes after the caller owns the prepared data."""
    n_train, n_val, n_test = enzymado.training.train._dataset_sizes(data)
    return TaskSizes(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        use_validation=n_val > 0,
    )
