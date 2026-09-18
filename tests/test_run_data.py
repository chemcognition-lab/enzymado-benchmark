"""Characterization tests for benchmark task-data preparation."""

from pathlib import Path

import pytest

import enzymado.data_loading.data_registry
import enzymado.data_loading.task_data
import enzymado.pipeline.run_data
import enzymado.training.train


def _config() -> enzymado.data_loading.data_registry.TaskConfig:
    """Build a minimal registered-style configuration for preparation tests."""
    return enzymado.data_loading.data_registry.TaskConfig(
        name="solubility",
        batch_kind=enzymado.data_loading.data_registry.TaskType.PROTEIN_GLOBAL,
        label_kind=enzymado.data_loading.data_registry.LabelType.BINARY,
        dataset_paths={"similarity": Path("/tmp/solubility.csv")},
        target_column="solubility",
    )


def test_pooled_data_selects_global_route_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock pooled loading arguments, route selection, sizes, and validation."""
    prepared = object.__new__(enzymado.data_loading.task_data.PooledTaskData)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_data(
        *args: object,
        **kwargs: object,
    ) -> enzymado.data_loading.task_data.PooledTaskData:
        calls.append((args, kwargs))
        return prepared

    def dataset_sizes(_data: object) -> tuple[int, int, int]:
        return 2, 1, 1

    monkeypatch.setattr(enzymado.data_loading.task_data, "load_task_data", load_data)
    monkeypatch.setattr(enzymado.training.train, "_dataset_sizes", dataset_sizes)

    result = enzymado.pipeline.run_data.prepare(
        config=_config(),
        task_name="solubility",
        emb_path="/tmp/emb.safetensors",
        split_variant="similarity",
        allowed_ids={"P1"},
        sequence_to_id={"SEQ": "P1"},
        residue_store_mode="lazy",
        embedding_dim=128,
    )
    sizes = enzymado.pipeline.run_data.summarize(result.data)

    assert calls == [
        (
            ("solubility", "/tmp/emb.safetensors", "similarity"),
            {
                "allowed_ids": {"P1"},
                "sequence_to_id": {"SEQ": "P1"},
                "residue_store_mode": "lazy",
                "embedding_dim": 128,
            },
        )
    ]
    assert result.data is prepared
    assert result.route is enzymado.training.train.GLOBAL_ROUTE
    assert (sizes.n_train, sizes.n_val, sizes.n_test) == (2, 1, 1)
    assert sizes.use_validation is True


def test_loader_data_selects_residue_route_without_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock residue route selection and empty-validation behavior."""
    prepared = object.__new__(enzymado.data_loading.task_data.LoaderTaskData)

    def load_data(
        *_args: object,
        **_kwargs: object,
    ) -> enzymado.data_loading.task_data.LoaderTaskData:
        return prepared

    def dataset_sizes(_data: object) -> tuple[int, int, int]:
        return 8, 0, 2

    monkeypatch.setattr(enzymado.data_loading.task_data, "load_task_data", load_data)
    monkeypatch.setattr(enzymado.training.train, "_dataset_sizes", dataset_sizes)

    result = enzymado.pipeline.run_data.prepare(
        config=_config(),
        task_name="binding_site",
        emb_path=Path("/tmp/emb.safetensors"),
        split_variant="similarity",
        allowed_ids=None,
        sequence_to_id={},
        residue_store_mode="auto",
        embedding_dim=None,
    )
    sizes = enzymado.pipeline.run_data.summarize(result.data)

    assert result.data is prepared
    assert result.route is enzymado.training.train.RESIDUE_ROUTE
    assert (sizes.n_train, sizes.n_val, sizes.n_test) == (8, 0, 2)
    assert sizes.use_validation is False


def test_attention_override_uses_attention_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock attention preparation arguments, route identity, and validation."""
    prepared = object.__new__(enzymado.data_loading.task_data.LoaderTaskData)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def prepare_attention(
        *args: object,
        **kwargs: object,
    ) -> enzymado.data_loading.task_data.LoaderTaskData:
        calls.append((args, kwargs))
        return prepared

    def train_attention(**_kwargs: object) -> None:
        return None

    def fail_load(
        *_args: object,
        **_kwargs: object,
    ) -> enzymado.data_loading.task_data.PooledTaskData:
        raise AssertionError("Standard task loading should not run for attention.")

    def dataset_sizes(_data: object) -> tuple[int, int, int]:
        return 2, 0, 1

    attention_route = enzymado.training.train.TrainingRouteSpec(
        name="attention",
        train=train_attention,
        prepare=prepare_attention,
    )
    monkeypatch.setattr(enzymado.training.train, "ATTENTION_ROUTE", attention_route)
    monkeypatch.setattr(enzymado.data_loading.task_data, "load_task_data", fail_load)
    monkeypatch.setattr(enzymado.training.train, "_dataset_sizes", dataset_sizes)
    config = _config()

    result = enzymado.pipeline.run_data.prepare(
        config=config,
        task_name="solubility",
        emb_path="/tmp/emb.safetensors",
        split_variant="similarity",
        allowed_ids={"P1"},
        sequence_to_id={"SEQ": "P1"},
        residue_store_mode="lazy",
        embedding_dim=128,
        route_override=attention_route,
    )
    sizes = enzymado.pipeline.run_data.summarize(result.data)

    assert calls == [
        (
            (config, "solubility", "/tmp/emb.safetensors", "similarity"),
            {
                "allowed_ids": {"P1"},
                "sequence_to_id": {"SEQ": "P1"},
                "residue_store_mode": "lazy",
                "embedding_dim": 128,
            },
        )
    ]
    assert result.data is prepared
    assert result.route is attention_route
    assert (sizes.n_train, sizes.n_val, sizes.n_test) == (2, 0, 1)
    assert sizes.use_validation is False


def test_preparation_finishes_before_size_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep loaded data available to the runner if size calculation fails."""
    prepared = object.__new__(enzymado.data_loading.task_data.PooledTaskData)

    def load_data(
        *_args: object,
        **_kwargs: object,
    ) -> enzymado.data_loading.task_data.PooledTaskData:
        return prepared

    def fail_sizes(_data: object) -> tuple[int, int, int]:
        raise RuntimeError("size failure")

    monkeypatch.setattr(enzymado.data_loading.task_data, "load_task_data", load_data)
    monkeypatch.setattr(enzymado.training.train, "_dataset_sizes", fail_sizes)

    result = enzymado.pipeline.run_data.prepare(
        config=_config(),
        task_name="solubility",
        emb_path="/tmp/emb.safetensors",
        split_variant="similarity",
        allowed_ids=None,
        sequence_to_id={},
        residue_store_mode="lazy",
        embedding_dim=128,
    )

    assert result.data is prepared
    with pytest.raises(RuntimeError, match="size failure"):
        enzymado.pipeline.run_data.summarize(result.data)
