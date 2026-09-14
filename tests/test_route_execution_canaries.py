"""Small end-to-end canaries for every benchmark training route."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import enzymado.artifacts.results_layout
import enzymado.data_loading.collators
import enzymado.data_loading.data_registry
import enzymado.data_loading.datasets
import enzymado.data_loading.labels
import enzymado.data_loading.task_data
import enzymado.evaluation.loaders
import enzymado.pipeline.run_data
import enzymado.pipeline.run_execution
import enzymado.training.attention_wrappers
import enzymado.training.torch_structured_wrappers
import enzymado.training.train
import enzymado.training.xgb_wrappers


class _StaticDataset:
    """Expose fixed rows and samples through the dataset protocol."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        samples: list[dict[str, Any]],
    ) -> None:
        self.rows = rows
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


def _config(
    name: str,
    batch_kind: enzymado.data_loading.data_registry.TaskType,
    label_kind: enzymado.data_loading.data_registry.LabelType,
    *,
    target_column: str | None,
    residue_positions_column: str | None = None,
) -> enzymado.data_loading.data_registry.TaskConfig:
    """Build one synthetic task configuration using canonical similarity naming."""
    return enzymado.data_loading.data_registry.TaskConfig(
        name=name,
        batch_kind=batch_kind,
        label_kind=label_kind,
        dataset_paths={"similarity": Path("canary.csv")},
        target_column=target_column,
        residue_positions_column=residue_positions_column,
    )


def _global_bundle(
    config: enzymado.data_loading.data_registry.TaskConfig,
    split_name: str,
    embeddings: list[list[list[float]]],
    targets: list[float],
) -> enzymado.data_loading.datasets.TaskDatasetBundle:
    """Build a loader-ready scalar bundle without reading repository data."""
    label_builder = enzymado.data_loading.labels.RegressionLabelBuilder("target")
    rows = [
        {
            "sample_id": f"{split_name}-{index}",
            "target": str(target),
        }
        for index, target in enumerate(targets)
    ]
    samples = [
        {
            "sample_id": row["sample_id"],
            "protein_emb": torch.tensor(embedding, dtype=torch.float32),
            "target": label_builder.build(row),
        }
        for row, embedding in zip(rows, embeddings, strict=True)
    ]
    return enzymado.data_loading.datasets.TaskDatasetBundle(
        config=config,
        rows=rows,
        dataset=_StaticDataset(rows, samples),  # type: ignore[arg-type]
        collator=enzymado.data_loading.collators.ProteinGlobalCollator(),
        label_builder=label_builder,
    )


def _residue_bundle(
    config: enzymado.data_loading.data_registry.TaskConfig,
    split_name: str,
    embeddings: list[list[list[float]]],
    positions: list[str],
) -> enzymado.data_loading.datasets.TaskDatasetBundle:
    """Build a loader-ready residue bundle with variable sequence lengths."""
    label_builder = enzymado.data_loading.labels.ResidueBinaryVectorLabelBuilder(
        positions_column="positions"
    )
    rows = [
        {
            "sample_id": f"{split_name}-{index}",
            "protein_sequence": "A" * len(embedding),
            "positions": position,
        }
        for index, (embedding, position) in enumerate(
            zip(embeddings, positions, strict=True)
        )
    ]
    samples = [
        {
            "sample_id": row["sample_id"],
            "protein_emb": torch.tensor(embedding, dtype=torch.float32),
            "target_per_residue": label_builder.build(row),
        }
        for row, embedding in zip(rows, embeddings, strict=True)
    ]
    return enzymado.data_loading.datasets.TaskDatasetBundle(
        config=config,
        rows=rows,
        dataset=_StaticDataset(rows, samples),  # type: ignore[arg-type]
        collator=enzymado.data_loading.collators.ProteinResidueCollator(),
        label_builder=label_builder,
    )


def _loader_data(
    config: enzymado.data_loading.data_registry.TaskConfig,
    train_bundle: enzymado.data_loading.datasets.TaskDatasetBundle,
    val_bundle: enzymado.data_loading.datasets.TaskDatasetBundle,
    test_bundle: enzymado.data_loading.datasets.TaskDatasetBundle,
) -> enzymado.data_loading.task_data.LoaderTaskData:
    """Join three fixed bundles into the loader-task contract."""
    return enzymado.data_loading.task_data.LoaderTaskData(
        config=config,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        cleanup=lambda: None,
    )


def _pooled_data(
    config: enzymado.data_loading.data_registry.TaskConfig,
    x_train: np.ndarray,
    y_train: Any,
    x_val: np.ndarray,
    y_val: Any,
    x_test: np.ndarray,
    y_test: Any,
    *,
    masks: tuple[Any, Any, Any] = (None, None, None),
) -> enzymado.data_loading.task_data.PooledTaskData:
    """Build the array-based data contract used by global models."""
    mask_train, mask_val, mask_test = masks
    return enzymado.data_loading.task_data.PooledTaskData(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        target_mask_train=mask_train,
        target_mask_val=mask_val,
        target_mask_test=mask_test,
        x_train_flat=None,
        y_train_flat=None,
        mask_train_flat=None,
        x_val_flat=None,
        y_val_flat=None,
        mask_val_flat=None,
        x_test_flat=None,
        y_test_flat=None,
        mask_test_flat=None,
        sample_ids_train=tuple(f"train-{index}" for index in range(len(x_train))),
        sample_ids_val=tuple(f"val-{index}" for index in range(len(x_val))),
        sample_ids_test=tuple(f"test-{index}" for index in range(len(x_test))),
        cleanup=lambda: None,
    )


def _run(
    tmp_path: Path,
    config: enzymado.data_loading.data_registry.TaskConfig,
    prepared: enzymado.data_loading.task_data.PooledTaskData
    | enzymado.data_loading.task_data.LoaderTaskData,
    route: enzymado.training.train.TrainingRouteSpec,
    model_name: str,
    model_info: dict[str, object],
    *,
    seed: int,
    attention_overrides: dict[str, object] | None = None,
) -> Path:
    """Execute one canary and return its canonical temporary run directory."""
    dataset_path = tmp_path / f"{config.name}_canary.csv"
    dataset_path.write_text("sample_id,split,target\n", encoding="utf-8")
    embedding_path = tmp_path / "canary_embedding.safetensors"
    embedding_path.write_bytes(b"canary-embedding")
    context = enzymado.pipeline.run_execution.RunContext(
        output_dir=tmp_path,
        task_scope="properties",
        task_name=config.name,
        emb_name="canary_embedding",
        split_variant="similarity",
        config=config,
        prepared=prepared,
        route=route,
        sizes=enzymado.pipeline.run_data.summarize(prepared),
        overwrite_results=True,
        random_seed=seed,
        dataset_path=dataset_path,
        embedding_path=embedding_path,
        attention_overrides=attention_overrides,
    )
    enzymado.pipeline.run_execution.run(context, model_name, model_info)
    return enzymado.artifacts.results_layout.build_results_run_dir(
        tmp_path,
        "similarity",
        config.name,
        "canary_embedding",
        model_name,
        task_scope="properties",
    )


def _assert_common_artifacts(
    run_dir: Path,
    *,
    wrapper_name: str,
    expected_test_ids: tuple[str, ...],
) -> enzymado.evaluation.loaders.PredictionPayload:
    """Validate metadata shared by all four route canaries."""
    hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    assert hparams["split_variant"] == "similarity"
    assert hparams["wrapper"] == wrapper_name
    assert hparams["sample_identity"]["test"]["count"] == len(expected_test_ids)
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "training_log.csv").is_file()

    payload = enzymado.evaluation.loaders.load_test_predictions(run_dir)
    assert payload is not None
    assert payload.sample_ids == expected_test_ids
    return payload


def test_residue_route_matches_frozen_canary(tmp_path: Path) -> None:
    """Lock residue padding, target masks, predictions, and model artifacts."""
    config = _config(
        "active_sites",
        enzymado.data_loading.data_registry.TaskType.PROTEIN_RESIDUE,
        enzymado.data_loading.data_registry.LabelType.RESIDUE_BINARY_VECTOR,
        target_column=None,
        residue_positions_column="positions",
    )
    train_embeddings = [
        [[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5]],
        [[1.5, 1.0], [2.0, 0.5], [-1.0, -1.5], [-0.5, -1.0]],
        [[-1.5, 1.0], [0.5, -1.0], [1.5, 1.0]],
        [[2.0, -1.0], [-2.0, 1.0], [1.0, -0.5], [-1.0, 0.5]],
    ]
    val_embeddings = [
        [[-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
    ]
    test_embeddings = [
        [[-0.75, -0.25], [0.75, 0.25], [1.5, 0.75]],
        [[1.25, -0.75], [-1.25, 0.75], [0.25, -0.25], [-0.25, 0.25]],
    ]
    prepared = _loader_data(
        config,
        _residue_bundle(
            config, "train", train_embeddings, ["[3]", "[1, 2]", "[2]", "[1, 3]"]
        ),
        _residue_bundle(config, "val", val_embeddings, ["[2, 3]", "[1, 4]"]),
        _residue_bundle(config, "test", test_embeddings, ["[2, 3]", "[1, 4]"]),
    )
    model_info = {
        "wrap": (
            enzymado.training.torch_structured_wrappers.ResidueBinaryClassifierTorch
        ),
        "params": {
            "hidden_dims": (),
            "lr": 0.05,
            "epochs": 4,
            "optimizer_name": "sgd",
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
        "loader_batch_size": 2,
        "loader_num_workers": 0,
        "residue_aware": True,
    }
    run_dir = _run(
        tmp_path,
        config,
        prepared,
        enzymado.training.train.RESIDUE_ROUTE,
        "linear_classification",
        model_info,
        seed=19,
    )

    payload = _assert_common_artifacts(
        run_dir,
        wrapper_name="ResidueBinaryClassifierTorch",
        expected_test_ids=("test-0", "test-1"),
    )
    np.testing.assert_array_equal(
        payload.y_true,
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(
        payload.target_mask,
        [[True, True, True, False], [True, True, True, True]],
    )
    np.testing.assert_array_equal(
        payload.predictions,
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]],
    )
    assert (run_dir / "model.pt").is_file()
    assert len(list(run_dir.glob("*.safetensors"))) == 1


def test_attention_route_matches_frozen_canary(tmp_path: Path) -> None:
    """Lock attention overrides, sequence pooling, outputs, and artifacts."""
    config = _config(
        "ph_opt_attention_canary",
        enzymado.data_loading.data_registry.TaskType.PROTEIN_GLOBAL,
        enzymado.data_loading.data_registry.LabelType.SCALAR_REGRESSION,
        target_column="target",
    )
    train_embeddings = [
        [[-2.0, -1.0], [-1.5, -0.5], [-1.0, 0.0]],
        [[-1.0, -2.0], [-0.5, -1.5], [0.0, -1.0], [0.5, -0.5]],
        [[0.5, 0.0], [1.0, 0.5], [1.5, 1.0]],
        [[1.0, 2.0], [1.5, 1.5], [2.0, 1.0], [2.5, 0.5]],
    ]
    val_embeddings = [
        [[-1.5, 0.5], [-1.0, 0.0], [-0.5, -0.5]],
        [[0.5, -0.5], [1.0, 0.0], [1.5, 0.5]],
    ]
    test_embeddings = [
        [[-0.75, 0.75], [-0.25, 0.25], [0.25, -0.25]],
        [[0.25, -0.75], [0.75, -0.25], [1.25, 0.25], [1.75, 0.75]],
    ]
    prepared = _loader_data(
        config,
        _global_bundle(config, "train", train_embeddings, [-2.0, -1.0, 1.0, 2.0]),
        _global_bundle(config, "val", val_embeddings, [-1.25, 1.25]),
        _global_bundle(config, "test", test_embeddings, [-0.75, 0.75]),
    )
    model_info = {
        "wrap": enzymado.training.attention_wrappers.AttentionRegressorTorch,
        "params": {
            "hidden_dims": (),
            "lr": 0.03,
            "epochs": 4,
            "optimizer_name": "sgd",
            "num_attention_heads": 1,
            "attention_dropout_rate": 0.0,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
        "loader_batch_size": 2,
        "loader_num_workers": 0,
        "residue_aware": True,
    }
    run_dir = _run(
        tmp_path,
        config,
        prepared,
        enzymado.training.train.ATTENTION_ROUTE,
        "attention_linear_regressor",
        model_info,
        seed=29,
        attention_overrides={"gradient_clip_norm": 0.75},
    )

    payload = _assert_common_artifacts(
        run_dir,
        wrapper_name="AttentionRegressorTorch",
        expected_test_ids=("test-0", "test-1"),
    )
    np.testing.assert_array_equal(payload.y_true, [-0.75, 0.75])
    # The initial Kendall score is perfect; ties must select trained epoch 1.
    history = list(csv.DictReader((run_dir / "training_log.csv").open()))
    assert history[-1]["best_epoch"] == "1"
    assert history[-1]["improved_over_initialization"] == "False"
    np.testing.assert_allclose(
        payload.predictions,
        [0.55078125, 0.66015625],
        rtol=0.0,
        atol=1e-7,
    )
    hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    assert hparams["params"]["gradient_clip_norm"] == 0.75
    assert not (run_dir / "model.pt").exists()
    assert len(list(run_dir.glob("*.safetensors"))) == 1


def _ec_targets() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Return fixed train, validation, and test targets for four EC levels."""
    train_pattern = np.asarray(
        [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=np.float32,
    )
    val_pattern = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    test_pattern = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    return (
        {
            f"level_{level}": np.roll(train_pattern, level % 2, axis=1)
            for level in range(1, 5)
        },
        {
            f"level_{level}": np.roll(val_pattern, level % 2, axis=1)
            for level in range(1, 5)
        },
        {
            f"level_{level}": np.roll(test_pattern, level % 2, axis=1)
            for level in range(1, 5)
        },
    )


def test_multilabel_route_matches_frozen_canary(tmp_path: Path) -> None:
    """Lock four EC heads, dense reconstruction, masks, and predictions."""
    config = _config(
        "ec_number",
        enzymado.data_loading.data_registry.TaskType.PROTEIN_GLOBAL,
        enzymado.data_loading.data_registry.LabelType.MULTILABEL_EC,
        target_column="ec_number",
    )
    x_train = np.asarray(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, -2.0, 0.5],
            [-0.5, 0.5, -1.0],
            [0.5, -0.5, 1.0],
            [1.0, 2.0, -0.5],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    x_val = np.asarray([[-1.5, 0.5, 0.25], [1.5, -0.5, -0.25]], dtype=np.float32)
    x_test = np.asarray([[-0.25, 1.0, -0.5], [0.25, -1.0, 0.5]], dtype=np.float32)
    y_train, y_val, y_test = _ec_targets()
    masks = (
        {key: np.ones_like(value, dtype=bool) for key, value in y_train.items()},
        {key: np.ones_like(value, dtype=bool) for key, value in y_val.items()},
        {key: np.ones_like(value, dtype=bool) for key, value in y_test.items()},
    )
    prepared = _pooled_data(
        config,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        masks=masks,
    )
    model_info = {
        "wrap": enzymado.training.torch_structured_wrappers.ECMultiHeadClassifierTorch,
        "params": {
            "hidden_dims": (),
            "lr": 0.05,
            "epochs": 4,
            "optimizer_name": "sgd",
            "weight_decay": 0.0,
            "dropout": 0.0,
            "batch_size": 3,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
        "residue_aware": False,
    }
    run_dir = _run(
        tmp_path,
        config,
        prepared,
        enzymado.training.train.GLOBAL_ROUTE,
        "linear_classification",
        model_info,
        seed=31,
    )

    payload = _assert_common_artifacts(
        run_dir,
        wrapper_name="ECMultiHeadClassifierTorch",
        expected_test_ids=("test-0", "test-1"),
    )
    assert payload.y_true.shape == (2, 8)
    np.testing.assert_array_equal(payload.target_mask, np.ones((2, 8), dtype=bool))
    blocks = enzymado.evaluation.loaders.load_split_blocks(run_dir, "test")
    assert blocks is not None
    assert len(blocks) == 4
    np.testing.assert_array_equal(
        payload.predictions,
        [
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )
    assert (run_dir / "model.pt").is_file()
    assert len(list(run_dir.glob("*.safetensors"))) == 1


def test_xgboost_route_matches_frozen_canary(tmp_path: Path) -> None:
    """Lock native XGBoost seeding, regression outputs, and joblib artifacts."""
    assert enzymado.training.xgb_wrappers.XGBRegressor is not None
    config = _config(
        "ph_opt_xgboost_canary",
        enzymado.data_loading.data_registry.TaskType.PROTEIN_GLOBAL,
        enzymado.data_loading.data_registry.LabelType.SCALAR_REGRESSION,
        target_column="target",
    )
    x_train = np.asarray(
        [
            [-2.0, -1.0],
            [-1.5, 0.5],
            [-1.0, -2.0],
            [-0.5, 1.0],
            [0.5, -1.0],
            [1.0, 2.0],
            [1.5, -0.5],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = 1.25 * x_train[:, 0] - 0.75 * x_train[:, 1]
    x_val = np.asarray([[-1.25, 0.25], [1.25, -0.25]], dtype=np.float32)
    y_val = 1.25 * x_val[:, 0] - 0.75 * x_val[:, 1]
    x_test = np.asarray([[-0.25, 1.0], [0.25, -1.0]], dtype=np.float32)
    y_test = 1.25 * x_test[:, 0] - 0.75 * x_test[:, 1]
    prepared = _pooled_data(
        config,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )
    model_info = {
        "wrap": enzymado.training.xgb_wrappers.XGBRegressor,
        "params": {
            "n_estimators": 6,
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_jobs": 1,
            "device": "cpu",
            "tree_method": "hist",
        },
        "residue_aware": False,
    }
    run_dir = _run(
        tmp_path,
        config,
        prepared,
        enzymado.training.train.GLOBAL_ROUTE,
        "xgb_regression",
        model_info,
        seed=37,
    )

    payload = _assert_common_artifacts(
        run_dir,
        wrapper_name="XGBRegressor",
        expected_test_ids=("test-0", "test-1"),
    )
    np.testing.assert_allclose(payload.y_true, [-1.0625, 1.0625], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        payload.predictions,
        [-1.1159404516220093, -1.1159404516220093],
        rtol=0.0,
        atol=1e-7,
    )
    hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    assert hparams["params"]["random_state"] == 37
    assert (run_dir / "model.joblib").is_file()
    assert not list(run_dir.glob("*.safetensors"))
