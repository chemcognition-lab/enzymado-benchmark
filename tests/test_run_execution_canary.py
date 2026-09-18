"""End-to-end canary for one deterministic benchmark model run."""

import json
from pathlib import Path

import numpy as np
import pytest
import scripts.run_benchmark

import enzymado.data_loading.data_registry
import enzymado.data_loading.task_data
import enzymado.evaluation.loaders
import enzymado.pipeline.run_data
import enzymado.training.torch_standard_wrappers
import enzymado.training.train


def _config(dataset_path: Path) -> enzymado.data_loading.data_registry.TaskConfig:
    """Build the fixed task contract used by the execution canary."""
    return enzymado.data_loading.data_registry.TaskConfig(
        name="ph_opt",
        batch_kind=enzymado.data_loading.data_registry.TaskType.PROTEIN_GLOBAL,
        label_kind=enzymado.data_loading.data_registry.LabelType.SCALAR_REGRESSION,
        dataset_paths={"similarity": dataset_path},
        target_column="target",
    )


def _prepared(
    config: enzymado.data_loading.data_registry.TaskConfig,
    cleanup: list[str],
) -> enzymado.data_loading.task_data.PooledTaskData:
    """Build a tiny pooled dataset with fixed sample order."""
    x_train = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-0.5, 0.25],
            [0.5, -0.25],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = 1.5 * x_train[:, 0] - 0.5 * x_train[:, 1]
    x_val = np.asarray([[-1.5, 0.5], [1.5, -0.5]], dtype=np.float32)
    y_val = 1.5 * x_val[:, 0] - 0.5 * x_val[:, 1]
    x_test = np.asarray([[-0.25, 1.0], [0.25, -1.0]], dtype=np.float32)
    y_test = 1.5 * x_test[:, 0] - 0.5 * x_test[:, 1]

    return enzymado.data_loading.task_data.PooledTaskData(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        target_mask_train=None,
        target_mask_val=None,
        target_mask_test=None,
        x_train_flat=None,
        y_train_flat=None,
        mask_train_flat=None,
        x_val_flat=None,
        y_val_flat=None,
        mask_val_flat=None,
        x_test_flat=None,
        y_test_flat=None,
        mask_test_flat=None,
        sample_ids_train=tuple(f"train-{index}" for index in range(6)),
        sample_ids_val=("val-0", "val-1"),
        sample_ids_test=("test-0", "test-1"),
        cleanup=lambda: cleanup.append("cleanup"),
    )


def test_single_model_run_matches_frozen_canary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lock one model's selection, configuration, outputs, and artifacts."""
    dataset_path = tmp_path / "canary.csv"
    dataset_path.write_text("sample_id,split,target\n", encoding="utf-8")
    embedding_path = tmp_path / "unused.safetensors"
    embedding_path.write_bytes(b"canary-embedding")
    config = _config(dataset_path)
    cleanup: list[str] = []
    prepared = _prepared(config, cleanup)
    prepare_calls: list[dict[str, object]] = []
    selection_calls: list[tuple[str, str, str | None]] = []
    model_info = {
        "wrap": enzymado.training.torch_standard_wrappers.LinearRegressorTorch,
        "params": {
            "lr": 0.05,
            "epochs": 4,
            "optimizer_name": "sgd",
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
    }

    def prepare(**kwargs: object) -> enzymado.pipeline.run_data.PreparedTask:
        prepare_calls.append(kwargs)
        return enzymado.pipeline.run_data.PreparedTask(
            data=prepared,
            route=enzymado.training.train.GLOBAL_ROUTE,
        )

    def select_models(
        task_type: str,
        label_kind: str,
        model_prefix: str | None,
    ) -> dict[str, dict[str, object]]:
        selection_calls.append((task_type, str(label_kind), model_prefix))
        return {"linear_regressor": model_info}

    monkeypatch.setattr(
        enzymado.data_loading.data_registry,
        "get_task_config",
        lambda _name: config,
    )
    monkeypatch.setattr(enzymado.pipeline.run_data, "prepare", prepare)

    scripts.run_benchmark._run_task(
        task_scope="properties",
        task_name="ph_opt",
        emb_name="canary_embedding",
        emb_info={"path": embedding_path, "dim": 2},
        output_dir=tmp_path,
        model_prefix_filter="linear",
        allowed_ids={"P1"},
        sequence_to_id={"SEQ": "P1"},
        skip_label_kinds=set(),
        residue_store_mode="lazy",
        split_variant_filter="similarity",
        embedding_mode_filters={"no_attention"},
        overwrite_results=True,
        random_seed=17,
        model_map_fn=select_models,
    )

    assert selection_calls == [("regression", "scalar_regression", "linear")]
    assert prepare_calls == [
        {
            "config": config,
            "task_name": "ph_opt",
            "emb_path": embedding_path,
            "split_variant": "similarity",
            "allowed_ids": {"P1"},
            "sequence_to_id": {"SEQ": "P1"},
            "residue_store_mode": "lazy",
            "embedding_dim": 2,
            "route_override": None,
        }
    ]
    assert cleanup == ["cleanup"]

    run_dir = (
        tmp_path
        / "properties"
        / "no_attention"
        / "similarity"
        / "linear"
        / "ph_opt"
        / "canary_embedding"
    )
    hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    assert hparams["model_name"] == "linear_regressor"
    assert hparams["split_variant"] == "similarity"
    assert hparams["random_seed"] == 17
    assert hparams["wrapper"] == "LinearRegressorTorch"
    assert hparams["params"] == model_info["params"]
    assert hparams["sample_identity"]["test"]["count"] == 2

    payload = enzymado.evaluation.loaders.load_test_predictions(run_dir)
    assert payload is not None
    np.testing.assert_array_equal(payload.y_true, [-0.875, 0.875])
    assert payload.sample_ids == ("test-0", "test-1")
    # Refrozen with the metric-based checkpoint selection; see the note in
    # tests/test_route_execution_canaries.py.
    np.testing.assert_allclose(
        payload.predictions,
        [0.5131318, 0.32744235],
        rtol=0.0,
        atol=1e-7,
    )

    artifact_names = {path.name for path in run_dir.iterdir()}
    stem = "linear_regressor_canary_embedding_ph_opt_similarity_no_attention"
    assert "hparams.json" in artifact_names
    assert "run_manifest.json" in artifact_names
    assert "training_log.csv" in artifact_names
    assert "model.pt" in artifact_names
    assert f"{stem}.safetensors" in artifact_names
    assert {
        name for name in artifact_names if name.endswith("_predictions.safetensors.xz")
    } == {
        f"{stem}_train_predictions.safetensors.xz",
        f"{stem}_val_predictions.safetensors.xz",
        f"{stem}_test_predictions.safetensors.xz",
    }
