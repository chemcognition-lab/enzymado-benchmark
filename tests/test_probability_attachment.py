"""A finished classification run must carry probabilities, not only hard labels.

AUROC and AUPRC are areas under a curve swept over a score. A confusion matrix
fixes one threshold, so reconstructing them from counts yields a degenerate lower
bound rather than the metric. The published protocol-v3 runs were scored that way
because probabilities were never written, so this locks the artifact contract:
training a classifier whose wrapper exposes ``predict_proba`` leaves a ``proba``
block beside the hard labels.
"""

import lzma
from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy
import scripts.run_benchmark

import enzymado.data_loading.data_registry
import enzymado.data_loading.task_data
import enzymado.pipeline.run_data
import enzymado.training.torch_standard_wrappers
import enzymado.training.train


def _config(dataset_path: Path) -> enzymado.data_loading.data_registry.TaskConfig:
    registry = enzymado.data_loading.data_registry
    return registry.TaskConfig(
        name="solubility",
        batch_kind=registry.TaskType.PROTEIN_GLOBAL,
        label_kind=registry.LabelType.BINARY,
        dataset_paths={"random": dataset_path},
        target_column="target",
    )


def _prepared(config, cleanup: list[str]):
    x = np.asarray(
        [[-2.0, -1.0], [-1.5, -1.0], [1.5, 1.0], [2.0, 1.0]], dtype=np.float32
    )
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    x_test = np.asarray([[-1.75, -1.0], [1.75, 1.0]], dtype=np.float32)
    y_test = np.asarray([0.0, 1.0], dtype=np.float32)
    return enzymado.data_loading.task_data.PooledTaskData(
        config=config,
        x_train=x,
        y_train=y,
        x_val=x_test,
        y_val=y_test,
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
        sample_ids_train=tuple(f"train-{index}" for index in range(4)),
        sample_ids_val=("test-0", "test-1"),
        sample_ids_test=("test-0", "test-1"),
        cleanup=lambda: cleanup.append("cleanup"),
    )


def test_training_a_classifier_writes_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A classifier run stores a probability block next to its hard labels."""
    dataset_path = tmp_path / "probe.csv"
    dataset_path.write_text("sample_id,split,target\n", encoding="utf-8")
    embedding_path = tmp_path / "unused.safetensors"
    embedding_path.write_bytes(b"probe-embedding")
    config = _config(dataset_path)
    cleanup: list[str] = []
    prepared = _prepared(config, cleanup)
    model_info = {
        "wrap": enzymado.training.torch_standard_wrappers.LinearClassifierTorch,
        "params": {
            "lr": 0.1,
            "epochs": 5,
            "optimizer_name": "sgd",
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            "device": "cpu",
        },
    }

    def prepare(**_kwargs: object) -> enzymado.pipeline.run_data.PreparedTask:
        return enzymado.pipeline.run_data.PreparedTask(
            data=prepared,
            route=enzymado.training.train.GLOBAL_ROUTE,
        )

    monkeypatch.setattr(
        enzymado.data_loading.data_registry,
        "get_task_config",
        lambda _name: config,
    )
    monkeypatch.setattr(enzymado.pipeline.run_data, "prepare", prepare)

    scripts.run_benchmark._run_task(
        task_scope="properties",
        task_name="solubility",
        emb_name="probe_embedding",
        emb_info={"path": embedding_path, "dim": 2},
        output_dir=tmp_path,
        model_prefix_filter="linear",
        allowed_ids={"P1"},
        sequence_to_id={"SEQ": "P1"},
        skip_label_kinds=set(),
        residue_store_mode="lazy",
        split_variant_filter="random",
        embedding_mode_filters={"no_attention"},
        overwrite_results=True,
        random_seed=17,
        model_map_fn=lambda *_a, **_k: {"linear_classification": model_info},
    )

    artifacts = sorted(tmp_path.rglob("*_test_predictions.safetensors.xz"))
    assert len(artifacts) == 1, f"expected one test artifact, found {artifacts}"
    with lzma.open(artifacts[0], "rb") as handle:
        tensors = safetensors.numpy.load(handle.read())

    proba_keys = [key for key in tensors if key.startswith("proba")]
    assert proba_keys, f"no probability block written; keys={sorted(tensors)}"

    # The hard labels must survive untouched beside the probabilities.
    assert "predictions" in tensors
    assert tensors[proba_keys[0]].shape == tensors["predictions"].shape
