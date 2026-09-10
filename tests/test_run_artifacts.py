"""Characterization tests for benchmark run artifact persistence."""

from pathlib import Path
import types
import typing

import numpy as np
import pytest

import enzymado.artifacts.serialization
import enzymado.data_loading.stores
import enzymado.data_loading.task_data
import enzymado.prediction.predict
import enzymado.prediction.run_artifacts
import enzymado.training.residue


def _pooled_data() -> enzymado.data_loading.task_data.PooledTaskData:
    """Build the sample-identity portion of pooled task data."""
    prepared = object.__new__(enzymado.data_loading.task_data.PooledTaskData)
    prepared.sample_ids_train = ("train-1", "train-2")
    prepared.sample_ids_val = ("val-1",)
    prepared.sample_ids_test = ("test-1", "test-2")
    return prepared


def _outputs() -> enzymado.prediction.predict.ModelRunOutputs:
    """Build deterministic targets, predictions, and masks for all splits."""
    return enzymado.prediction.predict.ModelRunOutputs(
        trained=object(),
        pred_train=np.asarray([10, 20]),
        pred_val=np.asarray([30]),
        pred_test=np.asarray([40, 50]),
        y_train=np.asarray([1, 2]),
        y_val=np.asarray([3]),
        y_test=np.asarray([4, 5]),
        target_mask_train=np.asarray([True, False]),
        target_mask_val=None,
        target_mask_test=np.asarray([False, True]),
    )


def test_split_predictions_preserve_payload_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lock split order, arrays, masks, IDs, context, and returned paths."""
    calls: list[dict[str, object]] = []

    def save_predictions(
        y_true: object,
        pred: object,
        target_mask: object,
        **kwargs: object,
    ) -> Path:
        calls.append(
            {
                "y_true": y_true,
                "pred": pred,
                "target_mask": target_mask,
                **kwargs,
            }
        )
        split_name = typing.cast(str, kwargs["split_name"])
        return tmp_path / f"{split_name}.artifact"

    monkeypatch.setattr(
        enzymado.artifacts.serialization,
        "save_raw_predictions",
        save_predictions,
    )
    paths = enzymado.prediction.run_artifacts.save_split_predictions(
        _outputs(),
        prepared=_pooled_data(),
        run_dir=tmp_path,
        task_name="binary_enzyme",
        emb_name="ankh_base",
        model_name="linear_classification",
        split_variant="similarity",
        use_validation=True,
    )

    assert tuple(call["split_name"] for call in calls) == ("train", "val", "test")
    assert tuple(paths) == ("train", "val", "test")
    assert paths == {
        "train": tmp_path / "train.artifact",
        "val": tmp_path / "val.artifact",
        "test": tmp_path / "test.artifact",
    }
    np.testing.assert_array_equal(calls[0]["y_true"], [1, 2])
    np.testing.assert_array_equal(calls[0]["pred"], [10, 20])
    np.testing.assert_array_equal(calls[0]["target_mask"], [True, False])
    assert calls[0]["sample_ids"] == ("train-1", "train-2")
    assert calls[1]["sample_ids"] == ("val-1",)
    assert calls[2]["sample_ids"] == ("test-1", "test-2")
    assert all(call["split_variant"] == "similarity" for call in calls)

    assert enzymado.prediction.run_artifacts.build_sample_manifest(_pooled_data()) == {
        "train": {
            "count": 2,
            "ordered_sha256": (
                "230f6a60a42534c16e86eef100ec104b2ff10f9b12d00d37ef7b3018f54997a5"
            ),
        },
        "val": {
            "count": 1,
            "ordered_sha256": (
                "38c7435f6d8b61d269ae72e07848dd58576055e105c4281b587b2dcdaaa5a1cb"
            ),
        },
        "test": {
            "count": 2,
            "ordered_sha256": (
                "9ba0348bc2b7144ff91243a695b423c6686c7fed15e61b56dd5a16ba98ef5460"
            ),
        },
    }


def test_validation_predictions_remain_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep train/test output order when validation is disabled."""
    split_names: list[str] = []

    def save_predictions(
        _y_true: object,
        _pred: object,
        _target_mask: object,
        **kwargs: object,
    ) -> Path:
        split_name = typing.cast(str, kwargs["split_name"])
        split_names.append(split_name)
        return tmp_path / f"{split_name}.artifact"

    monkeypatch.setattr(
        enzymado.artifacts.serialization,
        "save_raw_predictions",
        save_predictions,
    )
    paths = enzymado.prediction.run_artifacts.save_split_predictions(
        _outputs(),
        prepared=_pooled_data(),
        run_dir=tmp_path,
        task_name="binary_enzyme",
        emb_name="ankh_base",
        model_name="linear_classification",
        split_variant="similarity",
        use_validation=False,
    )

    assert split_names == ["train", "test"]
    assert tuple(paths) == ("train", "test")


def test_residue_predictions_preserve_split_order_and_batch_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lock streamed residue prediction order, loader settings, and sample IDs."""

    def build_bundle(
        sample_id: str,
    ) -> enzymado.data_loading.datasets.TaskDatasetBundle:
        bundle = object.__new__(enzymado.data_loading.datasets.TaskDatasetBundle)
        bundle.dataset = types.SimpleNamespace(
            rows=[{"prediction_id": sample_id}],
        )
        return bundle

    prepared = object.__new__(enzymado.data_loading.task_data.LoaderTaskData)
    prepared.train_bundle = build_bundle("train-1")
    prepared.val_bundle = build_bundle("val-1")
    prepared.test_bundle = build_bundle("test-1")

    loader_calls: list[tuple[str, str]] = []

    def predict_loader(loader: object, *, batch_kind: str) -> np.ndarray:
        loader_name = typing.cast(str, loader)
        loader_calls.append((loader_name, batch_kind))
        values = {
            "train-loader": [10],
            "val-loader": [20],
            "test-loader": [30],
        }
        return np.asarray(values[loader_name])

    trained_payload = enzymado.training.residue.ResidueTrainingPayload(
        trained=types.SimpleNamespace(predict_loader=predict_loader),
        loaders=types.SimpleNamespace(
            train_predict_loader="train-loader",
            val_loader="val-loader",
            test_loader="test-loader",
        ),
    )

    def get_target_payload(bundle: object) -> tuple[np.ndarray, np.ndarray]:
        typed_bundle = typing.cast(
            enzymado.data_loading.datasets.TaskDatasetBundle, bundle
        )
        sample_id = typed_bundle.dataset.rows[0]["prediction_id"]
        value = {"train-1": 1, "val-1": 2, "test-1": 3}[sample_id]
        return np.asarray([value]), np.asarray([True])

    saved: list[tuple[str, list[int], tuple[str, ...]]] = []

    def save_predictions(
        _y_true: object,
        pred: object,
        _target_mask: object,
        **kwargs: object,
    ) -> Path:
        split_name = typing.cast(str, kwargs["split_name"])
        pred_array = typing.cast(np.ndarray, pred)
        sample_ids = typing.cast(tuple[str, ...], kwargs["sample_ids"])
        saved.append((split_name, pred_array.tolist(), sample_ids))
        return tmp_path / f"{split_name}.artifact"

    monkeypatch.setattr(
        enzymado.data_loading.stores,
        "get_target_payload",
        get_target_payload,
    )
    monkeypatch.setattr(
        enzymado.artifacts.serialization,
        "save_raw_predictions",
        save_predictions,
    )
    paths = enzymado.prediction.run_artifacts.save_residue_predictions(
        trained_payload,
        prepared=prepared,
        use_validation=True,
        batch_kind="protein_residue",
        run_dir=tmp_path,
        task_name="binding_site",
        emb_name="ankh_base",
        model_name="linear_classification",
        split_variant="similarity",
    )

    assert loader_calls == [
        ("train-loader", "protein_residue"),
        ("val-loader", "protein_residue"),
        ("test-loader", "protein_residue"),
    ]
    assert saved == [
        ("train", [10], ("train-1",)),
        ("val", [20], ("val-1",)),
        ("test", [30], ("test-1",)),
    ]
    assert tuple(paths) == ("train", "val", "test")


def test_standard_model_formats_and_paths_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep native and safetensors outputs for standard training routes."""
    calls: list[tuple[str, Path | None]] = []

    def save_native(
        _trained: object,
        run_dir: Path,
        *,
        model_path: Path | None = None,
    ) -> Path:
        calls.append(("native", model_path))
        return model_path or run_dir / "model.joblib"

    def save_safetensors(
        _trained: object,
        _run_dir: Path,
        *,
        model_path: Path | None = None,
    ) -> Path | None:
        calls.append(("safetensors", model_path))
        return model_path

    monkeypatch.setattr(enzymado.artifacts.serialization, "save_model", save_native)
    monkeypatch.setattr(
        enzymado.artifacts.serialization,
        "save_model_safetensors",
        save_safetensors,
    )
    enzymado.prediction.run_artifacts.save_trained_model(
        object(),
        route_name="global",
        run_dir=tmp_path,
        task_name="binary_enzyme",
        emb_name="ankh_base",
        model_name="linear_classification",
        split_variant="similarity",
    )

    expected_path = (
        tmp_path / "linear_classification_ankh_base_binary_enzyme_similarity_"
        "no_attention.safetensors"
    )
    assert calls == [("native", None), ("safetensors", expected_path)]


def test_attention_model_format_and_path_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the canonical safetensors-only output for attention routes."""
    calls: list[tuple[str, Path | None]] = []

    def fail_native(
        _trained: object,
        _run_dir: Path,
        *,
        model_path: Path | None = None,
    ) -> Path:
        raise AssertionError(f"Unexpected native model save: {model_path}")

    def save_safetensors(
        _trained: object,
        _run_dir: Path,
        *,
        model_path: Path | None = None,
    ) -> Path | None:
        calls.append(("safetensors", model_path))
        return model_path

    monkeypatch.setattr(enzymado.artifacts.serialization, "save_model", fail_native)
    monkeypatch.setattr(
        enzymado.artifacts.serialization,
        "save_model_safetensors",
        save_safetensors,
    )
    enzymado.prediction.run_artifacts.save_trained_model(
        object(),
        route_name="attention",
        run_dir=tmp_path,
        task_name="binary_enzyme",
        emb_name="ankh_base",
        model_name="attention_linear_classification",
        split_variant="similarity",
    )

    expected_path = (
        tmp_path / "attention_linear_classification_ankh_base_binary_enzyme_similarity_"
        "attention.safetensors"
    )
    assert calls == [("safetensors", expected_path)]
