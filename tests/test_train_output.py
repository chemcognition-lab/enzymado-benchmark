from __future__ import annotations

import json
import lzma
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from evaluation.artifacts import (
    evaluate_predictions,
    make_safetensors_predictions_file,
    save_raw_predictions,
    write_hparams_file,
    write_metrics_file,
    write_training_log_file,
)


class TrainOutputTests(unittest.TestCase):
    def test_evaluate_predictions_regression(self) -> None:
        metrics = evaluate_predictions(
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([1.0, 2.5, 2.5], dtype=np.float32),
            "regression",
            split="val",
        )

        self.assertEqual(metrics["split"], "val")
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertIn("r2", metrics)
        self.assertIn("kendall_tau", metrics)

    def test_evaluate_predictions_classification_with_mask(self) -> None:
        metrics = evaluate_predictions(
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            np.asarray([[1, 1], [0, 0]], dtype=np.float32),
            "classification",
            np.asarray([[True, False], [True, True]], dtype=bool),
            split="test",
        )

        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)

    def test_experiment_artifact_helpers_write_self_contained_metadata(self) -> None:
        class FakeWrapper:
            pass

        class FakeTrained:
            training_log_ = [{"epoch": 1, "best_epoch": 1, "best_val_loss": 0.25}]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "target_audit_invalid.json").write_text("{}", encoding="utf-8")
            model_info = {"wrap": FakeWrapper, "params": {"epochs": 3, "lr": 0.01}}
            hparams_file = write_hparams_file(
                run_dir,
                task_name="phopt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                model_info=model_info,
                task_config=types.SimpleNamespace(
                    name="ph_opt",
                    batch_kind="protein_global",
                    label_kind="scalar_regression",
                    target_column="phopt_median_standardized",
                    dataset_paths={"random": run_dir / "phopt.csv"},
                ),
                sample_identity_manifest={
                    "test": {"count": 2, "ordered_sha256": "a" * 64}
                },
                random_seed=42,
            )
            metrics_file = write_metrics_file(run_dir, {"val": {"mae": 0.1}})
            training_log_file = write_training_log_file(FakeTrained(), run_dir)

            self.assertEqual(metrics_file, run_dir / "metrics.json")
            self.assertEqual(training_log_file, run_dir / "training_log.csv")
            self.assertEqual(json.loads(metrics_file.read_text())["val"]["mae"], 0.1)
            self.assertIn("epoch", training_log_file.read_text())
            hparams = json.loads(hparams_file.read_text())
            self.assertEqual(hparams["params"]["epochs"], 3)
            self.assertEqual(
                hparams["task_config"]["target_column"], "phopt_median_standardized"
            )
            self.assertEqual(hparams["sample_identity"]["test"]["count"], 2)
            self.assertEqual(hparams["random_seed"], 42)
            self.assertEqual(hparams["prediction_kind"], "value")
            self.assertFalse((run_dir / "target_audit_invalid.json").exists())

    def test_hparams_records_probability_classification_predictions(self) -> None:
        class FakeWrapper:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            hparams_file = write_hparams_file(
                run_dir,
                task_name="solubility",
                emb_name="ankh_base",
                model_name="linear_classifier",
                split_variant="random",
                model_info={"wrap": FakeWrapper, "params": {}},
                task_config=types.SimpleNamespace(
                    name="solubility",
                    batch_kind="protein_global",
                    label_kind="binary",
                    target_column="solubility",
                    dataset_paths={},
                ),
            )

            hparams = json.loads(hparams_file.read_text())
            self.assertEqual(hparams["prediction_kind"], "probability")

    def test_safetensors_prediction_file_is_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = make_safetensors_predictions_file(
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.5, 1.8], dtype=np.float32),
                run_dir,
                "val",
            )

            from safetensors.torch import load as load_tensors

            payload = load_tensors(lzma.open(path, "rb").read())
            self.assertIn("y_true", payload)
            self.assertIn("predictions", payload)

    def test_save_raw_predictions_use_safetensors_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pred_file = save_raw_predictions(
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.5, 1.8], dtype=np.float32),
                None,
                run_dir=run_dir,
                task_name="phopt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                split_name="test",
            )

            self.assertEqual(pred_file.suffixes[-2:], [".safetensors", ".xz"])
            self.assertTrue(pred_file.exists())

    def test_sample_ids_round_trip_inside_prediction_artifact(self) -> None:
        from evaluation import loaders, sample_identity

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pred_file = save_raw_predictions(
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.5, 1.8], dtype=np.float32),
                None,
                run_dir=run_dir,
                task_name="ph_opt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                split_name="test",
                sample_ids=("P1", "proteína-2"),
            )

            from safetensors.torch import load as load_tensors

            tensors = load_tensors(lzma.open(pred_file, "rb").read())
            self.assertIn(sample_identity.SAMPLE_ID_BYTES_KEY, tensors)
            self.assertIn(sample_identity.SAMPLE_ID_OFFSETS_KEY, tensors)
            payload = loaders.load_test_predictions(run_dir)
            self.assertIsNotNone(payload)
            self.assertEqual(payload.sample_ids, ("P1", "proteína-2"))

    def test_prediction_writer_rejects_mismatched_sample_id_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "1 sample IDs for 2"):
                make_safetensors_predictions_file(
                    np.asarray([1.0, 2.0], dtype=np.float32),
                    np.asarray([1.5, 1.8], dtype=np.float32),
                    Path(tmp),
                    "test",
                    sample_ids=("P1",),
                )

    def test_save_raw_predictions_carries_no_metrics(self) -> None:
        """save_raw_predictions is I/O only; evaluate_predictions computes metrics separately."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pred_file = save_raw_predictions(
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.5, 1.8], dtype=np.float32),
                None,
                run_dir=run_dir,
                task_name="phopt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                split_name="test",
            )

            from safetensors.torch import load as load_tensors

            payload = load_tensors(lzma.open(pred_file, "rb").read())
            metrics = evaluate_predictions(
                payload["y_true"].numpy(),
                payload["predictions"].numpy(),
                "regression",
                split="test",
            )
            self.assertEqual(metrics["split"], "test")
            self.assertEqual(metrics["n"], 2)


if __name__ == "__main__":
    unittest.main()
