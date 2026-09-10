from __future__ import annotations

import json
import lzma
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from enzymado.artifacts.serialization import (
    make_safetensors_predictions_file,
    save_raw_predictions,
    write_hparams_file,
    write_training_log_file,
)
from enzymado.evaluation import binary_statistics, metrics_registry
from enzymado.evaluation.loaders import PredictionPayload
from enzymado.evaluation.metrics_registry import evaluate_predictions


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
        """Masking drops the unknown entry; accuracy comes from the count path.

        ``evaluate_predictions`` still reports ``n`` for classification because
        the flattening is shared, but it no longer computes the metrics: those
        are derived from confusion counts in production, so the value is
        asserted there.
        """
        y_true = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        predictions = np.asarray([[1, 1], [0, 0]], dtype=np.float32)
        mask = np.asarray([[True, False], [True, True]], dtype=bool)

        metrics = evaluate_predictions(
            y_true,
            predictions,
            "classification",
            mask,
            split="test",
        )
        self.assertEqual(metrics["n"], 3)
        self.assertNotIn("accuracy", metrics)

        counts = binary_statistics.count_payload(
            PredictionPayload(y_true, predictions, mask)
        )
        self.assertAlmostEqual(
            metrics_registry.evaluate(counts, split="test")["accuracy"], 2.0 / 3.0
        )

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
            training_log_file = write_training_log_file(FakeTrained(), run_dir)

            self.assertEqual(training_log_file, run_dir / "training_log.csv")
            self.assertIn("epoch", training_log_file.read_text())
            hparams = json.loads(hparams_file.read_text())
            self.assertEqual(hparams["params"]["epochs"], 3)
            self.assertEqual(
                hparams["task_config"]["target_column"], "phopt_median_standardized"
            )
            self.assertEqual(hparams["sample_identity"]["test"]["count"], 2)
            self.assertEqual(hparams["random_seed"], 42)
            self.assertFalse((run_dir / "target_audit_invalid.json").exists())

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
        from enzymado.artifacts import sample_identity
        from enzymado.evaluation import loaders

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
        """save_raw_predictions is I/O only; metrics are computed separately."""
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
