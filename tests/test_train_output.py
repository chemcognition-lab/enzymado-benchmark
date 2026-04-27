from __future__ import annotations

import json
import lzma
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.train_output import (
    evaluate_predictions,
    make_result_entry,
    make_safetensors_predictions_file,
    save_prediction_artifacts,
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

    def test_experiment_artifact_helpers_are_linked_from_result_entry(self) -> None:
        class FakeWrapper:
            pass

        class FakeTrained:
            training_log_ = [{"epoch": 1, "best_epoch": 1, "best_val_loss": 0.25}]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            model_info = {"wrap": FakeWrapper, "params": {"epochs": 3, "lr": 0.01}}
            hparams_file = write_hparams_file(
                run_dir,
                task_name="phopt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                model_info=model_info,
            )
            metrics_file = write_metrics_file(run_dir, {"val": {"mae": 0.1}})
            training_log_file = write_training_log_file(FakeTrained(), run_dir)

            entry = make_result_entry(
                "ankh_base",
                "mlp_regressor",
                model_info,
                run_dir,
                run_dir / "model.pt",
                task_name="phopt",
                split_variant="random",
                train_pred_safetensors_file=run_dir / "train_predictions.safetensors.xz",
                test_pred_safetensors_file=run_dir / "test_predictions.safetensors.xz",
                model_safetensors_file=run_dir / "model.safetensors",
                metrics={"val": {"mae": 0.1}},
                metrics_file=metrics_file,
                hparams_file=hparams_file,
                training_log_file=training_log_file,
            )

            self.assertEqual(entry["metrics"]["val"]["mae"], 0.1)
            self.assertEqual(entry["hparams_file"], str(hparams_file))
            self.assertEqual(entry["metrics_file"], str(metrics_file))
            self.assertEqual(entry["training_log_file"], str(training_log_file))
            self.assertEqual(
                entry["train_pred_safetensors_file"],
                str(run_dir / "train_predictions.safetensors.xz"),
            )
            self.assertEqual(
                entry["train_pred_file"],
                str(run_dir / "train_predictions.safetensors.xz"),
            )
            self.assertEqual(entry["model_safetensors_file"], str(run_dir / "model.safetensors"))
            self.assertEqual(json.loads(hparams_file.read_text())["params"]["epochs"], 3)

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

    def test_prediction_artifacts_use_safetensors_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pred_file, metrics = save_prediction_artifacts(
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.5, 1.8], dtype=np.float32),
                None,
                run_dir=run_dir,
                task_name="phopt",
                emb_name="ankh_base",
                model_name="mlp_regressor",
                split_variant="random",
                split_name="test",
                task_type="regression",
            )

            self.assertEqual(pred_file.suffixes[-2:], [".safetensors", ".xz"])
            self.assertTrue(pred_file.exists())
            self.assertEqual(metrics["split"], "test")


if __name__ == "__main__":
    unittest.main()
