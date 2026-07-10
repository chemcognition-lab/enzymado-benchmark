from __future__ import annotations

import json
import lzma
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from evaluation.artifacts import (
    evaluate_predictions,
    make_result_entry,
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

    def test_experiment_artifact_helpers_are_linked_from_result_entry(self) -> None:
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
            hparams = json.loads(hparams_file.read_text())
            self.assertEqual(hparams["params"]["epochs"], 3)
            self.assertEqual(hparams["task_config"]["target_column"], "phopt_median_standardized")
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
