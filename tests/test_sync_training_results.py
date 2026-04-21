from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_sync_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "enzyme_utils" / "sync_training_results.py"
    spec = importlib.util.spec_from_file_location("sync_training_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def write_run_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class SyncTrainingResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_repo = self.root / "source_repo"
        self.target_repo = self.root / "target_repo"
        self.module = load_sync_module()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sync_copies_run_dir_and_keeps_relative_paths(self) -> None:
        run_dir_rel = Path("results/predictions/random/binding/ankh_base/mlp_classification")
        for filename in ("train_predictions.npz", "val_predictions.npz", "test_predictions.npz", "model.pt"):
            write_run_file(self.source_repo / run_dir_rel / filename, filename)

        source_card = {
            "properties": {
                "binding": {
                    "training_results": {
                        "ankh_base__mlp_classification": {
                            "embedding_model": "ankh_base",
                            "name": "mlp_classification",
                            "params": {"epochs": 3},
                            "train_pred_file": str(run_dir_rel / "train_predictions.npz"),
                            "test_pred_file": str(run_dir_rel / "test_predictions.npz"),
                            "model_file": str(run_dir_rel / "model.pt"),
                        }
                    }
                }
            }
        }
        target_card = {"properties": {"binding": {"training_results": {}}}}
        source_card_path = self.source_repo / "benchmark_card.json"
        target_card_path = self.target_repo / "benchmark_card.json"
        write_json(source_card_path, source_card)
        write_json(target_card_path, target_card)

        stats = self.module.sync_training_results(
            source_card_path,
            target_card_path,
            source_root=self.source_repo,
            target_root=self.target_repo,
            copy_artifacts=True,
        )

        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["copied_run_dirs"], 1)

        merged_card = json.loads(target_card_path.read_text(encoding="utf-8"))
        merged_entry = merged_card["properties"]["binding"]["training_results"]["ankh_base__mlp_classification"]
        self.assertEqual(merged_entry["train_pred_file"], str(run_dir_rel / "train_predictions.npz"))
        self.assertEqual(merged_entry["test_pred_file"], str(run_dir_rel / "test_predictions.npz"))
        self.assertEqual(merged_entry["model_file"], str(run_dir_rel / "model.pt"))

        self.assertTrue((self.target_repo / run_dir_rel / "train_predictions.npz").exists())
        self.assertTrue((self.target_repo / run_dir_rel / "val_predictions.npz").exists())
        self.assertTrue((self.target_repo / run_dir_rel / "test_predictions.npz").exists())
        self.assertTrue((self.target_repo / run_dir_rel / "model.pt").exists())

    def test_sync_rewrites_absolute_source_paths_to_target_relative_paths(self) -> None:
        run_dir_rel = Path("results/predictions/random/km/esm2_650M/xgb_regression")
        model_path = self.source_repo / run_dir_rel / "model.joblib"
        train_path = self.source_repo / run_dir_rel / "train_predictions.npz"
        test_path = self.source_repo / run_dir_rel / "test_predictions.npz"
        for path in (model_path, train_path, test_path):
            write_run_file(path, path.name)

        source_card = {
            "properties": {
                "km": {
                    "training_results": {
                        "esm2_650M__xgb_regression": {
                            "embedding_model": "esm2_650M",
                            "name": "xgb_regression",
                            "params": {"depth": 6},
                            "train_pred_file": str(train_path),
                            "test_pred_file": str(test_path),
                            "model_file": str(model_path),
                        }
                    }
                }
            }
        }
        target_card = {"properties": {"km": {"training_results": {}}}}
        source_card_path = self.source_repo / "benchmark_card.json"
        target_card_path = self.target_repo / "benchmark_card.json"
        write_json(source_card_path, source_card)
        write_json(target_card_path, target_card)

        stats = self.module.sync_training_results(
            source_card_path,
            target_card_path,
            source_root=self.source_repo,
            target_root=self.target_repo,
            copy_artifacts=True,
        )

        self.assertEqual(stats["added"], 1)
        merged_card = json.loads(target_card_path.read_text(encoding="utf-8"))
        merged_entry = merged_card["properties"]["km"]["training_results"]["esm2_650M__xgb_regression"]
        self.assertEqual(merged_entry["model_file"], str(run_dir_rel / "model.joblib"))
        self.assertEqual(merged_entry["train_pred_file"], str(run_dir_rel / "train_predictions.npz"))
        self.assertTrue((self.target_repo / run_dir_rel / "model.joblib").exists())

    def test_sync_skips_existing_entries_without_overwrite(self) -> None:
        source_card = {
            "properties": {
                "binding": {
                    "training_results": {
                        "ankh3_large__sgd_classification": {
                            "embedding_model": "ankh3_large",
                            "name": "sgd_classification",
                            "params": {"lr": 0.1},
                            "train_pred_file": "results/predictions/foo/train_predictions.npz",
                            "test_pred_file": "results/predictions/foo/test_predictions.npz",
                            "model_file": "results/predictions/foo/model.pt",
                        }
                    }
                }
            }
        }
        target_card = {
            "properties": {
                "binding": {
                    "training_results": {
                        "ankh3_large__sgd_classification": {
                            "embedding_model": "ankh3_large",
                            "name": "sgd_classification",
                            "params": {"lr": 999},
                            "train_pred_file": "keep/train.npz",
                            "test_pred_file": "keep/test.npz",
                            "model_file": "keep/model.pt",
                        }
                    }
                }
            }
        }
        source_card_path = self.source_repo / "benchmark_card.json"
        target_card_path = self.target_repo / "benchmark_card.json"
        write_json(source_card_path, source_card)
        write_json(target_card_path, target_card)

        stats = self.module.sync_training_results(source_card_path, target_card_path)

        self.assertEqual(stats["skipped_existing"], 1)
        merged_card = json.loads(target_card_path.read_text(encoding="utf-8"))
        merged_entry = merged_card["properties"]["binding"]["training_results"]["ankh3_large__sgd_classification"]
        self.assertEqual(merged_entry["params"]["lr"], 999)
        self.assertEqual(merged_entry["model_file"], "keep/model.pt")


if __name__ == "__main__":
    unittest.main()
