from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_migration_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "enzyme_utils" / "migrate_results_card.py"
    spec = importlib.util.spec_from_file_location("migrate_results_card", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_results_card_module():
    module_path = Path(__file__).resolve().parents[1] / "models" / "results_card.py"
    spec = importlib.util.spec_from_file_location("results_card_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ResultsCardTests(unittest.TestCase):
    def test_build_result_run_dir_uses_canonical_layout(self) -> None:
        module = load_results_card_module()
        build_result_run_dir = module.build_results_run_dir
        root = Path("/tmp/results")
        self.assertEqual(
            build_result_run_dir(root, "random", "binding", "ankh_base", "sgd_classification"),
            root / "mean" / "random" / "linear" / "binding" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(root, "random", "ddg", "ankh_base", "linear_regressor"),
            root / "mean" / "random" / "linear" / "ddg" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(root, "random", "solubility", "ankh_base", "linear_classification"),
            root / "mean" / "random" / "linear" / "solubility" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(root, "similarity", "kcat", "esm2_650M", "attention_mlp_regressor"),
            root / "attention" / "loco" / "mlp" / "kcat" / "esm2_650M",
        )
        self.assertEqual(
            build_result_run_dir(root, "random", "active_sites", "prot5_xl_uniref50", "xgb_classification"),
            root / "per_residue" / "random" / "xgboost" / "active_sites" / "prot5_xl_uniref50",
        )

    def test_migration_prefers_existing_complete_entry(self) -> None:
        module = load_migration_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_card_path = root / "benchmark_card.json"
            source_main = root / "legacy_main.json"
            source_mlp = root / "legacy_mean_mlp.json"
            output_path = root / "results_card.json"

            benchmark_card = {
                "benchmark_name": "enzymado",
                "benchmark_version": "test",
                "properties": {
                    "binding": {
                        "task_type": "classification",
                        "datasets": [{"splits": {"default_split": "random_80_10_10"}}],
                    }
                },
                "real_case": {"cleeveland_galox": {}},
            }
            stale_entry = {
                "embedding_model": "ankh_base",
                "name": "mlp_classification",
                "params": {"epochs": 200},
                "train_pred_file": str(root / "results" / "predictions" / "old" / "train_predictions.npz"),
                "test_pred_file": str(root / "results" / "predictions" / "old" / "test_predictions.npz"),
                "model_file": str(root / "results" / "predictions" / "old" / "model.pt"),
            }
            preferred_dir = root / "results" / "predictions_mean_pooled_mlp" / "ankh_base__binding__mlp_classification"
            for filename in ("train_predictions.npz", "val_predictions.npz", "test_predictions.npz", "model.pt"):
                write_file(preferred_dir / filename)
            preferred_entry = {
                "embedding_model": "ankh_base",
                "name": "mlp_classification",
                "params": {
                    "epochs": 200,
                    "early_stopping_patience": 20,
                    "early_stopping_min_delta": 1e-4,
                },
                "train_pred_file": str(preferred_dir / "train_predictions.npz"),
                "test_pred_file": str(preferred_dir / "test_predictions.npz"),
                "model_file": str(preferred_dir / "model.pt"),
            }

            write_json(benchmark_card_path, benchmark_card)
            write_json(
                source_main,
                {"properties": {"binding": {"training_results": {"ankh_base__mlp_classification": stale_entry}}}},
            )
            write_json(
                source_mlp,
                {"properties": {"binding": {"training_results": {"ankh_base__mlp_classification": preferred_entry}}}},
            )

            stats = module.migrate_results_card(
                benchmark_card_path=benchmark_card_path,
                source_card_paths=[source_main, source_mlp],
                output_path=output_path,
            )

            self.assertEqual(stats["selected_entries"], 1)
            results_card = json.loads(output_path.read_text(encoding="utf-8"))
            entry = results_card["predictions"]["mean"]["random"]["mlp"]["binding"]["ankh_base"]
            self.assertEqual(entry["model_file"], str(preferred_dir / "model.pt"))
            self.assertEqual(entry["val_pred_file"], str(preferred_dir / "val_predictions.npz"))
            self.assertEqual(entry["params"]["early_stopping_patience"], 20)

    def test_artifact_paths_use_full_run_context(self) -> None:
        module = load_results_card_module()
        run_dir = Path("/tmp/results/mean/random/mlp/binding/ankh_base")
        self.assertEqual(
            module.build_prediction_artifact_path(
                run_dir,
                task_name="binding",
                emb_name="ankh_base",
                model_name="mlp_classification",
                split_variant="random",
                split_name="test",
            ),
            run_dir
            / "mlp_classification_ankh_base_binding_random_mean_test_predictions.safetensors.xz",
        )
        self.assertEqual(
            module.build_model_artifact_path(
                run_dir,
                task_name="binding",
                emb_name="ankh_base",
                model_name="mlp_classification",
                split_variant="random",
                extension=".joblib",
            ),
            run_dir / "mlp_classification_ankh_base_binding_random_mean.joblib",
        )

    def test_named_artifact_helpers_accept_canonical_filenames(self) -> None:
        results_card_module = load_results_card_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "results" / "predictions" / "mean" / "random" / "xgboost" / "ec_number" / "ankh_base"
            write_file(
                results_card_module.build_model_artifact_path(
                    run_dir,
                    task_name="ec_number",
                    emb_name="ankh_base",
                    model_name="xgb_classification",
                    split_variant="random",
                    extension=".joblib",
                )
            )
            write_file(
                results_card_module.build_prediction_artifact_path(
                    run_dir,
                    task_name="ec_number",
                    emb_name="ankh_base",
                    model_name="xgb_classification",
                    split_variant="random",
                    split_name="test",
                )
            )
            self.assertIsNotNone(results_card_module.find_model_artifact(run_dir))
            self.assertIsNotNone(results_card_module.find_prediction_artifact(run_dir, "test"))

    def test_result_entry_complete_requires_current_artifact_set(self) -> None:
        results_card_module = load_results_card_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "results" / "predictions" / "per_residue" / "random" / "linear" / "active_sites" / "ankh_base"
            legacy_entry = {
                "model_file": str(run_dir / "model.pt"),
                "test_pred_file": str(run_dir / "test_predictions.npz"),
            }
            write_file(run_dir / "model.pt")
            write_file(run_dir / "test_predictions.npz")
            self.assertFalse(results_card_module.is_result_entry_complete(legacy_entry))

            current_entry = {
                "model_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_per_residue.pt"),
                "train_pred_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_per_residue_train_predictions.safetensors.xz"),
                "test_pred_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_per_residue_test_predictions.safetensors.xz"),
                "metrics_file": str(run_dir / "metrics.json"),
                "hparams_file": str(run_dir / "hparams.json"),
                "training_log_file": str(run_dir / "training_log.csv"),
            }
            for path in current_entry.values():
                write_file(Path(path))
            self.assertTrue(results_card_module.is_result_entry_complete(current_entry))


if __name__ == "__main__":
    unittest.main()
