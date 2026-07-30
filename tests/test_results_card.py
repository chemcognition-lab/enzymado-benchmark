from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_results_card_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "results_layout.py"
    spec = importlib.util.spec_from_file_location("results_card_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ResultsCardTests(unittest.TestCase):
    def test_build_result_run_dir_uses_canonical_layout(self) -> None:
        module = load_results_card_module()
        build_result_run_dir = module.build_results_run_dir
        root = Path("/tmp/results")
        self.assertEqual(
            build_result_run_dir(
                root, "random", "binding", "ankh_base", "linear_classification",
                task_scope="properties",
            ),
            root / "properties" / "no_attention" / "random" / "linear" / "binding" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root, "random", "ddg", "ankh_base", "linear_regressor",
                task_scope="properties",
            ),
            root / "properties" / "no_attention" / "random" / "linear" / "ddg" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root, "random", "solubility", "ankh_base", "linear_classification",
                task_scope="properties",
            ),
            root / "properties" / "no_attention" / "random" / "linear" / "solubility" / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root, "similarity", "kcat", "esm2_650M", "attention_mlp_regressor",
                task_scope="properties",
            ),
            root / "properties" / "attention" / "loco" / "mlp" / "kcat" / "esm2_650M",
        )
        self.assertEqual(
            build_result_run_dir(
                root, "random", "active_sites", "prot5_xl_uniref50", "xgb_classification",
                task_scope="properties",
            ),
            root / "properties" / "no_attention" / "random" / "xgboost" / "active_sites" / "prot5_xl_uniref50",
        )
        self.assertEqual(
            build_result_run_dir(
                root, "random", "erickson_petase", "ankh_base", "linear_regressor",
                task_scope="real_case",
            ),
            root / "real_case" / "no_attention" / "random" / "linear" / "erickson_petase" / "ankh_base",
        )

    def test_artifact_paths_use_full_run_context(self) -> None:
        module = load_results_card_module()
        run_dir = Path("/tmp/results/no_attention/random/mlp/binding/ankh_base")
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
            / "mlp_classification_ankh_base_binding_random_no_attention_test_predictions.safetensors.xz",
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
            run_dir / "mlp_classification_ankh_base_binding_random_no_attention.joblib",
        )

    def test_named_artifact_helpers_accept_canonical_filenames(self) -> None:
        results_card_module = load_results_card_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "results" / "predictions" / "no_attention" / "random" / "xgboost" / "ec_number" / "ankh_base"
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
            run_dir = Path(tmp) / "results" / "predictions" / "no_attention" / "random" / "linear" / "active_sites" / "ankh_base"
            legacy_entry = {
                "model_file": str(run_dir / "model.pt"),
                "test_pred_file": str(run_dir / "test_predictions.npz"),
            }
            write_file(run_dir / "model.pt")
            write_file(run_dir / "test_predictions.npz")
            self.assertFalse(results_card_module.is_result_entry_complete(legacy_entry))

            current_entry = {
                "model_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_no_attention.pt"),
                "train_pred_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_no_attention_train_predictions.safetensors.xz"),
                "test_pred_file": str(run_dir / "linear_classification_ankh_base_active_sites_random_no_attention_test_predictions.safetensors.xz"),
                "metrics_file": str(run_dir / "metrics.json"),
                "hparams_file": str(run_dir / "hparams.json"),
                "training_log_file": str(run_dir / "training_log.csv"),
            }
            for path in current_entry.values():
                write_file(Path(path))
            self.assertTrue(results_card_module.is_result_entry_complete(current_entry))

            current_entry["invalidated"] = {"reason": "target_payload_mismatch_current_dataset"}
            self.assertFalse(results_card_module.is_result_entry_complete(current_entry))
            del current_entry["invalidated"]

            write_file(run_dir / results_card_module.INVALIDATION_MARKER)
            self.assertFalse(results_card_module.is_result_entry_complete(current_entry))


if __name__ == "__main__":
    unittest.main()
