from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


def load_results_layout_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "evaluation" / "results_layout.py"
    )
    spec = importlib.util.spec_from_file_location("results_layout_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ResultsLayoutTests(unittest.TestCase):
    def test_unknown_split_family_is_rejected(self) -> None:
        module = load_results_layout_module()
        with self.assertRaisesRegex(ValueError, "Unknown split variant"):
            module.canonical_split_family("unsupported")

    def test_build_result_run_dir_uses_canonical_layout(self) -> None:
        module = load_results_layout_module()
        build_result_run_dir = module.build_results_run_dir
        root = Path("/tmp/results")
        self.assertEqual(
            build_result_run_dir(
                root,
                "random",
                "binding",
                "ankh_base",
                "linear_classification",
                task_scope="properties",
            ),
            root
            / "properties"
            / "no_attention"
            / "random"
            / "linear"
            / "binding"
            / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root,
                "random",
                "ddg",
                "ankh_base",
                "linear_regressor",
                task_scope="properties",
            ),
            root
            / "properties"
            / "no_attention"
            / "random"
            / "linear"
            / "ddg"
            / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root,
                "random",
                "solubility",
                "ankh_base",
                "linear_classification",
                task_scope="properties",
            ),
            root
            / "properties"
            / "no_attention"
            / "random"
            / "linear"
            / "solubility"
            / "ankh_base",
        )
        self.assertEqual(
            build_result_run_dir(
                root,
                "similarity",
                "kcat",
                "esm2_650M",
                "attention_mlp_regressor",
                task_scope="properties",
            ),
            root
            / "properties"
            / "attention"
            / "similarity"
            / "mlp"
            / "kcat"
            / "esm2_650M",
        )
        self.assertEqual(
            build_result_run_dir(
                root,
                "random",
                "active_sites",
                "prot5_xl_uniref50",
                "xgb_classification",
                task_scope="properties",
            ),
            root
            / "properties"
            / "no_attention"
            / "random"
            / "xgboost"
            / "active_sites"
            / "prot5_xl_uniref50",
        )
        self.assertEqual(
            build_result_run_dir(
                root,
                "random",
                "erickson_petase",
                "ankh_base",
                "linear_regressor",
                task_scope="real_case",
            ),
            root
            / "real_case"
            / "no_attention"
            / "random"
            / "linear"
            / "erickson_petase"
            / "ankh_base",
        )

    def test_artifact_paths_use_full_run_context(self) -> None:
        module = load_results_layout_module()
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
        results_layout_module = load_results_layout_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = (
                Path(tmp)
                / "results"
                / "predictions"
                / "no_attention"
                / "random"
                / "xgboost"
                / "ec_number"
                / "ankh_base"
            )
            write_file(
                results_layout_module.build_model_artifact_path(
                    run_dir,
                    task_name="ec_number",
                    emb_name="ankh_base",
                    model_name="xgb_classification",
                    split_variant="random",
                    extension=".joblib",
                )
            )
            write_file(
                results_layout_module.build_prediction_artifact_path(
                    run_dir,
                    task_name="ec_number",
                    emb_name="ankh_base",
                    model_name="xgb_classification",
                    split_variant="random",
                    split_name="test",
                )
            )
            self.assertIsNotNone(results_layout_module.find_model_artifact(run_dir))
            self.assertIsNotNone(
                results_layout_module.find_prediction_artifact(run_dir, "test")
            )

    def test_run_complete_requires_current_artifact_set(self) -> None:
        results_layout_module = load_results_layout_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = (
                Path(tmp)
                / "results"
                / "predictions"
                / "properties"
                / "no_attention"
                / "random"
                / "linear"
                / "active_sites"
                / "ankh_base"
            )
            write_file(run_dir / "model.pt")
            write_file(run_dir / "test_predictions.npz")
            self.assertFalse(results_layout_module.is_run_complete(run_dir))

            for split_name in ("train", "test"):
                write_file(
                    results_layout_module.build_prediction_artifact_path(
                        run_dir,
                        task_name="active_sites",
                        emb_name="ankh_base",
                        model_name="linear_classification",
                        split_variant="random",
                        split_name=split_name,
                    )
                )
            for filename in ("metrics.json", "hparams.json", "training_log.csv"):
                write_file(run_dir / filename)
            self.assertTrue(results_layout_module.is_run_complete(run_dir))

            write_file(run_dir / results_layout_module.INVALIDATION_MARKER)
            self.assertFalse(results_layout_module.is_run_complete(run_dir))


if __name__ == "__main__":
    unittest.main()
