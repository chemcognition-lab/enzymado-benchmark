"""Unit tests for the bootstrap + ratings pipeline."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
from unittest import TestCase
import unittest.mock

import numpy as np
from scripts.results import compute_ratings as ratings_script
import sklearn.metrics as skm

from enzymado import paths_and_constants
from enzymado.evaluation import binary_statistics, loaders, statistics_cache, task_types
from enzymado.evaluation import metrics_registry as metrics_module
from enzymado.evaluation.bootstrap import (
    _bootstrap_additive_regression,
    _bootstrap_classification_counts,
    _precompute_binary_counts,
    _precompute_regression_errors,
    bootstrap_replicates,
    make_resample_indices,
    n_resample_units,
    summarise_replicates,
)
from enzymado.evaluation.discover import (
    Experiment,
    discover_experiments,
)
from enzymado.evaluation.loaders import PredictionPayload, derive_task_type
import enzymado.evaluation.metrics_registry as metrics_registry
from enzymado.evaluation.metrics_registry import (
    higher_is_better_for,
    primary_metric_for,
    regression_mae,
    regression_rmse,
)
from enzymado.evaluation.rating_lib import (
    fit_weighted_mle,
)
import enzymado.types

_P = paths_and_constants.Paths


class TestMetrics(TestCase):
    def test_regression_mae_with_mask(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.5, 0.0])
        mask = np.array([True, True, False])
        self.assertAlmostEqual(regression_mae(y, p, mask), 0.5)

    def test_kendall_tau_scores_full_ties_as_no_skill(self) -> None:
        y = np.array([0.1, 0.4, 0.2, 0.9])
        kendall = metrics_registry.regression_kendall_tau
        self.assertEqual(kendall(y, np.full(4, 0.3)), 0.0)
        self.assertEqual(kendall(np.full(4, 0.3), y), 0.0)
        self.assertAlmostEqual(kendall(y, y), 1.0)
        self.assertTrue(np.isnan(kendall(y[:1], y[:1])))

    def test_classification_f1_binary(self) -> None:
        """F1 from the production count path, against a hand-computed value."""
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float32)
        p = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float32)
        metrics = metrics_registry.evaluate(
            binary_statistics.count_payload(PredictionPayload(y, p, None)),
            split="test",
        )
        self.assertAlmostEqual(metrics["f1"], 2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4))

    def test_metric_catalog_dispatch(self) -> None:
        expected = {
            "regression": {"mae", "rmse", "r2", "kendall_tau"},
            "classification": {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
                "auprc",
            },
        }
        for task_type, names in expected.items():
            specs = metrics_module.metric_specs(task_type)
            self.assertEqual({spec.name for spec in specs}, names)
        with self.assertRaises(ValueError):
            metrics_module.metric_specs("nonsense")

    def test_primary_metric_table(self) -> None:
        self.assertEqual(primary_metric_for("regression"), "mae")
        self.assertEqual(primary_metric_for("classification"), "f1")
        self.assertEqual(higher_is_better_for("regression")["mae"], -1)
        self.assertEqual(higher_is_better_for("classification")["f1"], +1)

    def test_count_metrics_match_sklearn_reference(self) -> None:
        """The confusion-count path must agree with sklearn on the same data.

        sklearn is the reference implementation and is called here, from the
        test. Production derives these six metrics from per-row counts, which
        is what keeps bootstrapping the largest tasks tractable; registering a
        sklearn version as the metric's own ``fn`` would declare a code path
        production never executes.
        """
        y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
        predictions = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
        payload = PredictionPayload(y_true, predictions, None)
        count_metrics = metrics_registry.evaluate(
            binary_statistics.count_payload(payload),
            split="test",
        )
        y_label = np.rint(y_true).astype(np.int8)
        p_label = np.rint(predictions).astype(np.int8)
        expected = {
            "accuracy": skm.accuracy_score(y_label, p_label),
            "precision": skm.precision_score(
                y_label, p_label, average="binary", zero_division=0
            ),
            "recall": skm.recall_score(
                y_label, p_label, average="binary", zero_division=0
            ),
            "f1": skm.f1_score(y_label, p_label, average="binary", zero_division=0),
            # AUROC/AUPRC rank by the raw prediction, as production does when a
            # run carries probabilities; with hard labels this is the same order.
            "auroc": skm.roc_auc_score(y_label, predictions),
            "auprc": skm.average_precision_score(y_label, predictions),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(count_metrics[name], float(value))

    def test_catalog_entry_reaches_all_classification_consumers(self) -> None:
        recall_spec = next(
            spec
            for spec in metrics_module.metric_specs("classification")
            if spec.name == "recall"
        )
        alias_spec = dataclasses.replace(
            recall_spec,
            name="sensitivity_alias",
            is_primary=False,
        )
        catalogue = metrics_module.METRICS + (alias_spec,)
        y_true = np.array([1, 0, 1, 0], dtype=np.float32)
        predictions = np.array([1, 0, 0, 0], dtype=np.float32)
        payload = PredictionPayload(y_true, predictions, None)
        counts = binary_statistics.count_payload(payload)
        indices = np.arange(y_true.size, dtype=np.int64)[None, :]

        with unittest.mock.patch.object(metrics_module, "METRICS", catalogue):
            # evaluate_predictions is deliberately NOT a classification
            # consumer: classification metrics declare no `fn` and are derived
            # from counts, so a new entry must reach the count, bootstrap and
            # direction consumers instead.
            direct = metrics_module.evaluate_predictions(
                y_true,
                predictions,
                "classification",
                split="test",
            )
            aggregated = metrics_module.evaluate(counts, split="test")
            replicates = metrics_module.bootstrap(counts, indices)
            directions = metrics_module.higher_is_better_for("classification")

        self.assertNotIn(alias_spec.name, direct)
        self.assertIn(alias_spec.name, aggregated)
        self.assertIn(alias_spec.name, replicates)
        self.assertIn(alias_spec.name, directions)


class TestBootstrap(TestCase):
    def test_make_resample_indices_shape_and_determinism(self) -> None:
        idx = make_resample_indices(n_units=10, n_replicates=5, seed=42)
        self.assertEqual(idx.shape, (5, 10))
        idx_again = make_resample_indices(n_units=10, n_replicates=5, seed=42)
        np.testing.assert_array_equal(idx, idx_again)

    def test_bootstrap_replicates_runs_metric_per_replicate(self) -> None:
        y = np.arange(20, dtype=np.float32)
        p = y + 0.5
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=None)
        idx = make_resample_indices(20, 8, seed=0)
        out = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("regression"),
        )
        repeated = bootstrap_replicates(
            payload,
            make_resample_indices(20, 8, seed=0),
            metrics_module.metric_specs("regression"),
        )
        self.assertEqual(set(out.keys()), {"mae", "rmse", "r2", "kendall_tau"})
        self.assertEqual(out["mae"].shape, (8,))
        for metric_name in out:
            np.testing.assert_array_equal(out[metric_name], repeated[metric_name])
        # Predictions are y + 0.5 → mae across any resample = 0.5
        np.testing.assert_allclose(out["mae"], 0.5, rtol=1e-6)

    def test_bootstrap_regression_task_type_not_misdetected_from_value_range(
        self,
    ) -> None:
        """A [0, 1]-scaled regression target must not route to classification."""
        rng = np.random.default_rng(3)
        y = rng.random(30).astype(np.float32)
        p = y + rng.normal(0, 0.05, 30).astype(np.float32)
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=None)
        idx = make_resample_indices(30, 10, seed=0)
        out = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("regression"),
        )
        self.assertEqual(set(out.keys()), {"mae", "rmse", "r2", "kendall_tau"})

    def test_bootstrap_per_residue_uses_protein_unit(self) -> None:
        rng = np.random.default_rng(0)
        n_proteins, max_len = 12, 5
        y = (rng.random((n_proteins, max_len)) > 0.5).astype(np.float32)
        p = y.copy()
        # Flip some predictions on the second half of proteins
        p[6:] = 1.0 - p[6:]
        mask = np.ones((n_proteins, max_len), dtype=bool)
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=mask)
        self.assertEqual(n_resample_units(payload), n_proteins)

        idx = make_resample_indices(n_proteins, 50, seed=1)
        out = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("classification"),
        )
        # Across replicates the F1 should vary (some samples include only
        # good proteins).
        self.assertGreater(out["f1"].std(), 0.0)

    def test_summarise_replicates_handles_nan(self) -> None:
        arr = np.array([0.5, 0.6, np.nan, 0.55])
        s = summarise_replicates({"f1": arr}, confidence_level=0.95)
        self.assertEqual(s["f1"]["n_finite"], 3)
        self.assertAlmostEqual(s["f1"]["mean"], np.mean([0.5, 0.6, 0.55]))


class TestLoaders(TestCase):
    def test_multi_head_ec_unification_concatenates_levels(self) -> None:
        from enzymado.evaluation.loaders import _multi_head_blocks

        out = {
            "y_true_level_1": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "predictions_level_1": np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32),
            "target_mask_level_1": np.array([True, True], dtype=bool),
            "y_true_level_2": np.array([[0, 1], [1, 1]], dtype=np.float32),
            "predictions_level_2": np.array([[0, 1], [1, 0]], dtype=np.float32),
            "target_mask_level_2": np.array(
                [True, False], dtype=bool
            ),  # row 1 unknown @ level 2
        }
        with unittest.mock.patch(
            "enzymado.evaluation.loaders.load_split_blocks",
            return_value=_multi_head_blocks(out),
        ):
            payload = loaders.load_split_predictions(Path("/unused"), "test")
        assert payload is not None
        self.assertEqual(payload.y_true.shape, (2, 5))
        self.assertEqual(payload.predictions.shape, (2, 5))
        self.assertEqual(payload.target_mask.shape, (2, 5))
        # Row 0 fully known, row 1 known at level_1 but masked at level_2
        np.testing.assert_array_equal(
            payload.target_mask,
            np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, False, False],
                ]
            ),
        )
        # The concatenation order must follow ascending level number.
        np.testing.assert_array_equal(payload.y_true[0], [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(payload.y_true[1], [0, 1, 0, 1, 1])

    def test_multi_head_blocks_returns_none_for_flat_schema(self) -> None:
        from enzymado.evaluation.loaders import _multi_head_blocks

        flat = {"y_true": np.zeros((2, 3)), "predictions": np.zeros((2, 3))}
        self.assertIsNone(_multi_head_blocks(flat))

    def test_multi_head_loader_preserves_encoded_sample_ids(self) -> None:
        from enzymado.artifacts import sample_identity
        from enzymado.evaluation.loaders import _multi_head_blocks

        byte_values, offsets = sample_identity.encode(("P1", "P2"))
        tensors = {
            "y_true_level_1": np.zeros((2, 1), dtype=np.float32),
            "predictions_level_1": np.zeros((2, 1), dtype=np.float32),
            "target_mask_level_1": np.ones((2, 1), dtype=bool),
            sample_identity.SAMPLE_ID_BYTES_KEY: byte_values,
            sample_identity.SAMPLE_ID_OFFSETS_KEY: offsets,
        }

        payload = _multi_head_blocks(tensors)[0]

        self.assertIsNotNone(payload)
        self.assertEqual(payload.sample_ids, ("P1", "P2"))

    def test_sample_identity_fingerprint_depends_on_order(self) -> None:
        from enzymado.artifacts import sample_identity

        first = sample_identity.manifest(("P1", "P2"))
        same = sample_identity.manifest(("P1", "P2"))
        reversed_order = sample_identity.manifest(("P2", "P1"))

        self.assertEqual(first, same)
        self.assertNotEqual(
            first["ordered_sha256"],
            reversed_order["ordered_sha256"],
        )

    def test_derive_task_type_from_hparams(self) -> None:
        self.assertEqual(
            derive_task_type({"task_config": {"label_kind": "binary"}}),
            "classification",
        )
        self.assertEqual(
            derive_task_type({"task_config": {"label_kind": "scalar_regression"}}),
            "regression",
        )
        self.assertEqual(
            derive_task_type({"task_config": {"label_kind": "multilabel_ec"}}),
            "classification",
        )
        self.assertEqual(
            derive_task_type({"task_config": {"label_kind": "residue_binary_vector"}}),
            "classification",
        )
        self.assertIsNone(derive_task_type({}))
        self.assertIsNone(derive_task_type({"task_config": {}}))

    def test_task_type_falls_back_to_benchmark_card(self) -> None:
        card = {
            "properties": {
                "ec_number": {"task_type": "classification"},
            },
            "real_case": {},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            card_path = Path(temporary_directory) / "benchmark_card.json"
            card_path.write_text(json.dumps(card), encoding="utf-8")
            task_type = task_types.resolve("ec_number", card_path, hparams={})
        self.assertEqual(task_type, "classification")


class TestBootstrapClassificationCounts(TestCase):
    """The additive classification path (TP/FP/FN/TN counts) must match a direct,
    per-replicate call to the metric functions themselves."""

    def _assert_metrics_match(
        self, direct: dict, additive: dict, atol: float = 1e-12
    ) -> None:
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
        ):
            np.testing.assert_allclose(
                additive[name], direct[name], atol=atol, err_msg=f"{name} mismatch"
            )

    def test_additive_path_matches_direct_for_1d_binary(self) -> None:
        rng = np.random.default_rng(0)
        n = 80
        y = (rng.random(n) > 0.5).astype(np.float32)
        p = y.copy()
        p[rng.choice(n, size=15, replace=False)] = (
            1 - p[rng.choice(n, size=15, replace=False)]
        )
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=None)

        idx = make_resample_indices(n, 30, seed=42)
        direct = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("classification"),
        )
        counts = _precompute_binary_counts(payload)
        additive = _bootstrap_classification_counts(counts, idx)
        self._assert_metrics_match(direct, additive)

    def test_additive_path_matches_direct_for_per_residue(self) -> None:
        rng = np.random.default_rng(1)
        n_proteins, max_len = 25, 12
        y = (rng.random((n_proteins, max_len)) > 0.6).astype(np.float32)
        p = y.copy()
        # Inject ~20% disagreement only on valid positions
        flip = rng.random((n_proteins, max_len)) < 0.2
        p = np.where(flip, 1.0 - p, p).astype(np.float32)
        # Valid mask varies per protein
        mask = rng.random((n_proteins, max_len)) > 0.1
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=mask)

        idx = make_resample_indices(n_proteins, 50, seed=7)
        direct = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("classification"),
        )
        counts = _precompute_binary_counts(payload)
        additive = _bootstrap_classification_counts(counts, idx)
        self._assert_metrics_match(direct, additive)

    def test_additive_path_matches_direct_for_multilabel(self) -> None:
        rng = np.random.default_rng(2)
        n_rows, n_cols = 40, 6
        y = (rng.random((n_rows, n_cols)) > 0.7).astype(np.float32)
        p = (rng.random((n_rows, n_cols)) > 0.6).astype(np.float32)
        mask = np.ones_like(y, dtype=bool)
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=mask)

        idx = make_resample_indices(n_rows, 40, seed=3)
        direct = bootstrap_replicates(
            payload,
            idx,
            metrics_module.metric_specs("classification"),
        )
        counts = _precompute_binary_counts(payload)
        additive = _bootstrap_classification_counts(counts, idx)
        self._assert_metrics_match(direct, additive)

    def test_multi_head_counts_avoid_dense_concatenation(self) -> None:
        first = PredictionPayload(
            y_true=np.array([[1, 0], [0, 1]], dtype=np.float32),
            predictions=np.array([[1, 1], [0, 1]], dtype=np.float32),
            target_mask=np.array([True, True]),
        )
        second = PredictionPayload(
            y_true=np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32),
            predictions=np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
            target_mask=np.array([True, False]),
        )
        counts = binary_statistics.count_blocks((first, second))
        np.testing.assert_array_equal(
            counts,
            np.array([[2, 1, 0, 2], [1, 0, 0, 1]], dtype=np.int64),
        )
        metrics = metrics_registry.evaluate(counts, split="test")
        self.assertEqual(metrics["n"], 7)

    def test_statistics_cache_validates_prediction_fingerprint(self) -> None:
        counts = np.array([[1, 0, 0, 2]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_path = root / "prediction.safetensors.xz"
            cache_path = root / "counts.npz"
            prediction_path.write_bytes(b"first")
            statistics_cache.save_counts(cache_path, counts, prediction_path)
            loaded = statistics_cache.load_counts(cache_path, prediction_path)
            np.testing.assert_array_equal(loaded, counts)

            prediction_path.write_bytes(b"second")
            self.assertIsNone(statistics_cache.load_counts(cache_path, prediction_path))


class TestBootstrapAdditiveRegression(TestCase):
    """The additive regression path (mae/rmse from per-row error sums) must match a
    direct, per-replicate call to regression_mae/regression_rmse themselves."""

    def _direct_mae_rmse(
        self, payload: PredictionPayload, idx: np.ndarray
    ) -> dict[str, np.ndarray]:
        out = {"mae": np.empty(idx.shape[0]), "rmse": np.empty(idx.shape[0])}
        for r, row in enumerate(idx):
            y = payload.y_true[row]
            p = payload.predictions[row]
            m = payload.target_mask[row] if payload.target_mask is not None else None
            out["mae"][r] = regression_mae(y, p, m)
            out["rmse"][r] = regression_rmse(y, p, m)
        return out

    def test_additive_path_matches_direct_for_1d_with_mask(self) -> None:
        rng = np.random.default_rng(4)
        n = 60
        y = rng.random(n).astype(np.float32)
        p = y + rng.normal(0, 0.1, n).astype(np.float32)
        mask = rng.random(n) > 0.2
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=mask)

        idx = make_resample_indices(n, 25, seed=11)
        direct = self._direct_mae_rmse(payload, idx)
        stats = _precompute_regression_errors(payload)
        specs = metrics_module.metric_specs_for_statistic(
            enzymado.types.MetricStatistic.ERROR_SUMS,
            "regression",
        )
        additive = _bootstrap_additive_regression(stats, idx, specs)
        np.testing.assert_allclose(additive["mae"], direct["mae"], atol=1e-6)
        np.testing.assert_allclose(additive["rmse"], direct["rmse"], atol=1e-6)

    def test_additive_path_matches_direct_for_per_residue_with_mask(self) -> None:
        rng = np.random.default_rng(5)
        n_proteins, max_len = 18, 9
        y = rng.random((n_proteins, max_len)).astype(np.float32)
        p = y + rng.normal(0, 0.1, (n_proteins, max_len)).astype(np.float32)
        mask = rng.random((n_proteins, max_len)) > 0.15
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=mask)

        idx = make_resample_indices(n_proteins, 20, seed=13)
        direct = self._direct_mae_rmse(payload, idx)
        stats = _precompute_regression_errors(payload)
        specs = metrics_module.metric_specs_for_statistic(
            enzymado.types.MetricStatistic.ERROR_SUMS,
            "regression",
        )
        additive = _bootstrap_additive_regression(stats, idx, specs)
        np.testing.assert_allclose(additive["mae"], direct["mae"], atol=1e-6)
        np.testing.assert_allclose(additive["rmse"], direct["rmse"], atol=1e-6)


class TestDiscover(TestCase):
    def test_discover_experiments_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "properties/no_attention/random/linear/binding/ankh_base").mkdir(
                parents=True
            )
            (root / "properties/attention/similarity/mlp/kcat/esm2_650M").mkdir(
                parents=True
            )
            (
                root / "properties/no_attention/random/xgboost/active_sites/ankh_large"
            ).mkdir(parents=True)
            (
                root / "real_case/no_attention/random/mlp/erickson_petase/ankh_base"
            ).mkdir(parents=True)
            exps = discover_experiments(root)
            sigs = {
                (
                    e.task_scope,
                    e.pool_type,
                    e.split,
                    e.model_family,
                    e.task,
                    e.embedding_model,
                )
                for e in exps
            }
            self.assertEqual(len(exps), 4)
            self.assertIn(
                (
                    "properties",
                    "no_attention",
                    "random",
                    "linear",
                    "binding",
                    "ankh_base",
                ),
                sigs,
            )
            self.assertIn(
                (
                    "properties",
                    "attention",
                    "similarity",
                    "mlp",
                    "kcat",
                    "esm2_650M",
                ),
                sigs,
            )
            self.assertIn(
                (
                    "properties",
                    "no_attention",
                    "random",
                    "xgboost",
                    "active_sites",
                    "ankh_large",
                ),
                sigs,
            )
            self.assertIn(
                (
                    "real_case",
                    "no_attention",
                    "random",
                    "mlp",
                    "erickson_petase",
                    "ankh_base",
                ),
                sigs,
            )

    def test_experiment_task_type_prefers_hparams_over_metrics(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "hparams.json").write_text(
                json.dumps({"task_config": {"label_kind": "binary"}})
            )
            (run_dir / "metrics.json").write_text(
                json.dumps({"test": {"task_type": "regression"}})
            )
            exp = Experiment(
                "properties",
                "no_attention",
                "random",
                "linear",
                "binding",
                "ankh_base",
                run_dir,
            )
            self.assertEqual(exp.task_type(), "classification")

    def test_experiment_task_type_falls_back_to_metrics_json(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text(
                json.dumps({"test": {"task_type": "regression"}})
            )
            exp = Experiment(
                "properties",
                "no_attention",
                "random",
                "linear",
                "binding",
                "ankh_base",
                run_dir,
            )
            self.assertEqual(exp.task_type(), "regression")


class TestRatingTieMargins(TestCase):
    def _cell(self, sigma) -> "ratings_script.Cell":
        scores = np.array([[0.50, 0.50], [0.49, 0.40], [0.50, 0.30]])
        return ratings_script.Cell(
            task="t",
            split="random",
            metric="kendall_tau",
            weight=1.0,
            scores=scores,
            sigma=sigma,
        )

    def test_zero_margins_match_the_strict_comparison(self) -> None:
        pair_a, pair_b = ratings_script._pair_indices(3)
        strict, _ = ratings_script._battle_outcomes(
            [self._cell(None)], pair_a, pair_b, 2
        )
        zero, _ = ratings_script._battle_outcomes(
            [self._cell(np.zeros(3))], pair_a, pair_b, 2
        )
        np.testing.assert_array_equal(strict, zero)

    def test_gaps_within_the_noise_margin_are_ties(self) -> None:
        pair_a, pair_b = ratings_script._pair_indices(3)
        outcomes, _ = ratings_script._battle_outcomes(
            [self._cell(np.full(3, 0.01))], pair_a, pair_b, 2, tie_z=1.96
        )
        # The margin is 1.96 * sqrt(2) * 0.01 = 0.0277: a 0.01 gap ties, 0.10 wins.
        self.assertEqual(outcomes[0, 0], 0.5)
        self.assertEqual(outcomes[1, 0], 1.0)

    def test_every_competitor_needs_a_margin(self) -> None:
        competitors = ["a/mlp/no_attention", "b/linear/attention"]
        margins = {("t", "random", "kendall_tau", "mlp", "no_attention"): 0.01}
        with self.assertRaisesRegex(ValueError, "linear"):
            ratings_script._cell_sigma(
                "t", "random", "kendall_tau", competitors, margins
            )


class TestGlobalRatings(TestCase):
    def test_expected_competitors_include_all_five_configurations(self) -> None:
        competitors = ratings_script._expected_competitors()
        self.assertEqual(len(competitors), 75)
        self.assertEqual(len(set(competitors)), 75)
        self.assertIn("noplm/xgboost/no_attention", competitors)

    _RATING = json.loads(_P.benchmark_card.read_text(encoding="utf-8"))["rating"]

    def test_property_task_weights_follow_five_descending_tiers(self) -> None:
        tasks = sorted(self._RATING["task_weights"]["properties"])
        w = ratings_script._task_weights(self._RATING, "properties", tasks)
        self.assertEqual({w["active_sites"], w["ec_number"]}, {1.0})
        self.assertEqual(w["kcat"], w["binding_site"])
        self.assertEqual(w["km"], w["ph_opt"])
        self.assertEqual(w["ddg"], w["dtm"])
        self.assertEqual(w["binding"], w["solubility"])
        tiers = [w["ec_number"], w["kcat"], w["km"], w["ddg"], w["binding"]]
        self.assertEqual(tiers, sorted(tiers, reverse=True))
        self.assertEqual(len(set(tiers)), 5)

    def test_card_covers_every_property_task_and_split(self) -> None:
        card = json.loads(_P.benchmark_card.read_text(encoding="utf-8"))
        self.assertEqual(
            set(self._RATING["task_weights"]["properties"]), set(card["properties"])
        )
        for table in self._RATING["split_weights"].values():
            self.assertAlmostEqual(sum(table.values()), 1.0)

    def test_scope_without_a_weight_table_weighs_tasks_equally(self) -> None:
        cfg = {"task_weights": {}}
        weights = ratings_script._task_weights(cfg, "real_case", ["a", "b"])
        self.assertEqual(weights, {"a": 1.0, "b": 1.0})

    def test_unweighted_property_task_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "new_task"):
            tasks = ["kcat", "new_task"]
            ratings_script._task_weights(self._RATING, "properties", tasks)

    def test_kendall_weight_is_normalized_within_regression(self) -> None:
        weights = ratings_script._metric_weights(
            "regression", self._RATING["metric_weights"]["regression"]
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["kendall_tau"], 0.75)
        self.assertAlmostEqual(weights["mae"], 0.25)
        self.assertNotIn("rmse", weights)

    def test_unrated_metrics_never_reach_the_leaderboard(self) -> None:
        """Metrics the card lists as unrated must cast no battles and must exist."""
        unrated = set(self._RATING["unrated_metrics"])
        self.assertTrue(unrated)
        for task_type, table in self._RATING["metric_weights"].items():
            known = set(metrics_registry.higher_is_better_for(task_type))
            weights = ratings_script._metric_weights(task_type, table)
            self.assertFalse(unrated & set(weights))
            self.assertTrue(set(weights) <= known)

    def test_unknown_metric_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "spearman"):
            ratings_script._metric_weights("regression", {"spearman": 1.0})

    def test_excluded_metrics_are_renormalized_within_the_cell(self) -> None:
        """Dropping a metric must not cost its task any leaderboard influence."""
        names = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")
        table = {name: 1.0 for name in names}
        weights = ratings_script._metric_weights(
            "classification", table, frozenset({"accuracy"})
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertNotIn("accuracy", weights)
        self.assertTrue(all(value == 0.2 for value in weights.values()))

    def test_metric_weights_reject_a_fully_excluded_cell(self) -> None:
        table = self._RATING["metric_weights"]["classification"]
        with self.assertRaisesRegex(ValueError, "ec_number/random"):
            ratings_script._metric_weights(
                "classification", table, frozenset(table), label="ec_number/random"
            )

    def test_prevalence_rule_drops_only_the_imbalanced_splits(self) -> None:
        """The rule reads the card, so a balanced split must keep accuracy."""
        rule = {"metrics": ["accuracy"], "threshold": 0.05}
        excluded = ratings_script._prevalence_exclusions(
            rule,
            {"rare": "classification", "even": "classification", "num": "regression"},
            {"random": 1.0},
            {("rare", "random"): 0.004, ("even", "random"): 0.5},
        )
        self.assertEqual(
            [(item.task, item.split, item.metric) for item in excluded],
            [("rare", "random", "accuracy")],
        )

    def test_prevalence_rule_requires_a_declared_prevalence(self) -> None:
        with self.assertRaisesRegex(ValueError, "class_prevalence"):
            ratings_script._prevalence_exclusions(
                {"metrics": ["accuracy"], "threshold": 0.05},
                {"rare": "classification"},
                {"random": 1.0},
                {},
            )

    def test_ceiling_rule_drops_only_cells_without_headroom(self) -> None:
        """The ceiling is the best competitor, not the average one."""
        rule = {"thresholds": {"kendall_tau": 0.10}}
        arrays = {
            ("flat", "random", "kendall_tau"): {
                "a": np.full(4, 0.01),
                "b": np.full(4, 0.05),
            },
            ("live", "random", "kendall_tau"): {
                "a": np.full(4, 0.01),
                "b": np.full(4, 0.40),
            },
        }
        excluded = ratings_script._ceiling_exclusions(
            rule,
            {"flat": "regression", "live": "regression"},
            {"random": 1.0},
            arrays,
        )
        self.assertEqual([item.task for item in excluded], ["flat"])
        self.assertAlmostEqual(excluded[0].measured, 0.05)

    def test_ceiling_rule_rejects_a_lower_is_better_metric(self) -> None:
        """A threshold on rmse would silently invert the comparison."""
        with self.assertRaisesRegex(ValueError, "higher-is-better"):
            ratings_script._ceiling_exclusions(
                {"thresholds": {"rmse": 0.10}},
                {"num": "regression"},
                {"random": 1.0},
                {("num", "random", "rmse"): {"a": np.full(4, -1.0)}},
            )

    def test_weighted_mle_ranks_fractional_pairwise_wins(self) -> None:
        pair_a = np.array([0, 0, 1], dtype=np.int64)
        pair_b = np.array([1, 2, 2], dtype=np.int64)
        outcomes = np.array([0.8, 0.9, 0.7], dtype=np.float64)
        weights = np.full(3, 5.0)
        ratings = fit_weighted_mle(
            pair_a,
            pair_b,
            outcomes,
            weights,
            ["A", "B", "C"],
        )
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])
        self.assertAlmostEqual(sum(ratings.values()) / 3, 1500.0)
