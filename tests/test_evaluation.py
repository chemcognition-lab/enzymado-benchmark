"""Unit tests for the bootstrap + ratings pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest import TestCase

import numpy as np
from scripts.results import compute_ratings as ratings_script

from src.evaluation import binary_statistics, statistics_cache, task_types
from src.evaluation import metrics_registry as metrics_module
from src.evaluation.bootstrap import (
    _bootstrap_additive_regression,
    _bootstrap_classification_counts,
    _precompute_binary_counts,
    _precompute_regression_errors,
    bootstrap_replicates,
    make_resample_indices,
    n_resample_units,
    summarise_replicates,
)
from src.evaluation.discover import (
    Experiment,
    TrackKey,
    assign_track,
    discover_experiments,
    group_into_tracks,
)
from src.evaluation.loaders import PredictionPayload, derive_task_type
from src.evaluation.metrics_registry import (
    classification_f1,
    higher_is_better_for,
    metric_set_for,
    primary_metric_for,
    regression_mae,
    regression_rmse,
)
from src.evaluation.rating_lib import (
    determine_outcome_aggregate,
    determine_outcome_raw,
    fit_batch_mle,
    fit_weighted_mle,
)


class TestMetrics(TestCase):
    def test_regression_mae_with_mask(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.5, 0.0])
        mask = np.array([True, True, False])
        self.assertAlmostEqual(regression_mae(y, p, mask), 0.5)

    def test_classification_f1_binary(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float32)
        p = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float32)
        self.assertAlmostEqual(
            classification_f1(y, p, None), 2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4)
        )

    def test_metric_set_dispatch(self) -> None:
        self.assertIn("mae", metric_set_for("regression"))
        self.assertIn("f1", metric_set_for("classification"))
        self.assertIn("auroc", metric_set_for("classification"))
        self.assertIn("auprc", metric_set_for("classification"))
        with self.assertRaises(ValueError):
            metric_set_for("nonsense")

    def test_primary_metric_table(self) -> None:
        self.assertEqual(primary_metric_for("regression"), "mae")
        self.assertEqual(primary_metric_for("classification"), "f1")
        self.assertEqual(higher_is_better_for("regression")["mae"], -1)
        self.assertEqual(higher_is_better_for("classification")["f1"], +1)

    def test_count_metrics_match_registered_classification_metrics(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
        predictions = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
        payload = PredictionPayload(y_true, predictions, None)
        count_metrics = binary_statistics.evaluate(
            binary_statistics.count_payload(payload),
            split="test",
        )
        direct_metrics = metrics_module.evaluate_predictions(
            y_true,
            predictions,
            "classification",
            split="test",
        )
        for name in ("accuracy", "precision", "recall", "f1", "auroc", "auprc"):
            self.assertAlmostEqual(count_metrics[name], direct_metrics[name])


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
            payload, idx, metric_set_for("regression"), "regression"
        )
        self.assertEqual(set(out.keys()), {"mae", "rmse", "r2", "kendall_tau"})
        self.assertEqual(out["mae"].shape, (8,))
        # Predictions are y + 0.5 → mae across any resample = 0.5
        np.testing.assert_allclose(out["mae"], 0.5, rtol=1e-6)

    def test_bootstrap_regression_task_type_not_misdetected_from_value_range(
        self,
    ) -> None:
        """A [0, 1]-scaled regression target must not be routed to classification metrics."""
        rng = np.random.default_rng(3)
        y = rng.random(30).astype(np.float32)
        p = y + rng.normal(0, 0.05, 30).astype(np.float32)
        payload = PredictionPayload(y_true=y, predictions=p, target_mask=None)
        idx = make_resample_indices(30, 10, seed=0)
        out = bootstrap_replicates(
            payload, idx, metric_set_for("regression"), "regression"
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
            payload, idx, metric_set_for("classification"), "classification"
        )
        # Across replicates the F1 should vary (some samples include only good proteins).
        self.assertGreater(out["f1"].std(), 0.0)

    def test_summarise_replicates_handles_nan(self) -> None:
        arr = np.array([0.5, 0.6, np.nan, 0.55])
        s = summarise_replicates({"f1": arr}, confidence_level=0.95)
        self.assertEqual(s["f1"]["n_finite"], 3)
        self.assertAlmostEqual(s["f1"]["mean"], np.mean([0.5, 0.6, 0.55]))


class TestLoaders(TestCase):
    def test_multi_head_ec_unification_concatenates_levels(self) -> None:
        from src.evaluation.loaders import _unify_multi_head_levels

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
        payload = _unify_multi_head_levels(out)
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

    def test_multi_head_unify_returns_none_for_flat_schema(self) -> None:
        from src.evaluation.loaders import _unify_multi_head_levels

        flat = {"y_true": np.zeros((2, 3)), "predictions": np.zeros((2, 3))}
        self.assertIsNone(_unify_multi_head_levels(flat))

    def test_multi_head_loader_preserves_encoded_sample_ids(self) -> None:
        from src.evaluation import sample_identity
        from src.evaluation.loaders import _unify_multi_head_levels

        byte_values, offsets = sample_identity.encode(("P1", "P2"))
        tensors = {
            "y_true_level_1": np.zeros((2, 1), dtype=np.float32),
            "predictions_level_1": np.zeros((2, 1), dtype=np.float32),
            "target_mask_level_1": np.ones((2, 1), dtype=bool),
            sample_identity.SAMPLE_ID_BYTES_KEY: byte_values,
            sample_identity.SAMPLE_ID_OFFSETS_KEY: offsets,
        }

        payload = _unify_multi_head_levels(tensors)

        self.assertIsNotNone(payload)
        self.assertEqual(payload.sample_ids, ("P1", "P2"))

    def test_sample_identity_fingerprint_depends_on_order(self) -> None:
        from src.evaluation import sample_identity

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
            payload, idx, metric_set_for("classification"), "classification"
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
            payload, idx, metric_set_for("classification"), "classification"
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
            payload, idx, metric_set_for("classification"), "classification"
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
        metrics = binary_statistics.evaluate(counts, split="test")
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
        additive = _bootstrap_additive_regression(stats, idx, {"mae", "rmse"})
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
        additive = _bootstrap_additive_regression(stats, idx, {"mae", "rmse"})
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

    def test_assign_track_routes_into_no_attention(self) -> None:
        e_mean = Experiment(
            "properties",
            "no_attention",
            "random",
            "linear",
            "binding",
            "ankh_base",
            Path("/tmp"),
        )
        e_residue = Experiment(
            "properties",
            "no_attention",
            "random",
            "xgboost",
            "active_sites",
            "ankh_large",
            Path("/tmp"),
        )
        e_attn = Experiment(
            "properties",
            "attention",
            "similarity",
            "mlp",
            "kcat",
            "esm2_650M",
            Path("/tmp"),
        )
        e_real = Experiment(
            "real_case",
            "no_attention",
            "random",
            "linear",
            "align_amylase_activity",
            "ankh_base",
            Path("/tmp"),
        )
        e_real_attn = Experiment(
            "real_case",
            "attention",
            "random",
            "mlp",
            "align_amylase_activity",
            "ankh_base",
            Path("/tmp"),
        )
        rc = {"align_amylase_activity"}
        self.assertEqual(assign_track(e_mean, rc), "no_attention")
        self.assertEqual(assign_track(e_residue, rc), "no_attention")
        self.assertEqual(assign_track(e_attn, rc), "attention")
        self.assertEqual(assign_track(e_real, rc), "real_case")
        self.assertEqual(assign_track(e_real_attn, rc), "real_case_attention")

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

    def test_group_into_tracks_buckets_by_split_and_family(self) -> None:
        exps = [
            Experiment(
                "properties",
                "no_attention",
                "random",
                "linear",
                "binding",
                "ankh_base",
                Path("/tmp/a"),
            ),
            Experiment(
                "properties",
                "no_attention",
                "random",
                "linear",
                "active_sites",
                "ankh_base",
                Path("/tmp/b"),
            ),
            Experiment(
                "properties",
                "attention",
                "random",
                "mlp",
                "kcat",
                "esm2_650M",
                Path("/tmp/c"),
            ),
        ]
        groups = group_into_tracks(exps, real_case_tasks=set())
        keys = sorted(groups.keys(), key=lambda k: (k.track, k.split, k.model_family))
        self.assertEqual(keys[0], TrackKey("attention", "random", "mlp"))
        self.assertEqual(keys[1], TrackKey("no_attention", "random", "linear"))
        # no_attention track should hold both experiments sharing (split, model_family)
        self.assertEqual(len(groups[TrackKey("no_attention", "random", "linear")]), 2)


class TestRatingLib(TestCase):
    def test_determine_outcome_raw_detects_clear_win(self) -> None:
        baseline = np.full(200, 0.50)
        candidate = np.full(200, 0.60)
        # All differences are +0.10 → highly significant
        self.assertEqual(determine_outcome_raw(baseline, candidate), 1.0)

    def test_determine_outcome_raw_ties_when_identical(self) -> None:
        arr = np.linspace(0.4, 0.6, 200)
        # All differences = 0 → ValueError fallback returns tie
        self.assertEqual(determine_outcome_raw(arr, arr), 0.5)

    def test_determine_outcome_aggregate_zero_se(self) -> None:
        # Both CIs are zero-width
        a = (0.5, 0.5, 0.5)
        b = (0.6, 0.6, 0.6)
        self.assertEqual(determine_outcome_aggregate(a, b), 1.0)
        self.assertEqual(determine_outcome_aggregate(b, a), 0.0)
        self.assertEqual(determine_outcome_aggregate(a, a), 0.5)

    def test_fit_batch_mle_produces_consistent_ranking(self) -> None:
        # Three players: A always beats B, B always beats C
        matches = [
            ("A", "B", 1.0),
            ("A", "B", 1.0),
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("B", "C", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 1.0),
            ("A", "C", 1.0),
        ]
        ratings = fit_batch_mle(
            matches, ["A", "B", "C"], anchor_model="C", anchor_rating=1500.0
        )
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])
        self.assertEqual(ratings["C"], 1500.0)


class TestGlobalRatings(TestCase):
    def test_expected_competitors_include_all_five_configurations(self) -> None:
        competitors = ratings_script._expected_competitors()
        self.assertEqual(len(competitors), 70)
        self.assertEqual(len(set(competitors)), 70)

    def test_kendall_weight_is_normalized_within_regression(self) -> None:
        weights = ratings_script._metric_weights("regression", 2.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["kendall_tau"], 0.4)
        self.assertTrue(all(weights[name] == 0.2 for name in ("mae", "rmse", "r2")))

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
