"""Unit tests for the bootstrap + ratings pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from src.evaluation.bootstrap import (
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
from src.evaluation.loaders import PredictionPayload
from src.evaluation.metrics import (
    HIGHER_IS_BETTER,
    PRIMARY_METRIC,
    classification_f1,
    metric_set_for,
    regression_mae,
)
from src.evaluation.rating_lib import (
    determine_outcome_aggregate,
    determine_outcome_raw,
    fit_batch_mle,
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
        self.assertAlmostEqual(classification_f1(y, p, None), 2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4))

    def test_metric_set_dispatch(self) -> None:
        self.assertIn("mae", metric_set_for("regression"))
        self.assertIn("f1", metric_set_for("classification"))
        with self.assertRaises(ValueError):
            metric_set_for("nonsense")

    def test_primary_metric_table(self) -> None:
        self.assertEqual(PRIMARY_METRIC["regression"], "mae")
        self.assertEqual(PRIMARY_METRIC["classification"], "f1")
        self.assertEqual(HIGHER_IS_BETTER["mae"], -1)
        self.assertEqual(HIGHER_IS_BETTER["f1"], +1)


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
        out = bootstrap_replicates(payload, idx, metric_set_for("regression"))
        self.assertEqual(set(out.keys()), {"mae", "rmse", "r2", "kendall_tau"})
        self.assertEqual(out["mae"].shape, (8,))
        # Predictions are y + 0.5 → mae across any resample = 0.5
        np.testing.assert_allclose(out["mae"], 0.5, rtol=1e-6)

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
        out = bootstrap_replicates(payload, idx, metric_set_for("classification"))
        # Across replicates the F1 should vary (some samples include only good proteins).
        self.assertGreater(out["f1"].std(), 0.0)

    def test_summarise_replicates_handles_nan(self) -> None:
        arr = np.array([0.5, 0.6, np.nan, 0.55])
        s = summarise_replicates({"f1": arr}, confidence_level=0.95)
        self.assertEqual(s["f1"]["n_finite"], 3)
        self.assertAlmostEqual(s["f1"]["mean"], np.mean([0.5, 0.6, 0.55]))


class TestDiscover(TestCase):
    def test_discover_experiments_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mean/random/linear/binding/ankh_base").mkdir(parents=True)
            (root / "attention/loco/mlp/kcat/esm2_650M").mkdir(parents=True)
            (root / "per_residue/random/xgboost/active_sites/ankh_large").mkdir(parents=True)
            exps = discover_experiments(root)
            sigs = {(e.pool_type, e.split, e.model_family, e.task, e.embedding_model) for e in exps}
            self.assertEqual(len(exps), 3)
            self.assertIn(("mean", "random", "linear", "binding", "ankh_base"), sigs)
            self.assertIn(("attention", "loco", "mlp", "kcat", "esm2_650M"), sigs)
            self.assertIn(("per_residue", "random", "xgboost", "active_sites", "ankh_large"), sigs)

    def test_assign_track_routes_per_residue_into_mean(self) -> None:
        e_mean = Experiment("mean", "random", "linear", "binding", "ankh_base", Path("/tmp"))
        e_residue = Experiment(
            "per_residue", "random", "xgboost", "active_sites", "ankh_large", Path("/tmp")
        )
        e_attn = Experiment("attention", "loco", "mlp", "kcat", "esm2_650M", Path("/tmp"))
        e_real = Experiment("mean", "random", "linear", "align_amylase_activity", "ankh_base", Path("/tmp"))
        rc = {"align_amylase_activity"}
        self.assertEqual(assign_track(e_mean, rc), "mean")
        self.assertEqual(assign_track(e_residue, rc), "mean")
        self.assertEqual(assign_track(e_attn, rc), "attention")
        self.assertEqual(assign_track(e_real, rc), "real_case")

    def test_group_into_tracks_buckets_by_split_and_family(self) -> None:
        exps = [
            Experiment("mean", "random", "linear", "binding", "ankh_base", Path("/tmp/a")),
            Experiment("per_residue", "random", "linear", "active_sites", "ankh_base", Path("/tmp/b")),
            Experiment("attention", "random", "mlp", "kcat", "esm2_650M", Path("/tmp/c")),
        ]
        groups = group_into_tracks(exps, real_case_tasks=set())
        keys = sorted(groups.keys(), key=lambda k: (k.track, k.split, k.model_family))
        self.assertEqual(keys[0], TrackKey("attention", "random", "mlp"))
        self.assertEqual(keys[1], TrackKey("mean", "random", "linear"))
        # mean track should hold both the mean and per_residue experiments
        self.assertEqual(len(groups[TrackKey("mean", "random", "linear")]), 2)


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
        ratings = fit_batch_mle(matches, ["A", "B", "C"], anchor_model="C", anchor_rating=1500.0)
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])
        self.assertEqual(ratings["C"], 1500.0)
