"""Tests for weighted multi-metric benchmark ratings."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from scripts.results import compute_ratings, plot_real_case_ratings

from src.evaluation import discover


class WeightedRatingTests(unittest.TestCase):
    def test_expected_matrix_contains_all_five_baselines_per_embedding(self) -> None:
        baselines = compute_ratings._expected_baselines(
            set(compute_ratings.DEFAULT_FOUNDATION_MODELS)
        )

        self.assertEqual(len(baselines), 70)
        self.assertIn("ankh_base::linear::attention", baselines)
        self.assertIn("ankh_base::xgboost::no_attention", baselines)

    def test_baseline_id_includes_embedding_model_and_pooling(self) -> None:
        exp = discover.Experiment(
            "real_case",
            "attention",
            "random",
            "mlp",
            "erickson_petase",
            "ankh_base",
            Path("/tmp/run"),
        )

        self.assertEqual(
            compute_ratings._baseline_id(exp),
            "ankh_base::mlp::attention",
        )

    def test_regression_matches_weight_all_metrics(self) -> None:
        high = np.linspace(0.7, 0.9, 200)
        low = np.linspace(0.1, 0.3, 200)
        high[0] = np.nan
        arrays_by_metric = {
            "kendall_tau": {"A": high, "B": low},
            "mae": {"A": high, "B": low},
            "rmse": {"A": low, "B": high},
            "r2": {"A": high, "B": low},
        }
        signs = {
            "kendall_tau": 1,
            "mae": -1,
            "rmse": -1,
            "r2": 1,
        }

        matches, details = compute_ratings._build_weighted_matches_for_task(
            arrays_by_metric,
            compute_ratings.REGRESSION_METRIC_WEIGHTS,
            signs,
            alpha=0.05,
        )

        self.assertEqual(matches, [("A", "B", 0.8)])
        pair = details["A__vs__B"]
        self.assertEqual(pair["weighted_outcome_a"], 0.8)
        self.assertEqual(set(pair["metric_outcomes"]), set(arrays_by_metric))
        self.assertEqual(
            pair["metric_outcomes"]["kendall_tau"]["n_paired_replicates"],
            199,
        )

    def test_kendall_has_largest_regression_weight(self) -> None:
        weights = compute_ratings._metric_weights("regression")

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(weights["kendall_tau"], 0.4)
        self.assertTrue(
            all(
                weights["kendall_tau"] > weight
                for metric, weight in weights.items()
                if metric != "kendall_tau"
            )
        )

    def test_plot_centers_ratings_and_produces_finite_standard_errors(self) -> None:
        report = {
            "anchor_baseline": "B",
            "ratings": {"A": 1600.0, "B": 1500.0, "C": 1400.0},
            "per_task_report": {
                "task": {
                    "pairwise_outcomes": {
                        "A__vs__B": {
                            "baseline_a": "A",
                            "baseline_b": "B",
                            "weighted_outcome_a": 1.0,
                        },
                        "A__vs__C": {
                            "baseline_a": "A",
                            "baseline_b": "C",
                            "weighted_outcome_a": 1.0,
                        },
                        "B__vs__C": {
                            "baseline_a": "B",
                            "baseline_b": "C",
                            "weighted_outcome_a": 1.0,
                        },
                    }
                }
            },
        }

        ratings, standard_errors = (
            plot_real_case_ratings._centered_ratings_and_se(report)
        )

        self.assertAlmostEqual(float(np.mean(list(ratings.values()))), 1500.0)
        self.assertTrue(all(np.isfinite(list(standard_errors.values()))))
        self.assertTrue(all(error > 0.0 for error in standard_errors.values()))


if __name__ == "__main__":
    unittest.main()
