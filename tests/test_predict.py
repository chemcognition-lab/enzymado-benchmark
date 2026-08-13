"""Tests for task-aware prediction adapters."""

from __future__ import annotations

import numpy as np

from training.predict import predict_split_from_arrays, predict_split_from_loader


class _ProbabilityEstimator:
    def predict(self, features):
        return np.ones(len(features), dtype=np.int8)

    def predict_proba(self, features):
        positive = np.asarray(features, dtype=np.float32).reshape(-1)
        return np.column_stack([1.0 - positive, positive])


class _StructuredScoreEstimator:
    def predict(self, features):
        return {"level_1": np.ones((len(features), 1), dtype=np.float32)}

    def predict_scores(self, features):
        return {
            "level_1": np.asarray(features, dtype=np.float32).reshape(-1, 1)
        }


class _LoaderEstimator:
    def predict_loader(self, dataloader, batch_kind):
        return np.asarray([10.0], dtype=np.float32)

    def predict_scores_loader(self, dataloader, batch_kind):
        return np.asarray([0.2, 0.8], dtype=np.float32)


def test_array_classification_uses_positive_class_probability() -> None:
    estimator = _ProbabilityEstimator()
    features = np.asarray([0.1, 0.7], dtype=np.float32)

    predictions = predict_split_from_arrays(
        estimator,
        features,
        "classification",
    )

    np.testing.assert_allclose(predictions, [0.1, 0.7])


def test_array_classification_preserves_structured_scores() -> None:
    estimator = _StructuredScoreEstimator()

    predictions = predict_split_from_arrays(
        estimator,
        np.asarray([0.25, 0.75], dtype=np.float32),
        "classification",
    )

    np.testing.assert_allclose(predictions["level_1"], [[0.25], [0.75]])


def test_loader_adapter_selects_scores_only_for_classification() -> None:
    estimator = _LoaderEstimator()

    classification = predict_split_from_loader(
        estimator, object(), "protein_global", "classification"
    )
    regression = predict_split_from_loader(
        estimator, object(), "protein_global", "regression"
    )

    np.testing.assert_allclose(classification, [0.2, 0.8])
    np.testing.assert_allclose(regression, [10.0])
