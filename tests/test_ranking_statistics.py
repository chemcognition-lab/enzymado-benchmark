"""Cover the histogram sufficient statistics that back probability-based AUC."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics as skm

import enzymado.artifacts.serialization as artifacts
import enzymado.evaluation.binary_statistics as binary_statistics
import enzymado.evaluation.loaders as loaders
import enzymado.evaluation.metrics_registry as metrics_registry

RNG = np.random.default_rng(0)


def _payload(y_true, proba, target_mask=None):
    return loaders.PredictionPayload(
        y_true=y_true,
        predictions=(np.asarray(proba) >= 0.5).astype(np.float32),
        target_mask=target_mask,
        proba=np.asarray(proba, dtype=np.float32),
    )


def _flat(payload):
    mask = payload.target_mask
    valid = np.ones(np.asarray(payload.y_true).shape, bool) if mask is None else mask
    y = np.rint(np.asarray(payload.y_true)[valid]).astype(np.int8)
    return y, np.asarray(payload.proba)[valid]


def _pooled_payload(n=4000):
    """One score per protein: y_true and proba are 1D."""
    y = (RNG.random(n) < 0.3).astype(np.float32)
    proba = np.clip(RNG.random(n) * 0.6 + y * 0.3, 0, 1)
    return _payload(y, proba)


def _residue_payload(rows=400, width=120):
    """A padded row per protein, with a validity mask: y_true and proba are 2D."""
    y = (RNG.random((rows, width)) < 0.12).astype(np.float32)
    proba = np.clip(RNG.random((rows, width)) * 0.6 + y * 0.3, 0, 1)
    return _payload(y, proba, target_mask=RNG.random((rows, width)) < 0.8)


@pytest.mark.parametrize("factory", [_pooled_payload, _residue_payload])
def test_histogram_auc_matches_sklearn(factory) -> None:
    """Binned sufficient statistics must track the reference implementation."""
    payload = factory()
    y, scores = _flat(payload)

    metrics = metrics_registry.evaluate_ranking(
        binary_statistics.ranking_histograms(payload)
    )
    assert metrics["auroc"] == pytest.approx(skm.roc_auc_score(y, scores), abs=1e-4)
    assert metrics["auprc"] == pytest.approx(
        skm.average_precision_score(y, scores), abs=1e-3
    )


@pytest.mark.parametrize("factory", [_pooled_payload, _residue_payload])
def test_histograms_carry_one_row_per_resample_unit(factory) -> None:
    """Bootstrapping resamples rows, so histograms must stay row-aligned."""
    payload = factory()
    histograms = binary_statistics.ranking_histograms(payload)
    assert histograms.shape[0] == np.asarray(payload.y_true).shape[0]
    assert histograms.shape[1] == 2 * binary_statistics.RANKING_BINS


def test_histogram_totals_count_only_valid_entries() -> None:
    """Masked-out residues must not contribute to any bin."""
    payload = _residue_payload()
    histograms = binary_statistics.ranking_histograms(payload)
    assert int(histograms.sum()) == int(np.asarray(payload.target_mask).sum())


def test_bootstrap_ranking_is_additive_over_rows() -> None:
    """Selecting every row once must reproduce the point estimate exactly."""
    payload = _pooled_payload()
    histograms = binary_statistics.ranking_histograms(payload)
    identity = np.arange(histograms.shape[0])[None, :]

    replicates = metrics_registry.bootstrap_ranking(histograms, identity)
    point = metrics_registry.evaluate_ranking(histograms)
    assert replicates["auroc"][0] == pytest.approx(point["auroc"], abs=1e-12)
    assert replicates["auprc"][0] == pytest.approx(point["auprc"], abs=1e-12)


def test_bootstrap_ranking_rejects_mismatched_indices() -> None:
    payload = _pooled_payload()
    histograms = binary_statistics.ranking_histograms(payload)
    with pytest.raises(ValueError, match="Indices expect"):
        metrics_registry.bootstrap_ranking(histograms, np.zeros((3, 7), dtype=np.int64))


def test_ranking_histograms_require_probabilities() -> None:
    payload = _pooled_payload()
    without = loaders.PredictionPayload(
        y_true=payload.y_true,
        predictions=payload.predictions,
        target_mask=payload.target_mask,
    )
    with pytest.raises(ValueError, match="require stored probabilities"):
        binary_statistics.ranking_histograms(without)


def test_ranking_blocks_return_none_without_probabilities() -> None:
    """EC heads without probabilities fall back to the hard-label path."""
    payload = _pooled_payload()
    without = loaders.PredictionPayload(
        y_true=payload.y_true,
        predictions=payload.predictions,
        target_mask=payload.target_mask,
    )
    assert binary_statistics.ranking_histogram_blocks((payload, without)) is None
    assert binary_statistics.ranking_histogram_blocks(()) is None


def test_multi_head_blocks_propagate_probabilities(tmp_path) -> None:
    """EC heads must carry probabilities through the artifact round trip.

    The multi-head reader rebuilds one payload per EC level; dropping the
    probability there silently downgrades auroc/auprc back to hard labels
    without raising, so the round trip is asserted explicitly.
    """
    rows, levels = 32, (3, 5)
    y_true = {
        f"level_{i + 1}": (RNG.random((rows, w)) < 0.3).astype(np.float32)
        for i, w in enumerate(levels)
    }
    proba = {
        name: RNG.random(value.shape).astype(np.float32)
        for name, value in y_true.items()
    }
    predictions = {name: (proba[name] >= 0.5).astype(np.float32) for name in y_true}
    target_mask = {
        name: np.ones(value.shape, dtype=bool) for name, value in y_true.items()
    }

    path = artifacts.make_safetensors_predictions_file(
        y_true,
        predictions,
        tmp_path,
        "test",
        target_mask,
        pred_proba=proba,
    )

    blocks = loaders._read_safetensors_blocks(path)
    assert len(blocks) == len(levels)
    for block in blocks:
        assert block.proba is not None
    for block, name in zip(blocks, sorted(y_true)):
        assert block.proba == pytest.approx(proba[name], abs=2e-05)

    assert binary_statistics.ranking_histogram_blocks(blocks) is not None
