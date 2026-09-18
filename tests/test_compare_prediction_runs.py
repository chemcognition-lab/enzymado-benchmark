"""Tests for semantic comparison of benchmark prediction runs."""

from pathlib import Path

import numpy as np
import scripts.results.compare_prediction_runs

import enzymado.artifacts.serialization


def _write_test_predictions(run_dir: Path, predictions: np.ndarray) -> None:
    """Write one minimal run with fixed identities and targets."""
    run_dir.mkdir()
    enzymado.artifacts.serialization.make_safetensors_predictions_file(
        np.asarray([0.0, 1.0], dtype=np.float32),
        predictions,
        run_dir,
        "test",
        artifact_path=run_dir / "canary_test_predictions.safetensors.xz",
        sample_ids=("P1", "P2"),
    )


def test_compare_runs_reports_exact_and_changed_predictions(tmp_path: Path) -> None:
    """Distinguish an exact replay from a numerically different candidate."""
    reference = tmp_path / "reference"
    exact_candidate = tmp_path / "exact"
    changed_candidate = tmp_path / "changed"
    predictions = np.asarray([0.25, 0.75], dtype=np.float32)
    _write_test_predictions(reference, predictions)
    _write_test_predictions(exact_candidate, predictions.copy())
    _write_test_predictions(
        changed_candidate,
        np.asarray([0.25, 0.5], dtype=np.float32),
    )

    exact = scripts.results.compare_prediction_runs.compare_runs(
        reference,
        exact_candidate,
        ("test",),
        atol=0.0,
        rtol=0.0,
    )
    changed = scripts.results.compare_prediction_runs.compare_runs(
        reference,
        changed_candidate,
        ("test",),
        atol=0.0,
        rtol=0.0,
    )

    assert exact["exact"] is True
    assert changed["exact"] is False
    changed_split = changed["splits"]["test"]
    changed_predictions = changed_split["blocks"][0]["arrays"]["predictions"]
    assert changed_predictions["max_abs_error"] == 0.25
