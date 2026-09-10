#!/usr/bin/env python3
"""Compare the semantic prediction payloads produced by two benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import enzymado.evaluation.loaders  # noqa: E402


def _array_report(
    reference: np.ndarray | None,
    candidate: np.ndarray | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare one optional array without depending on serialization bytes."""
    if reference is None or candidate is None:
        exact = reference is None and candidate is None
        return {
            "reference_present": reference is not None,
            "candidate_present": candidate is not None,
            "exact": exact,
            "close": exact,
        }

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    same_shape = reference.shape == candidate.shape
    exact = same_shape and np.array_equal(reference, candidate, equal_nan=True)
    close = same_shape and np.allclose(
        reference,
        candidate,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    report: dict[str, object] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "exact": exact,
        "close": close,
    }
    if same_shape and reference.size:
        delta = reference.astype(np.float64) - candidate.astype(np.float64)
        finite_delta = delta[np.isfinite(delta)]
        report["max_abs_error"] = (
            float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.0
        )
    return report


def _block_report(
    reference: enzymado.evaluation.loaders.PredictionPayload,
    candidate: enzymado.evaluation.loaders.PredictionPayload,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare one logical prediction block."""
    arrays = {
        "y_true": _array_report(
            reference.y_true,
            candidate.y_true,
            atol=atol,
            rtol=rtol,
        ),
        "predictions": _array_report(
            reference.predictions,
            candidate.predictions,
            atol=atol,
            rtol=rtol,
        ),
        "target_mask": _array_report(
            reference.target_mask,
            candidate.target_mask,
            atol=atol,
            rtol=rtol,
        ),
        "proba": _array_report(
            reference.proba,
            candidate.proba,
            atol=atol,
            rtol=rtol,
        ),
    }
    sample_ids_exact = reference.sample_ids == candidate.sample_ids
    return {
        "sample_ids_exact": sample_ids_exact,
        "arrays": arrays,
        "exact": sample_ids_exact
        and all(bool(report["exact"]) for report in arrays.values()),
        "close": sample_ids_exact
        and all(bool(report["close"]) for report in arrays.values()),
    }


def compare_runs(
    reference_run: Path,
    candidate_run: Path,
    splits: tuple[str, ...],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Return semantic equality details for selected prediction splits."""
    split_reports: dict[str, object] = {}
    all_exact = True
    all_close = True
    for split in splits:
        reference = enzymado.evaluation.loaders.load_split_blocks(
            reference_run,
            split,
        )
        candidate = enzymado.evaluation.loaders.load_split_blocks(
            candidate_run,
            split,
        )
        if reference is None or candidate is None:
            exact = reference is None and candidate is None
            split_reports[split] = {
                "reference_present": reference is not None,
                "candidate_present": candidate is not None,
                "exact": exact,
                "close": exact,
            }
            all_exact &= exact
            all_close &= exact
            continue

        same_block_count = len(reference) == len(candidate)
        blocks = [
            _block_report(ref_block, candidate_block, atol=atol, rtol=rtol)
            for ref_block, candidate_block in zip(
                reference,
                candidate,
                strict=False,
            )
        ]
        exact = same_block_count and all(bool(block["exact"]) for block in blocks)
        close = same_block_count and all(bool(block["close"]) for block in blocks)
        split_reports[split] = {
            "reference_blocks": len(reference),
            "candidate_blocks": len(candidate),
            "blocks": blocks,
            "exact": exact,
            "close": close,
        }
        all_exact &= exact
        all_close &= close

    return {
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "run_manifest": {
            "reference": (reference_run / "run_manifest.json").is_file(),
            "candidate": (candidate_run / "run_manifest.json").is_file(),
        },
        "splits": split_reports,
        "exact": all_exact,
        "close": all_close,
        "atol": atol,
        "rtol": rtol,
    }


def parse_args() -> argparse.Namespace:
    """Parse two run directories and numeric comparison tolerances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        default=[],
    )
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Print the comparison as JSON and optionally require exact equality."""
    args = parse_args()
    splits = tuple(args.split or ("train", "val", "test"))
    report = compare_runs(
        args.reference_run,
        args.candidate_run,
        splits,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, indent=2))
    return int(args.require_exact and not report["exact"])


if __name__ == "__main__":
    raise SystemExit(main())
