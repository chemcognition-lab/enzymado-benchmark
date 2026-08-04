#!/usr/bin/env python3
"""Audit saved scalar-regression y_true payloads against current task targets."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
from pathlib import Path
import sys
from typing import Any

import numpy as np
from safetensors.torch import load as load_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import TASK_REGISTRY  # noqa: E402
from evaluation.discover import discover_experiments  # noqa: E402
from evaluation.results_layout import (  # noqa: E402
    INVALIDATION_MARKER,
    find_prediction_artifact,
)

SPLIT_TO_DATASET_VARIANT = {
    "random": "random",
    "loco": "similarity",
    "similarity": "similarity",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _load_split_targets(
    config, dataset_variant: str, split_name: str
) -> np.ndarray | None:
    dataset_path = config.dataset_paths.get(dataset_variant)
    if dataset_path is None or config.target_column is None:
        return None
    values: list[float] = []
    with Path(dataset_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(config.split_column) != split_name:
                continue
            raw_value = (row.get(config.target_column) or "").strip()
            if raw_value:
                values.append(float(raw_value))
    return np.asarray(values, dtype=np.float64)


def _prediction_path(run_dir: Path, split_name: str) -> Path | None:
    return find_prediction_artifact(run_dir, split_name)


def _load_y_true(prediction_path: Path) -> np.ndarray:
    with lzma.open(prediction_path, "rb") as handle:
        payload = load_tensors(handle.read())
    y_keys = sorted(key for key in payload if key.startswith("y_true"))
    if not y_keys:
        raise ValueError(f"No y_true tensor found in {prediction_path}")
    y_true = payload[y_keys[0]].cpu().numpy().reshape(-1).astype(np.float64)
    return y_true[np.isfinite(y_true)]


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "max_abs": None,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _target_mismatch(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[bool, str | None]:
    if observed.size == 0 or expected.size == 0:
        return False, None

    observed_summary = _summary(observed)
    expected_summary = _summary(expected)
    if observed.size == expected.size:
        if np.allclose(np.sort(observed), np.sort(expected), atol=atol, rtol=rtol):
            return False, None
        return True, "same_n_different_values"

    observed_mean = float(observed_summary["mean"])
    expected_mean = float(expected_summary["mean"])
    expected_std = max(float(expected_summary["std"]), 1e-6)
    observed_max_abs = float(observed_summary["max_abs"])
    expected_max_abs = max(float(expected_summary["max_abs"]), 1e-6)
    severe_mean_shift = abs(observed_mean - expected_mean) > max(
        5.0, 5.0 * expected_std
    )
    severe_scale_shift = observed_max_abs > max(50.0, 10.0 * expected_max_abs)
    if severe_mean_shift or severe_scale_shift:
        return True, "count_and_scale_mismatch"
    return False, "count_mismatch_warning"


def audit_prediction_targets(predictions_root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    expected_cache: dict[tuple[str, str, str], np.ndarray | None] = {}

    for experiment in discover_experiments(predictions_root):
        task_name = experiment.task
        registry_name = {"phopt": "ph_opt", "binding_site": "binding_sites"}.get(
            task_name, task_name
        )
        config = TASK_REGISTRY.get(registry_name)
        if config is None or config.label_kind != "scalar_regression":
            continue

        split_family = experiment.split
        dataset_variant = SPLIT_TO_DATASET_VARIANT.get(split_family, split_family)
        if dataset_variant not in config.dataset_paths:
            continue

        split_findings = []
        for split_name in ("train", "val", "test"):
            prediction_path = _prediction_path(experiment.run_dir, split_name)
            if prediction_path is None:
                continue
            cache_key = (registry_name, dataset_variant, split_name)
            if cache_key not in expected_cache:
                expected_cache[cache_key] = _load_split_targets(
                    config, dataset_variant, split_name
                )
            expected = expected_cache[cache_key]
            if expected is None:
                continue
            try:
                observed = _load_y_true(prediction_path)
            except Exception as exc:  # noqa: BLE001 - corrupt artifacts are audit findings.
                split_findings.append(
                    {
                        "split": split_name,
                        "status": "invalid",
                        "reason": f"prediction_load_error: {exc}",
                        "prediction_file": prediction_path,
                    }
                )
                continue
            is_bad, reason = _target_mismatch(observed, expected)
            if is_bad or reason:
                split_findings.append(
                    {
                        "split": split_name,
                        "status": "invalid" if is_bad else "warning",
                        "reason": reason,
                        "prediction_file": prediction_path,
                        "observed": _summary(observed),
                        "expected_current_dataset": _summary(expected),
                    }
                )

        if any(item["status"] == "invalid" for item in split_findings):
            findings.append(
                {
                    "status": "invalid",
                    "path": {
                        "task_scope": experiment.task_scope,
                        "mode": experiment.pool_type,
                        "split_family": split_family,
                        "model_family": experiment.model_family,
                        "task_name": task_name,
                        "embedding_model": experiment.embedding_model,
                    },
                    "run_dir": experiment.run_dir,
                    "splits": split_findings,
                }
            )

    return findings


def invalidate_findings(findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        invalidation = {
            "reason": "target_payload_mismatch_current_dataset",
            "audit": finding,
        }
        marker_path = Path(finding["run_dir"]) / INVALIDATION_MARKER
        marker_path.write_text(
            json.dumps(_json_ready(invalidation), indent=4) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root", default=PROJECT_ROOT / "results/predictions", type=Path
    )
    parser.add_argument(
        "--report", default=PROJECT_ROOT / "results/target_audit_report.json", type=Path
    )
    parser.add_argument("--invalidate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = audit_prediction_targets(args.predictions_root)
    report_payload = {
        "n_invalid_entries": len(findings),
        "invalid_entries": findings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(_json_ready(report_payload), indent=4) + "\n",
        encoding="utf-8",
    )
    if args.invalidate and findings:
        invalidate_findings(findings)
    print(f"invalid_entries={len(findings)} report={args.report}")
    if args.invalidate:
        print(f"invalidated={len(findings)} predictions_root={args.predictions_root}")
    return 1 if findings and not args.invalidate else 0


if __name__ == "__main__":
    raise SystemExit(main())
