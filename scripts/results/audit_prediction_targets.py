#!/usr/bin/env python3
"""Audit saved scalar-regression y_true payloads against current task targets."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load as load_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import TASK_REGISTRY  # noqa: E402
from evaluation.results_layout import INVALIDATION_MARKER  # noqa: E402


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


def _iter_result_entries(card: dict[str, Any]):
    predictions = card.get("predictions", {})
    for mode, mode_bucket in predictions.items():
        if not isinstance(mode_bucket, dict):
            continue
        for split_family, split_bucket in mode_bucket.items():
            if not isinstance(split_bucket, dict):
                continue
            for model_family, model_bucket in split_bucket.items():
                if not isinstance(model_bucket, dict):
                    continue
                for task_name, task_bucket in model_bucket.items():
                    if not isinstance(task_bucket, dict):
                        continue
                    for emb_name, entry in task_bucket.items():
                        if isinstance(entry, dict):
                            yield (mode, split_family, model_family, task_name, emb_name), entry


def _load_split_targets(config, dataset_variant: str, split_name: str) -> np.ndarray | None:
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


def _prediction_path(entry: dict[str, Any], split_name: str) -> Path | None:
    for key in (f"{split_name}_pred_safetensors_file", f"{split_name}_pred_file"):
        value = entry.get(key)
        if value and str(value).endswith(".safetensors.xz"):
            return Path(value)
    return None


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
    severe_mean_shift = abs(observed_mean - expected_mean) > max(5.0, 5.0 * expected_std)
    severe_scale_shift = observed_max_abs > max(50.0, 10.0 * expected_max_abs)
    if severe_mean_shift or severe_scale_shift:
        return True, "count_and_scale_mismatch"
    return False, "count_mismatch_warning"


def audit_results_card(card: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    expected_cache: dict[tuple[str, str, str], np.ndarray | None] = {}

    for path_tokens, entry in _iter_result_entries(card):
        mode, split_family, model_family, task_name, emb_name = path_tokens
        registry_name = "ph_opt" if task_name == "phopt" else task_name
        config = TASK_REGISTRY.get(registry_name)
        if config is None or config.label_kind != "scalar_regression":
            continue

        dataset_variant = SPLIT_TO_DATASET_VARIANT.get(split_family, split_family)
        if dataset_variant not in config.dataset_paths:
            continue

        split_findings = []
        for split_name in ("train", "val", "test"):
            prediction_path = _prediction_path(entry, split_name)
            if prediction_path is None or not prediction_path.exists():
                continue
            cache_key = (registry_name, dataset_variant, split_name)
            if cache_key not in expected_cache:
                expected_cache[cache_key] = _load_split_targets(config, dataset_variant, split_name)
            expected = expected_cache[cache_key]
            if expected is None:
                continue
            observed = _load_y_true(prediction_path)
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
                        "mode": mode,
                        "split_family": split_family,
                        "model_family": model_family,
                        "task_name": task_name,
                        "embedding_model": emb_name,
                    },
                    "run_dir": entry.get("run_dir"),
                    "splits": split_findings,
                }
            )

    return findings


def invalidate_findings(card: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    by_key = {
        (
            finding["path"]["mode"],
            finding["path"]["split_family"],
            finding["path"]["model_family"],
            finding["path"]["task_name"],
            finding["path"]["embedding_model"],
        ): finding
        for finding in findings
    }
    for path_tokens, entry in _iter_result_entries(card):
        finding = by_key.get(path_tokens)
        if finding is None:
            continue
        invalidation = {
            "reason": "target_payload_mismatch_current_dataset",
            "audit": finding,
        }
        entry["invalidated"] = invalidation
        run_dir = entry.get("run_dir")
        if run_dir:
            marker_path = Path(run_dir) / INVALIDATION_MARKER
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(_json_ready(invalidation), indent=4) + "\n",
                encoding="utf-8",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-card", default=PROJECT_ROOT / "results_card.json", type=Path)
    parser.add_argument("--report", default=PROJECT_ROOT / "results/target_audit_report.json", type=Path)
    parser.add_argument("--invalidate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    card = json.loads(args.results_card.read_text(encoding="utf-8"))
    findings = audit_results_card(card)
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
        invalidate_findings(card, findings)
        args.results_card.write_text(
            json.dumps(_json_ready(card), indent=4) + "\n",
            encoding="utf-8",
        )
    print(f"invalid_entries={len(findings)} report={args.report}")
    if args.invalidate:
        print(f"invalidated={len(findings)} results_card={args.results_card}")
    return 1 if findings and not args.invalidate else 0


if __name__ == "__main__":
    raise SystemExit(main())
