#!/usr/bin/env python3
"""Compare the canonical results tree against results_card.json."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load as load_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import TASK_REGISTRY  # noqa: E402
from evaluation.results_layout import (  # noqa: E402
    INVALIDATION_MARKER,
    canonical_model_family,
    canonical_split_family,
)


SPLIT_TO_DATASET_VARIANT = {
    "random": "random",
    "loco": "similarity",
    "similarity": "similarity",
}
PHOPT_CARD_NAME = "phopt"


@dataclass(frozen=True)
class ResultKey:
    mode: str
    split_family: str
    model_family: str
    task_name: str
    embedding_model: str

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.mode,
            self.split_family,
            self.model_family,
            self.task_name,
            self.embedding_model,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "split_family": self.split_family,
            "model_family": self.model_family,
            "task_name": self.task_name,
            "embedding_model": self.embedding_model,
        }

    def label(self) -> str:
        return "/".join(self.as_tuple())


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _safe_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 - audit reports parse failures.
        return None, str(exc)


def _resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve(strict=False).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _prediction_artifact(run_dir: Path, split_name: str) -> Path | None:
    legacy = run_dir / f"{split_name}_predictions.safetensors.xz"
    if legacy.exists():
        return legacy
    matches = sorted(
        run_dir.glob(f"*_{split_name}_predictions.safetensors.xz"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return matches[-1] if matches else None


def _model_artifact(run_dir: Path) -> Path | None:
    matches = [
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.suffix in {".pt", ".joblib"}
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def _run_key_from_dir(run_dir: Path, predictions_root: Path) -> ResultKey:
    rel_parts = run_dir.relative_to(predictions_root).parts
    if len(rel_parts) != 5:
        raise ValueError(f"Expected run dir depth 5 under {predictions_root}: {run_dir}")
    return ResultKey(*rel_parts)


def _iter_run_dirs(predictions_root: Path):
    for mode_dir in sorted(path for path in predictions_root.iterdir() if path.is_dir()):
        for split_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
            for family_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for task_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
                    for emb_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                        yield emb_dir


def _scan_disk(predictions_root: Path) -> dict[ResultKey, dict[str, Any]]:
    runs: dict[ResultKey, dict[str, Any]] = {}
    for run_dir in _iter_run_dirs(predictions_root):
        key = _run_key_from_dir(run_dir, predictions_root)
        metrics_path = run_dir / "metrics.json"
        hparams_path = run_dir / "hparams.json"
        training_log_path = run_dir / "training_log.csv"
        model_path = _model_artifact(run_dir)
        predictions = {
            split: _prediction_artifact(run_dir, split)
            for split in ("train", "val", "test")
        }
        metrics, metrics_error = _safe_read_json(metrics_path) if metrics_path.exists() else (None, None)
        hparams, hparams_error = _safe_read_json(hparams_path) if hparams_path.exists() else (None, None)
        required_missing = [
            name
            for name, path in {
                "model_file": model_path,
                "train_pred_file": predictions["train"],
                "test_pred_file": predictions["test"],
                "metrics_file": metrics_path if metrics_path.exists() else None,
                "hparams_file": hparams_path if hparams_path.exists() else None,
                "training_log_file": training_log_path if training_log_path.exists() else None,
            }.items()
            if path is None
        ]
        runs[key] = {
            "run_dir": run_dir,
            "complete": not required_missing,
            "required_missing": required_missing,
            "invalid_marker": run_dir / INVALIDATION_MARKER
            if (run_dir / INVALIDATION_MARKER).exists()
            else None,
            "metrics_file": metrics_path if metrics_path.exists() else None,
            "metrics": metrics,
            "metrics_error": metrics_error,
            "hparams_file": hparams_path if hparams_path.exists() else None,
            "hparams": hparams,
            "hparams_error": hparams_error,
            "training_log_file": training_log_path if training_log_path.exists() else None,
            "model_file": model_path,
            "prediction_files": predictions,
            "artifact_counts": {
                "pt": len(list(run_dir.glob("*.pt"))),
                "joblib": len(list(run_dir.glob("*.joblib"))),
                "model_safetensors": len(list(run_dir.glob("*.safetensors"))),
                "prediction_safetensors_xz": len(list(run_dir.glob("*_predictions.safetensors.xz"))),
            },
        }
    return runs


def _iter_card_entries(card: dict[str, Any]):
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
                            yield ResultKey(mode, split_family, model_family, task_name, emb_name), entry


def _scan_card(results_card_path: Path) -> dict[ResultKey, dict[str, Any]]:
    card, error = _safe_read_json(results_card_path)
    if card is None:
        raise ValueError(f"Unable to parse {results_card_path}: {error}")
    return dict(_iter_card_entries(card))


def _float_equal(left: float, right: float, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _float_equal(float(left), float(right))
    if left is None or right is None:
        return left is right
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _registry_name(task_name: str) -> str:
    if task_name == PHOPT_CARD_NAME:
        return "ph_opt"
    if task_name == "binding_site":
        return "binding_sites"
    return task_name


def _load_expected_targets(config, dataset_variant: str, split_name: str) -> np.ndarray | None:
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


def _target_status(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[str, str | None]:
    if observed.size == 0 or expected.size == 0:
        return "ok", None
    if observed.size == expected.size:
        if np.allclose(np.sort(observed), np.sort(expected), atol=atol, rtol=rtol):
            return "ok", None
        return "invalid", "same_n_different_values"

    observed_summary = _summary(observed)
    expected_summary = _summary(expected)
    observed_mean = float(observed_summary["mean"])
    expected_mean = float(expected_summary["mean"])
    expected_std = max(float(expected_summary["std"]), 1e-6)
    observed_max_abs = float(observed_summary["max_abs"])
    expected_max_abs = max(float(expected_summary["max_abs"]), 1e-6)
    severe_mean_shift = abs(observed_mean - expected_mean) > max(5.0, 5.0 * expected_std)
    severe_scale_shift = observed_max_abs > max(50.0, 10.0 * expected_max_abs)
    if severe_mean_shift or severe_scale_shift:
        return "invalid", "count_and_scale_mismatch"
    return "warning", "count_mismatch"


def _audit_targets_from_disk(disk_runs: dict[ResultKey, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    expected_cache: dict[tuple[str, str, str], np.ndarray | None] = {}
    for key, run in disk_runs.items():
        registry_name = _registry_name(key.task_name)
        config = TASK_REGISTRY.get(registry_name)
        if config is None or config.label_kind != "scalar_regression":
            continue
        dataset_variant = SPLIT_TO_DATASET_VARIANT.get(key.split_family, key.split_family)
        if dataset_variant not in config.dataset_paths:
            continue
        split_findings = []
        for split_name, prediction_path in run["prediction_files"].items():
            if prediction_path is None:
                continue
            cache_key = (registry_name, dataset_variant, split_name)
            if cache_key not in expected_cache:
                expected_cache[cache_key] = _load_expected_targets(config, dataset_variant, split_name)
            expected = expected_cache[cache_key]
            if expected is None:
                continue
            try:
                observed = _load_y_true(prediction_path)
            except Exception as exc:  # noqa: BLE001 - audit reports corrupt prediction files.
                split_findings.append(
                    {
                        "split": split_name,
                        "status": "invalid",
                        "reason": f"prediction_load_error: {exc}",
                        "prediction_file": prediction_path,
                    }
                )
                continue
            status, reason = _target_status(observed, expected)
            if status != "ok":
                split_findings.append(
                    {
                        "split": split_name,
                        "status": status,
                        "reason": reason,
                        "prediction_file": prediction_path,
                        "observed": _summary(observed),
                        "expected_current_dataset": _summary(expected),
                    }
                )
        if split_findings:
            findings.append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "worst_status": "invalid"
                    if any(item["status"] == "invalid" for item in split_findings)
                    else "warning",
                    "splits": split_findings,
                }
            )
    return findings


def _compare_card_to_disk(
    disk_runs: dict[ResultKey, dict[str, Any]],
    card_entries: dict[ResultKey, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    disk_keys = set(disk_runs)
    card_keys = set(card_entries)
    discrepancies: dict[str, list[dict[str, Any]]] = {
        "disk_complete_not_in_card": [],
        "disk_incomplete_not_in_card": [],
        "card_not_on_disk": [],
        "card_run_dir_mismatch": [],
        "card_missing_artifact_paths": [],
        "card_metrics_bootstrap_only": [],
        "card_metrics_value_mismatch": [],
        "card_hparams_mismatch": [],
        "invalid_marker_mismatch": [],
        "incomplete_disk_runs_in_card": [],
        "ambiguous_disk_artifacts": [],
    }

    for key in sorted(disk_keys - card_keys, key=lambda item: item.as_tuple()):
        bucket = "disk_complete_not_in_card" if disk_runs[key]["complete"] else "disk_incomplete_not_in_card"
        discrepancies[bucket].append(
            {
                "path": key.as_dict(),
                "run_dir": disk_runs[key]["run_dir"],
                "required_missing": disk_runs[key]["required_missing"],
                "invalid_marker": disk_runs[key]["invalid_marker"],
            }
        )

    for key in sorted(card_keys - disk_keys, key=lambda item: item.as_tuple()):
        entry = card_entries[key]
        discrepancies["card_not_on_disk"].append(
            {
                "path": key.as_dict(),
                "run_dir": entry.get("run_dir"),
                "model_file": entry.get("model_file"),
            }
        )

    for key in sorted(disk_keys & card_keys, key=lambda item: item.as_tuple()):
        run = disk_runs[key]
        entry = card_entries[key]
        card_run_dir = _resolve_path(entry.get("run_dir"))
        disk_run_dir = run["run_dir"].resolve(strict=False)
        if card_run_dir is not None and card_run_dir != disk_run_dir:
            discrepancies["card_run_dir_mismatch"].append(
                {
                    "path": key.as_dict(),
                    "disk_run_dir": disk_run_dir,
                    "card_run_dir": card_run_dir,
                }
            )
        if not run["complete"]:
            discrepancies["incomplete_disk_runs_in_card"].append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "required_missing": run["required_missing"],
                    "invalid_marker": run["invalid_marker"],
                }
            )

        missing_fields = []
        for field_name in (
            "model_file",
            "train_pred_file",
            "test_pred_file",
            "metrics_file",
            "hparams_file",
            "training_log_file",
        ):
            field_path = _resolve_path(entry.get(field_name))
            if field_path is None or not field_path.exists():
                missing_fields.append(field_name)
        if missing_fields:
            discrepancies["card_missing_artifact_paths"].append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "missing_fields": missing_fields,
                }
            )

        if run["metrics"] is not None and entry.get("metrics") is not None:
            if not _json_equal(entry.get("metrics"), run["metrics"]):
                metrics_without_bootstrap = {
                    metric_key: metric_value
                    for metric_key, metric_value in run["metrics"].items()
                    if metric_key != "test_bootstrap"
                }
                bucket = (
                    "card_metrics_bootstrap_only"
                    if _json_equal(entry.get("metrics"), metrics_without_bootstrap)
                    else "card_metrics_value_mismatch"
                )
                discrepancies[bucket].append(
                    {
                        "path": key.as_dict(),
                        "run_dir": run["run_dir"],
                        "metrics_file": run["metrics_file"],
                    }
                )

        hparams = run["hparams"] or {}
        hparam_issues = []
        if hparams.get("task_name") and hparams.get("task_name") != key.task_name:
            hparam_issues.append({"field": "task_name", "disk": key.task_name, "hparams": hparams.get("task_name")})
        if hparams.get("embedding_model") and hparams.get("embedding_model") != key.embedding_model:
            hparam_issues.append(
                {
                    "field": "embedding_model",
                    "disk": key.embedding_model,
                    "hparams": hparams.get("embedding_model"),
                }
            )
        if hparams.get("model_name"):
            expected_family = canonical_model_family(hparams["model_name"])
            if expected_family != key.model_family:
                hparam_issues.append(
                    {
                        "field": "model_family",
                        "disk": key.model_family,
                        "hparams_model_name": hparams["model_name"],
                        "canonical_hparams_family": expected_family,
                    }
                )
        if hparams.get("split_variant"):
            expected_split = canonical_split_family(hparams["split_variant"])
            if expected_split != key.split_family:
                hparam_issues.append(
                    {
                        "field": "split_family",
                        "disk": key.split_family,
                        "hparams_split_variant": hparams["split_variant"],
                        "canonical_hparams_split": expected_split,
                    }
                )
        if hparam_issues:
            discrepancies["card_hparams_mismatch"].append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "issues": hparam_issues,
                }
            )

        marker_exists = run["invalid_marker"] is not None
        card_invalidated = bool(entry.get("invalidated"))
        if marker_exists != card_invalidated:
            discrepancies["invalid_marker_mismatch"].append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "disk_invalid_marker": marker_exists,
                    "card_invalidated": card_invalidated,
                }
            )

        artifact_counts = run["artifact_counts"]
        ambiguous = {
            name: count
            for name, count in artifact_counts.items()
            if (name in {"pt", "joblib"} and count > 1)
            or (name == "prediction_safetensors_xz" and count not in {2, 3})
        }
        if ambiguous:
            discrepancies["ambiguous_disk_artifacts"].append(
                {
                    "path": key.as_dict(),
                    "run_dir": run["run_dir"],
                    "artifact_counts": artifact_counts,
                }
            )

    return discrepancies


def _summarize(
    disk_runs: dict[ResultKey, dict[str, Any]],
    card_entries: dict[ResultKey, dict[str, Any]],
    discrepancies: dict[str, list[dict[str, Any]]],
    target_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_runs = sum(1 for run in disk_runs.values() if run["complete"])
    target_invalid = sum(1 for item in target_findings if item["worst_status"] == "invalid")
    target_warnings = sum(1 for item in target_findings if item["worst_status"] == "warning")
    return {
        "disk_run_dirs": len(disk_runs),
        "disk_complete_runs": complete_runs,
        "disk_incomplete_runs": len(disk_runs) - complete_runs,
        "card_entries": len(card_entries),
        "discrepancy_counts": {
            name: len(items)
            for name, items in discrepancies.items()
        },
        "target_payload_findings": len(target_findings),
        "target_payload_invalid": target_invalid,
        "target_payload_warnings": target_warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", default=PROJECT_ROOT / "results/predictions", type=Path)
    parser.add_argument("--results-card", default=PROJECT_ROOT / "results_card.json", type=Path)
    parser.add_argument("--report", default=PROJECT_ROOT / "results/results_tree_audit_report.json", type=Path)
    parser.add_argument("--skip-target-audit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disk_runs = _scan_disk(args.predictions_root)
    card_entries = _scan_card(args.results_card)
    discrepancies = _compare_card_to_disk(disk_runs, card_entries)
    target_findings = [] if args.skip_target_audit else _audit_targets_from_disk(disk_runs)
    report = {
        "summary": _summarize(disk_runs, card_entries, discrepancies, target_findings),
        "discrepancies": discrepancies,
        "target_payload_findings": target_findings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(_json_ready(report), indent=4) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print(f"report={_rel(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
