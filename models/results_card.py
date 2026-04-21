"""Helpers for organizing and persisting model artifacts in results_card.json."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any


RESULTS_CARD_VERSION = 2
RESULTS_CARD_NAME = "enzymado_results"
PER_RESIDUE_TASKS = {"active_sites", "binding_site"}
SPLIT_FAMILY_ALIASES = {
    "random": "random",
    "similarity": "loco",
    "loco": "loco",
}
MODEL_FAMILY_ALIASES = {
    "attention_linear_classification": "linear",
    "attention_linear_regressor": "linear",
    "attention_mlp_classification": "mlp",
    "attention_mlp_regressor": "mlp",
    "sgd_classification": "linear",
    "sgd_regressor": "linear",
    "mlp_classification": "mlp",
    "mlp_regressor": "mlp",
    "xgb_classification": "xgboost",
    "xgb_regression": "xgboost",
}


def _path_token(value: str) -> str:
    """Normalize dynamic path components used in artifact trees."""
    return value.strip().replace("/", "_")


def canonical_split_family(split_variant: str | None) -> str:
    """Normalize split names to the results-card vocabulary."""
    if split_variant is None:
        return "random"
    value = split_variant.strip().lower()
    for needle, canonical in SPLIT_FAMILY_ALIASES.items():
        if needle in value:
            return canonical
    return _path_token(value)


def canonical_model_family(model_name: str) -> str:
    """Map task-specific estimator names onto broad model families."""
    if model_name in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[model_name]
    if model_name.startswith("attention_linear_"):
        return "linear"
    if model_name.startswith("attention_mlp_"):
        return "mlp"
    if model_name.startswith("mlp_"):
        return "mlp"
    if model_name.startswith("sgd_"):
        return "linear"
    if model_name.startswith("xgb_"):
        return "xgboost"
    return _path_token(model_name)


def canonical_embedding_mode(task_name: str, model_name: str) -> str:
    """Return the top-level results-card bucket for one run."""
    if model_name.startswith("attention_"):
        return "attention"
    if task_name in PER_RESIDUE_TASKS:
        return "per_residue"
    return "mean"


def build_results_run_dir(
    output_root: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    model_name: str,
) -> Path:
    """Return the canonical on-disk artifact path for one trained run."""
    return (
        output_root
        / canonical_embedding_mode(task_name, model_name)
        / canonical_split_family(split_variant)
        / canonical_model_family(model_name)
        / _path_token(task_name)
        / _path_token(emb_name)
    )


def build_artifact_stem(
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
) -> str:
    """Return the canonical filename prefix for one trained run's artifacts."""
    return "_".join(
        (
            _path_token(model_name),
            _path_token(emb_name),
            _path_token(task_name),
            canonical_split_family(split_variant),
            canonical_embedding_mode(task_name, model_name),
        )
    )


def build_prediction_artifact_path(
    run_dir: Path,
    *,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    split_name: str,
) -> Path:
    """Return the canonical prediction filename for one split."""
    stem = build_artifact_stem(task_name, emb_name, model_name, split_variant)
    return run_dir / f"{stem}_{_path_token(split_name)}_predictions.npz"


def build_model_artifact_path(
    run_dir: Path,
    *,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    extension: str,
) -> Path:
    """Return the canonical model/checkpoint filename for one run."""
    stem = build_artifact_stem(task_name, emb_name, model_name, split_variant)
    return run_dir / f"{stem}.{extension.lstrip('.')}"


def find_prediction_artifact(run_dir: Path, split_name: str) -> Path | None:
    """Find a prediction artifact, supporting both legacy and canonical filenames."""
    legacy_path = run_dir / f"{split_name}_predictions.npz"
    if legacy_path.exists():
        return legacy_path
    matches = [path for path in run_dir.glob(f"*_{_path_token(split_name)}_predictions.npz") if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def find_model_artifact(run_dir: Path) -> Path | None:
    """Find a saved model artifact, supporting both legacy and canonical filenames."""
    for legacy_name in ("model.pt", "model.joblib"):
        legacy_path = run_dir / legacy_name
        if legacy_path.exists():
            return legacy_path
    matches = [path for path in run_dir.glob("*.pt") if path.is_file()]
    matches.extend(path for path in run_dir.glob("*.joblib") if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def build_results_card_skeleton(benchmark_card: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a fresh results card with lightweight dataset metadata."""
    benchmark_card = benchmark_card or {}
    property_names = sorted(
        name for name in (benchmark_card.get("properties") or {}).keys()
        if name != "fluorescence"
    )
    return {
        "results_card_name": RESULTS_CARD_NAME,
        "results_card_version": RESULTS_CARD_VERSION,
        "benchmark_name": benchmark_card.get("benchmark_name"),
        "benchmark_version": benchmark_card.get("benchmark_version"),
        "properties": property_names,
        "real_case": sorted((benchmark_card.get("real_case") or {}).keys()),
        "predictions_root": "results/predictions",
        "predictions": {},
    }


def find_result_entry(
    card: dict[str, Any],
    *,
    task_scope: str,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
) -> dict[str, Any] | None:
    """Look up one entry in the nested results-card structure."""
    embedding_mode = canonical_embedding_mode(task_name, model_name)
    split_family = canonical_split_family(split_variant)
    model_family = canonical_model_family(model_name)
    return (
        card.get("predictions", {})
        .get(embedding_mode, {})
        .get(split_family, {})
        .get(model_family, {})
        .get(task_name, {})
        .get(emb_name)
    )


def is_result_entry_complete(entry: dict[str, Any] | None) -> bool:
    """Return whether a results-card entry points to a finished artifact set."""
    if not entry:
        return False
    model_file = entry.get("model_file")
    test_pred_file = entry.get("test_pred_file")
    if not model_file or not test_pred_file:
        return False
    return Path(model_file).exists() and Path(test_pred_file).exists()


def ensure_results_card(path: Path, benchmark_card: dict[str, Any] | None = None) -> None:
    """Create results_card.json if it does not yet exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_results_card_skeleton(benchmark_card)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def read_results_card(path: Path) -> dict[str, Any]:
    """Load a results-card JSON payload."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results_card(path: Path, payload: dict[str, Any]) -> None:
    """Write a results-card JSON payload with stable formatting."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def upsert_result_entry(
    card: dict[str, Any],
    *,
    task_scope: str,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    result_entry: dict[str, Any],
) -> None:
    """Insert or replace one result entry in the nested results-card structure."""
    embedding_mode = canonical_embedding_mode(task_name, model_name)
    split_family = canonical_split_family(split_variant)
    model_family = canonical_model_family(model_name)
    predictions = card.setdefault("predictions", {})
    mode_bucket = predictions.setdefault(embedding_mode, {})
    split_bucket = mode_bucket.setdefault(split_family, {})
    model_bucket = split_bucket.setdefault(model_family, {})
    task_bucket = model_bucket.setdefault(task_name, {})
    task_bucket[emb_name] = result_entry


def update_results_card_locked(
    results_card_path: Path,
    *,
    task_scope: str,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    result_entry: dict[str, Any],
    benchmark_card: dict[str, Any] | None = None,
) -> None:
    """Thread/process-safe update of a single entry in results_card.json."""
    ensure_results_card(results_card_path, benchmark_card=benchmark_card)
    lock_path = results_card_path.with_suffix(".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            card = read_results_card(results_card_path)
            upsert_result_entry(
                card,
                task_scope=task_scope,
                task_name=task_name,
                emb_name=emb_name,
                model_name=model_name,
                split_variant=split_variant,
                result_entry=result_entry,
            )
            write_results_card(results_card_path, card)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


__all__ = [
    "PER_RESIDUE_TASKS",
    "RESULTS_CARD_NAME",
    "RESULTS_CARD_VERSION",
    "build_results_card_skeleton",
    "build_results_run_dir",
    "build_artifact_stem",
    "build_model_artifact_path",
    "build_prediction_artifact_path",
    "canonical_embedding_mode",
    "canonical_model_family",
    "canonical_split_family",
    "ensure_results_card",
    "find_result_entry",
    "find_model_artifact",
    "find_prediction_artifact",
    "is_result_entry_complete",
    "read_results_card",
    "update_results_card_locked",
    "upsert_result_entry",
    "write_results_card",
]
