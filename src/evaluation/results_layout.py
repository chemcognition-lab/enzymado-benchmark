"""Helpers for organizing and discovering benchmark result artifacts."""

from __future__ import annotations

from pathlib import Path

INVALIDATION_MARKER = "target_audit_invalid.json"
PER_RESIDUE_TASKS = {"active_sites", "binding_site"}
SPLIT_FAMILIES = ("random", "similarity")
MODEL_FAMILY_ALIASES = {
    "attention_linear_classification": "linear",
    "attention_linear_regressor": "linear",
    "attention_mlp_classification": "mlp",
    "attention_mlp_regressor": "mlp",
    "linear_classification": "linear",
    "linear_regressor": "linear",
    "mlp_classification": "mlp",
    "mlp_regressor": "mlp",
    "xgb_classification": "xgboost",
    "xgb_regression": "xgboost",
}


def _path_token(value: str) -> str:
    """Normalize dynamic path components used in artifact trees."""
    return value.strip().replace("/", "_")


def canonical_split_family(split_variant: str | None) -> str:
    """Normalize split names to the canonical results-tree vocabulary."""
    if split_variant is None:
        return "random"
    value = split_variant.strip().lower()
    for family in SPLIT_FAMILIES:
        if family in value:
            return family
    raise ValueError(f"Unknown split variant '{split_variant}'.")


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

    if model_name.startswith("xgb_"):
        return "xgboost"
    return _path_token(model_name)


def canonical_embedding_mode(task_name: str, model_name: str) -> str:
    """Return the embedding-mode directory for one run."""
    if model_name.startswith("attention_"):
        return "attention"
    return "no_attention"


def build_results_run_dir(
    output_root: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    model_name: str,
    *,
    task_scope: str,
) -> Path:
    """Return the canonical on-disk artifact path for one trained run.

    ``task_scope`` ("properties" or "real_case", from ``TaskConfig.task_scope``)
    is the top-level bucket so the two experiment families live in separate
    subtrees instead of interleaved under the same attention/no_attention dirs.
    """
    return (
        output_root
        / _path_token(task_scope)
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
    return run_dir / f"{stem}_{_path_token(split_name)}_predictions.safetensors.xz"


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
    """Find a safetensors prediction artifact."""
    for legacy_name in (f"{split_name}_predictions.safetensors.xz",):
        legacy_path = run_dir / legacy_name
        if legacy_path.exists():
            return legacy_path
    matches = [
        path
        for path in run_dir.glob(
            f"*_{_path_token(split_name)}_predictions.safetensors.xz"
        )
        if path.is_file()
    ]
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
    matches.extend(path for path in run_dir.glob("*.safetensors") if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def is_run_complete(run_dir: Path) -> bool:
    """A run is complete when the current artifact set exists on disk."""
    if (run_dir / INVALIDATION_MARKER).exists():
        return False
    has_predictions = (
        find_prediction_artifact(run_dir, "train") is not None
        and find_prediction_artifact(run_dir, "test") is not None
    )
    has_model = find_model_artifact(run_dir) is not None
    has_metadata = all(
        (run_dir / filename).is_file()
        for filename in ("hparams.json", "metrics.json", "training_log.csv")
    )
    return has_predictions and has_model and has_metadata


__all__ = [
    "INVALIDATION_MARKER",
    "PER_RESIDUE_TASKS",
    "build_results_run_dir",
    "build_artifact_stem",
    "build_model_artifact_path",
    "build_prediction_artifact_path",
    "canonical_embedding_mode",
    "canonical_model_family",
    "canonical_split_family",
    "is_run_complete",
    "find_model_artifact",
    "find_prediction_artifact",
]
