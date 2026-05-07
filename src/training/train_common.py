"""Shared utilities used across training pipelines."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from data_loading import TASK_REGISTRY
from data_loading.splits import clean_text, normalize_sequence
from training.results_card import (
    INVALIDATION_MARKER,
    build_results_run_dir,
    find_model_artifact,
    find_prediction_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_LABEL_KINDS: set[str] = set()
CARD_TO_REGISTRY: dict[str, str] = {
    "phopt": "ph_opt",
    "binding_site": "binding_sites",
}
PREFERRED_SEQUENCE_INDEX_PATH = PROJECT_ROOT / "data/processed/protein_sequences_unique.csv"
LEGACY_SEQUENCE_INDEX_PATH = PROJECT_ROOT / "data/processed/protein_sequences_unique_no_fluorescence.csv"


def log(msg: str) -> None:
    """Print a timestamped log line that stays readable in batch-job logs."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def build_result_run_dir(
    output_root: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    model_name: str,
) -> Path:
    """Return a deterministic result directory under the canonical results tree."""
    return build_results_run_dir(output_root, split_variant, task_name, emb_name, model_name)


def discover_embedding_names(result_root: Path) -> list[str]:
    """Return embedding directory names from the canonical results tree."""
    discovered: set[str] = set()
    if not result_root.exists():
        return []
    for mode_dir in result_root.iterdir():
        if not mode_dir.is_dir():
            continue
        for split_dir in mode_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for model_dir in split_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for scope_name in ("properties", "real_case"):
                    scope_dir = model_dir / scope_name
                    if scope_dir.is_dir():
                        for task_dir in scope_dir.iterdir():
                            if not task_dir.is_dir():
                                continue
                            for emb_dir in task_dir.iterdir():
                                if emb_dir.is_dir():
                                    discovered.add(emb_dir.name)
                # Future-proof plain directory layout on disk where task folders
                # sit directly under the model family.
                for task_dir in model_dir.iterdir():
                    if not task_dir.is_dir() or task_dir.name in {"properties", "real_case"}:
                        continue
                    for emb_dir in task_dir.iterdir():
                        if emb_dir.is_dir():
                            discovered.add(emb_dir.name)
    return sorted(discovered)


def resolve_sequence_index_path() -> Path | None:
    """Return the preferred sequence index, falling back to the legacy filename."""
    if PREFERRED_SEQUENCE_INDEX_PATH.exists():
        return PREFERRED_SEQUENCE_INDEX_PATH
    if LEGACY_SEQUENCE_INDEX_PATH.exists():
        return LEGACY_SEQUENCE_INDEX_PATH
    return None


def resolve_registry_name(card_task_name: str) -> str | None:
    """Map dataset-card task names onto the internal task registry when needed."""
    name = CARD_TO_REGISTRY.get(card_task_name, card_task_name)
    return name if name in TASK_REGISTRY else None


def resolve_split_variant(default_split: str | None) -> str | None:
    """Infer the split family from the card's default split filename."""
    if not default_split:
        return None
    if "random" in default_split:
        return "random"
    if "similarity" in default_split:
        return "similarity"
    if "loco" in default_split:
        return "loco"
    return None


def load_allowed_ids(path: str | Path) -> set[str]:
    """Load the optional allowlist used to restrict training to shared embeddings."""
    allowed_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                allowed_ids.add(value)
    return allowed_ids


def load_sequence_to_id_index(index_path: Path) -> dict[str, str]:
    """Load a normalized sequence-to-sequence-id mapping for embedding filters."""
    mapping: dict[str, str] = {}
    with open(index_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("protein_sequence", "")
            sequence_id = row.get("sequence_id", "")
            normalized_sequence = normalize_sequence(sequence)
            normalized_id = clean_text(sequence_id)
            if normalized_sequence and normalized_id:
                mapping[normalized_sequence] = normalized_id
    return mapping


def candidate_embedding_keys(
    row: dict[str, str],
    sequence_column: str,
    id_columns: tuple[str, ...],
    sequence_to_id: dict[str, str],
) -> set[str]:
    """Collect every embedding key that could match a dataset row."""
    candidates: set[str] = set()

    sequence = normalize_sequence(row.get(sequence_column, ""))
    if sequence:
        sequence_id = sequence_to_id.get(sequence)
        if sequence_id:
            candidates.add(sequence_id)
        candidates.add(sequence)

    for column in id_columns:
        value = clean_text(row.get(column, ""))
        if value:
            candidates.add(value)

    return candidates


def row_matches_allowed_ids(
    row: dict[str, str],
    config,
    allowed_ids: set[str],
    sequence_to_id: dict[str, str],
) -> bool:
    """Return whether a row can be matched to at least one allowed embedding key."""
    main_candidates = candidate_embedding_keys(
        row,
        sequence_column=config.sequence_column,
        id_columns=config.sample_id_candidates,
        sequence_to_id=sequence_to_id,
    )
    if not main_candidates or not any(key in allowed_ids for key in main_candidates):
        return False

    if config.batch_kind == "mutation_delta_global" and config.wildtype_sequence_column:
        wildtype_candidates = candidate_embedding_keys(
            row,
            sequence_column=config.wildtype_sequence_column,
            id_columns=(),
            sequence_to_id=sequence_to_id,
        )
        if not wildtype_candidates or not any(key in allowed_ids for key in wildtype_candidates):
            return False

    return True


def filter_rows_to_allowed_ids(
    rows: list[dict[str, str]],
    config,
    allowed_ids: set[str],
    sequence_to_id: dict[str, str],
) -> list[dict[str, str]]:
    """Keep only rows whose embedding ids survive the optional allowlist filter."""
    return [
        row for row in rows
        if row_matches_allowed_ids(row, config, allowed_ids, sequence_to_id)
    ]


__all__ = [
    "PROJECT_ROOT",
    "SKIP_LABEL_KINDS",
    "build_result_run_dir",
    "candidate_embedding_keys",
    "discover_embedding_names",
    "filter_rows_to_allowed_ids",
    "is_run_complete",
    "load_allowed_ids",
    "load_sequence_to_id_index",
    "log",
    "resolve_registry_name",
    "resolve_sequence_index_path",
    "resolve_split_variant",
    "row_matches_allowed_ids",
]
