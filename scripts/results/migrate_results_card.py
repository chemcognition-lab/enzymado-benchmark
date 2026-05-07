#!/usr/bin/env python3
"""Build results_card.json from legacy training_results registries."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_results_card_module():
    module_path = PROJECT_ROOT / "models" / "results_card.py"
    spec = importlib.util.spec_from_file_location("results_card_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


results_card_module = load_results_card_module()
build_results_card_skeleton = results_card_module.build_results_card_skeleton
canonical_embedding_mode = results_card_module.canonical_embedding_mode
canonical_model_family = results_card_module.canonical_model_family
canonical_split_family = results_card_module.canonical_split_family
upsert_result_entry = results_card_module.upsert_result_entry
write_results_card = results_card_module.write_results_card


def resolve_split_variant(default_split: str | None) -> str | None:
    if not default_split:
        return None
    if "random" in default_split:
        return "random"
    if "similarity" in default_split:
        return "similarity"
    if "loco" in default_split:
        return "loco"
    return None


@dataclass(frozen=True)
class Candidate:
    """One migrated result-entry candidate plus its ranking score."""

    logical_key: tuple[str, str, str, str, str, str]
    score: tuple[int, int, int, int]
    task_scope: str
    task_name: str
    emb_name: str
    model_name: str
    split_variant: str
    result_entry: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_task_scope(task_name: str, benchmark_card: dict[str, Any]) -> str:
    if task_name in (benchmark_card.get("properties") or {}):
        return "properties"
    if task_name in (benchmark_card.get("real_case") or {}):
        return "real_case"
    return "properties"


def infer_run_dir(entry: dict[str, Any]) -> str | None:
    run_dir = entry.get("run_dir")
    if run_dir:
        return str(run_dir)
    model_file = entry.get("model_file")
    if model_file:
        return str(Path(model_file).parent)
    return None


def infer_val_pred_file(entry: dict[str, Any]) -> str | None:
    explicit = entry.get("val_pred_file")
    if explicit:
        return str(explicit)
    run_dir = infer_run_dir(entry)
    if run_dir is None:
        return None
    candidate = Path(run_dir) / "val_predictions.npz"
    if candidate.exists():
        return str(candidate)
    return None


def infer_split_variant(
    entry: dict[str, Any],
    task_name: str,
    benchmark_card: dict[str, Any],
) -> str:
    haystack = " ".join(
        str(value)
        for value in (
            entry.get("run_dir"),
            entry.get("train_pred_file"),
            entry.get("val_pred_file"),
            entry.get("test_pred_file"),
            entry.get("model_file"),
        )
        if value
    ).lower()
    if "/loco/" in haystack or "similarity" in haystack or "_loco_" in haystack:
        return "loco"
    if "/random/" in haystack or "random_" in haystack:
        return "random"

    prop = (benchmark_card.get("properties") or {}).get(task_name)
    if prop:
        dataset_info = (prop.get("datasets") or [{}])[0]
        default_split = dataset_info.get("splits", {}).get("default_split")
        resolved = resolve_split_variant(default_split)
        if resolved is not None:
            return resolved
    return "random"


def artifact_completeness(entry: dict[str, Any]) -> int:
    paths = [
        entry.get("train_pred_file"),
        entry.get("val_pred_file"),
        entry.get("test_pred_file"),
        entry.get("model_file"),
    ]
    return sum(1 for raw in paths if raw and Path(raw).exists())


def metadata_richness(entry: dict[str, Any]) -> int:
    params = entry.get("params") or {}
    return len(params) + (1 if entry.get("val_pred_file") else 0) + (1 if entry.get("run_dir") else 0)


def normalize_entry(
    *,
    task_name: str,
    task_scope: str,
    result_key: str,
    entry: dict[str, Any],
    split_variant: str,
    source_card_path: Path,
) -> tuple[str, dict[str, Any]]:
    emb_name = str(entry.get("embedding_model") or result_key.split("__", 1)[0])
    model_name = str(entry.get("model_name") or entry.get("name") or result_key.split("__", 1)[1])
    run_dir = infer_run_dir(entry)
    normalized = {
        "result_key": result_key,
        "embedding_model": emb_name,
        "name": model_name,
        "model_name": model_name,
        "model_family": canonical_model_family(model_name),
        "embedding_mode": canonical_embedding_mode(task_name, model_name),
        "split_family": canonical_split_family(split_variant),
        "task_name": task_name,
        "task_scope": task_scope,
        "params": entry.get("params") or {},
        "source_card": str(source_card_path),
        "train_pred_file": entry.get("train_pred_file"),
        "test_pred_file": entry.get("test_pred_file"),
        "model_file": entry.get("model_file"),
    }
    if run_dir is not None:
        normalized["run_dir"] = run_dir
    val_pred_file = infer_val_pred_file(entry)
    if val_pred_file is not None:
        normalized["val_pred_file"] = val_pred_file
    return emb_name, normalized


def collect_candidates(
    benchmark_card: dict[str, Any],
    source_card_path: Path,
    source_index: int,
) -> list[Candidate]:
    source_card = read_json(source_card_path)
    candidates: list[Candidate] = []
    for task_name, payload in (source_card.get("properties") or {}).items():
        training_results = payload.get("training_results") or {}
        task_scope = infer_task_scope(task_name, benchmark_card)
        for result_key, raw_entry in training_results.items():
            split_variant = infer_split_variant(raw_entry, task_name, benchmark_card)
            emb_name, entry = normalize_entry(
                task_name=task_name,
                task_scope=task_scope,
                result_key=result_key,
                entry=raw_entry,
                split_variant=split_variant,
                source_card_path=source_card_path,
            )
            model_name = entry["model_name"]
            logical_key = (
                task_scope,
                task_name,
                entry["embedding_mode"],
                entry["split_family"],
                entry["model_family"],
                emb_name,
            )
            score = (
                artifact_completeness(entry),
                metadata_richness(entry),
                -source_index,
                0,
            )
            candidates.append(
                Candidate(
                    logical_key=logical_key,
                    score=score,
                    task_scope=task_scope,
                    task_name=task_name,
                    emb_name=emb_name,
                    model_name=model_name,
                    split_variant=split_variant,
                    result_entry=entry,
                )
            )
    return candidates


def migrate_results_card(
    *,
    benchmark_card_path: Path,
    source_card_paths: list[Path],
    output_path: Path,
) -> dict[str, int]:
    benchmark_card = read_json(benchmark_card_path)
    payload = build_results_card_skeleton(benchmark_card)
    payload["source_cards"] = [str(path) for path in source_card_paths]

    best_by_key: dict[tuple[str, str, str, str, str, str], Candidate] = {}
    replaced = 0
    ignored = 0
    for source_index, source_card_path in enumerate(source_card_paths):
        for candidate in collect_candidates(benchmark_card, source_card_path, source_index):
            previous = best_by_key.get(candidate.logical_key)
            if previous is None or candidate.score > previous.score:
                if previous is not None:
                    replaced += 1
                best_by_key[candidate.logical_key] = candidate
            else:
                ignored += 1

    for candidate in best_by_key.values():
        upsert_result_entry(
            payload,
            task_scope=candidate.task_scope,
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            result_entry=candidate.result_entry,
        )

    write_results_card(output_path, payload)
    return {
        "selected_entries": len(best_by_key),
        "replaced_candidates": replaced,
        "ignored_candidates": ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create results_card.json from one or more benchmark cards that still contain "
            "legacy properties.<task>.training_results entries."
        )
    )
    parser.add_argument(
        "--benchmark-card",
        type=Path,
        default=PROJECT_ROOT / "benchmark_card.json",
        help="Dataset benchmark card used to seed the task catalog.",
    )
    parser.add_argument(
        "--source-card",
        dest="source_cards",
        action="append",
        type=Path,
        default=None,
        help="Legacy benchmark card to harvest training_results from. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results_card.json",
        help="Path to the results card to write.",
    )
    args = parser.parse_args()

    source_cards = args.source_cards or [
        PROJECT_ROOT / "benchmark_card.json",
        PROJECT_ROOT / "benchmark_card_mean_pooled_mlp.json",
    ]
    stats = migrate_results_card(
        benchmark_card_path=args.benchmark_card,
        source_card_paths=source_cards,
        output_path=args.output,
    )
    print(
        "Migrated results_card.json: "
        f"selected_entries={stats['selected_entries']} "
        f"replaced_candidates={stats['replaced_candidates']} "
        f"ignored_candidates={stats['ignored_candidates']}"
    )


if __name__ == "__main__":
    main()
