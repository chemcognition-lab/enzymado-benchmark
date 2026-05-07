#!/usr/bin/env python3
"""Reorganize result artifacts into the canonical predictions tree and rebuild results_card.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PER_RESIDUE_TASKS = {"active_sites", "binding_site"}
CLASSIFICATION_TASKS = {"binding", "ec_number", "solubility", "active_sites", "binding_site"}
LEGACY_SOURCE_ROOTS = [
    PROJECT_ROOT / "results" / "predictions" / "attention_pool",
    PROJECT_ROOT / "results" / "predictions" / "mean_pool",
    PROJECT_ROOT / "results" / "predictions_mlp",
    PROJECT_ROOT / "results" / "predictions_sgd",
    PROJECT_ROOT / "results" / "predictions_xgb",
    PROJECT_ROOT / "results" / "predictions_mean_pooled_mlp",
    PROJECT_ROOT / "results" / "predictions_mean_pooled_sgd",
    PROJECT_ROOT / "results" / "predictions_mean_pooled_xgb",
]


def load_results_card_module():
    module_path = PROJECT_ROOT / "models" / "results_card.py"
    spec = importlib.util.spec_from_file_location("results_card_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


results_card_module = load_results_card_module()
build_results_card_skeleton = results_card_module.build_results_card_skeleton
build_results_run_dir = results_card_module.build_results_run_dir
build_model_artifact_path = results_card_module.build_model_artifact_path
build_prediction_artifact_path = results_card_module.build_prediction_artifact_path
canonical_model_family = results_card_module.canonical_model_family
canonical_split_family = results_card_module.canonical_split_family
find_model_artifact = results_card_module.find_model_artifact
find_prediction_artifact = results_card_module.find_prediction_artifact
upsert_result_entry = results_card_module.upsert_result_entry
write_results_card = results_card_module.write_results_card


@dataclass(frozen=True)
class MetadataEntry:
    task_name: str
    emb_name: str
    model_name: str
    params: dict[str, Any]
    source_card: str
    source_result_key: str
    task_scope: str
    task_type: str | None
    score: tuple[int, int, int]


@dataclass(frozen=True)
class ArtifactCandidate:
    mode: str
    split_variant: str
    task_name: str
    emb_name: str
    model_name: str
    source_run_dir: Path
    model_file: Path
    train_pred_file: Path | None
    val_pred_file: Path | None
    test_pred_file: Path | None
    score: tuple[int, int, int, int]

    @property
    def logical_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.mode,
            canonical_split_family(self.split_variant),
            canonical_model_family(self.model_name),
            self.task_name,
            self.emb_name,
        )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def referenced_path_score(entry: dict[str, Any]) -> int:
    return sum(
        1
        for key in ("train_pred_file", "val_pred_file", "test_pred_file", "model_file")
        if entry.get(key) and Path(entry[key]).exists()
    )


def load_metadata_index(card_paths: list[Path], benchmark_card: dict[str, Any]) -> dict[tuple[str, str, str], MetadataEntry]:
    index: dict[tuple[str, str, str], MetadataEntry] = {}
    properties = benchmark_card.get("properties") or {}
    real_case = benchmark_card.get("real_case") or {}
    for card_priority, card_path in enumerate(card_paths):
        if not card_path.exists():
            continue
        card = read_json(card_path)
        for task_name, payload in (card.get("properties") or {}).items():
            if task_name == "fluorescence":
                continue
            task_scope = "properties" if task_name in properties else "real_case" if task_name in real_case else "properties"
            task_type = None
            if task_scope == "properties":
                task_type = (properties.get(task_name) or {}).get("task_type")
            training_results = payload.get("training_results") or {}
            for result_key, entry in training_results.items():
                emb_name = str(entry.get("embedding_model") or result_key.split("__", 1)[0])
                model_name = str(entry.get("model_name") or entry.get("name") or result_key.split("__", 1)[1])
                params = entry.get("params") or {}
                metadata = MetadataEntry(
                    task_name=task_name,
                    emb_name=emb_name,
                    model_name=model_name,
                    params=params,
                    source_card=str(card_path),
                    source_result_key=result_key,
                    task_scope=task_scope,
                    task_type=task_type,
                    score=(
                        referenced_path_score(entry),
                        len(params),
                        -card_priority,
                    ),
                )
                key = (task_name, emb_name, model_name)
                previous = index.get(key)
                if previous is None or metadata.score > previous.score:
                    index[key] = metadata
    return index


def file_count_score(run_dir: Path) -> int:
    return sum(1 for child in run_dir.iterdir() if child.is_file())


def path_priority(run_dir: Path) -> int:
    text = run_dir.as_posix()
    if "/results/predictions/attention_pool/" in text or "/results/predictions/mean_pool/" in text:
        return 40
    if "/results/predictions_mlp/" in text or "/results/predictions_sgd/" in text or "/results/predictions_xgb/" in text:
        return 35
    if "/results/predictions_mean_pooled_" in text and "/random/" in text:
        return 25
    if "/results/predictions_mean_pooled_" in text:
        return 20
    return 10


def build_candidate(
    mode: str,
    split_variant: str,
    task_name: str,
    emb_name: str,
    model_name: str,
    run_dir: Path,
) -> ArtifactCandidate:
    model_file = find_model_artifact(run_dir)
    train_pred_file = find_prediction_artifact(run_dir, "train")
    val_pred_file = find_prediction_artifact(run_dir, "val")
    test_pred_file = find_prediction_artifact(run_dir, "test")
    score = (
        int(model_file is not None) + int(train_pred_file is not None) + int(test_pred_file is not None),
        int(val_pred_file is not None),
        path_priority(run_dir),
        int(model_file.stat().st_mtime) if model_file is not None else 0,
    )
    return ArtifactCandidate(
        mode=mode,
        split_variant=split_variant,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        source_run_dir=run_dir,
        model_file=model_file,
        train_pred_file=train_pred_file,
        val_pred_file=val_pred_file,
        test_pred_file=test_pred_file,
        score=score,
    )


def infer_model_name_from_family(mode: str, model_family: str, task_name: str) -> str:
    is_classification = task_name in CLASSIFICATION_TASKS
    if mode == "attention":
        if model_family == "linear":
            return "attention_linear_classification" if is_classification else "attention_linear_regressor"
        if model_family == "mlp":
            return "attention_mlp_classification" if is_classification else "attention_mlp_regressor"
    if model_family == "linear":
        return "linear_classification" if is_classification else "linear_regressor"
    if model_family == "mlp":
        return "mlp_classification" if is_classification else "mlp_regressor"
    if model_family == "xgboost":
        return "xgb_classification" if is_classification else "xgb_regression"
    raise ValueError(f"Unsupported model family '{model_family}' for mode '{mode}'")


def parse_candidate_from_model_file(model_file: Path) -> ArtifactCandidate | None:
    parts = model_file.parts
    text = model_file.as_posix()

    if "/results/predictions/" in text:
        idx = parts.index("predictions")
        if len(parts) >= idx + 7 and parts[idx + 1] in {"attention", "mean", "per_residue"}:
            mode = parts[idx + 1]
            split_variant = parts[idx + 2]
            model_family = parts[idx + 3]
            task_name = parts[idx + 4]
            emb_name = parts[idx + 5]
            model_name = infer_model_name_from_family(mode, model_family, task_name)
            return build_candidate(mode, split_variant, task_name, emb_name, model_name, model_file.parent)
    if "/results/predictions/attention_pool/" in text:
        idx = parts.index("attention_pool")
        return build_candidate(
            "attention",
            parts[idx + 1],
            parts[idx + 2],
            parts[idx + 3],
            parts[idx + 4],
            model_file.parent,
        )
    if "/results/predictions/mean_pool/" in text:
        idx = parts.index("mean_pool")
        return build_candidate(
            "mean",
            parts[idx + 1],
            parts[idx + 2],
            parts[idx + 3],
            parts[idx + 4],
            model_file.parent,
        )
    if "/results/predictions_mlp/" in text:
        idx = parts.index("predictions_mlp")
        return build_candidate(
            "per_residue",
            parts[idx + 1],
            parts[idx + 2],
            parts[idx + 3],
            parts[idx + 4],
            model_file.parent,
        )
    if "/results/predictions_sgd/" in text:
        idx = parts.index("predictions_sgd")
        return build_candidate(
            "per_residue",
            parts[idx + 1],
            parts[idx + 2],
            parts[idx + 3],
            parts[idx + 4],
            model_file.parent,
        )
    if "/results/predictions_xgb/" in text:
        idx = parts.index("predictions_xgb")
        return build_candidate(
            "per_residue",
            parts[idx + 1],
            parts[idx + 2],
            parts[idx + 3],
            parts[idx + 4],
            model_file.parent,
        )
    if "/results/predictions_mean_pooled_" in text and "/random/" in text:
        prefix_idx = next(
            idx for idx, part in enumerate(parts)
            if part in {"predictions_mean_pooled_mlp", "predictions_mean_pooled_sgd", "predictions_mean_pooled_xgb"}
        )
        task_name = parts[prefix_idx + 2]
        mode = "per_residue" if task_name in PER_RESIDUE_TASKS else "mean"
        return build_candidate(
            mode,
            parts[prefix_idx + 1],
            task_name,
            parts[prefix_idx + 3],
            parts[prefix_idx + 4],
            model_file.parent,
        )
    if any(part in {"predictions_mean_pooled_mlp", "predictions_mean_pooled_sgd", "predictions_mean_pooled_xgb"} for part in parts):
        run_dir = model_file.parent
        try:
            emb_name, task_name, model_name = run_dir.name.split("__", 2)
        except ValueError:
            return None
        return build_candidate("mean", "random", task_name, emb_name, model_name, run_dir)
    return None


def scan_artifact_candidates() -> tuple[dict[tuple[str, str, str, str, str], ArtifactCandidate], dict[tuple[str, str, str, str, str], list[ArtifactCandidate]]]:
    best: dict[tuple[str, str, str, str, str], ArtifactCandidate] = {}
    duplicates: dict[tuple[str, str, str, str, str], list[ArtifactCandidate]] = {}
    model_files = sorted(list((PROJECT_ROOT / "results").rglob("*.pt")) + list((PROJECT_ROOT / "results").rglob("*.joblib")))
    for model_file in model_files:
        candidate = parse_candidate_from_model_file(model_file)
        if candidate is None:
            continue
        previous = best.get(candidate.logical_key)
        if previous is None or candidate.score > previous.score:
            if previous is not None:
                duplicates.setdefault(candidate.logical_key, []).append(previous)
            best[candidate.logical_key] = candidate
        else:
            duplicates.setdefault(candidate.logical_key, []).append(candidate)
    return best, duplicates


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
    try:
        root.rmdir()
    except OSError:
        pass


def reorganize_results(
    *,
    benchmark_card_path: Path,
    metadata_card_paths: list[Path],
    results_card_path: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    benchmark_card = read_json(benchmark_card_path)
    metadata_index = load_metadata_index(metadata_card_paths, benchmark_card)
    selected, duplicates = scan_artifact_candidates()
    predictions_root = PROJECT_ROOT / "results" / "predictions"

    moved = 0
    already_canonical = 0
    removed_duplicates = 0
    selected_targets: dict[tuple[str, str, str, str, str], Path] = {}
    entries_payload: list[tuple[ArtifactCandidate, Path, dict[str, Any]]] = []

    for logical_key, candidate in sorted(selected.items()):
        target_run_dir = build_results_run_dir(
            predictions_root,
            candidate.split_variant,
            candidate.task_name,
            candidate.emb_name,
            candidate.model_name,
        )
        if candidate.source_run_dir.resolve() == target_run_dir.resolve():
            already_canonical += 1
        else:
            if target_run_dir.exists():
                raise FileExistsError(f"Target run directory already exists: {target_run_dir}")
            if not dry_run:
                target_run_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidate.source_run_dir), str(target_run_dir))
            moved += 1

        selected_targets[logical_key] = target_run_dir

        metadata = metadata_index.get((candidate.task_name, candidate.emb_name, candidate.model_name))
        task_scope = "properties"
        task_type = None
        params: dict[str, Any] = {}
        metadata_source_card = None
        metadata_source_result_key = None
        if metadata is not None:
            task_scope = metadata.task_scope
            task_type = metadata.task_type
            params = metadata.params
            metadata_source_card = metadata.source_card
            metadata_source_result_key = metadata.source_result_key
        elif candidate.task_name in (benchmark_card.get("real_case") or {}):
            task_scope = "real_case"
        elif candidate.task_name in (benchmark_card.get("properties") or {}):
            task_type = (benchmark_card["properties"].get(candidate.task_name) or {}).get("task_type")

        final_run_dir = target_run_dir
        model_path = build_model_artifact_path(
            final_run_dir,
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            extension=candidate.model_file.suffix,
        )
        train_pred_path = build_prediction_artifact_path(
            final_run_dir,
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            split_name="train",
        )
        test_pred_path = build_prediction_artifact_path(
            final_run_dir,
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            split_name="test",
        )
        val_pred_path = build_prediction_artifact_path(
            final_run_dir,
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            split_name="val",
        )
        if not dry_run:
            current_model_path = final_run_dir / candidate.model_file.name
            if current_model_path.exists() and current_model_path != model_path:
                current_model_path.replace(model_path)
            if candidate.train_pred_file is not None:
                current_train_pred_path = final_run_dir / candidate.train_pred_file.name
                if current_train_pred_path.exists() and current_train_pred_path != train_pred_path:
                    current_train_pred_path.replace(train_pred_path)
            if candidate.test_pred_file is not None:
                current_test_pred_path = final_run_dir / candidate.test_pred_file.name
                if current_test_pred_path.exists() and current_test_pred_path != test_pred_path:
                    current_test_pred_path.replace(test_pred_path)
            if candidate.val_pred_file is not None:
                current_val_pred_path = final_run_dir / candidate.val_pred_file.name
                if current_val_pred_path.exists() and current_val_pred_path != val_pred_path:
                    current_val_pred_path.replace(val_pred_path)
        entry = {
            "result_key": f"{candidate.emb_name}__{candidate.model_name}",
            "embedding_model": candidate.emb_name,
            "name": candidate.model_name,
            "model_name": candidate.model_name,
            "model_family": canonical_model_family(candidate.model_name),
            "embedding_mode": candidate.mode,
            "split_family": canonical_split_family(candidate.split_variant),
            "task_name": candidate.task_name,
            "task_scope": task_scope,
            "task_type": task_type,
            "params": params,
            "run_dir": relative_to_repo(final_run_dir),
            "train_pred_file": relative_to_repo(train_pred_path),
            "test_pred_file": relative_to_repo(test_pred_path),
            "model_file": relative_to_repo(model_path),
            "metrics": {},
        }
        if val_pred_path.exists():
            entry["val_pred_file"] = relative_to_repo(val_pred_path)
        if metadata_source_card is not None:
            entry["metadata_source_card"] = metadata_source_card
        if metadata_source_result_key is not None:
            entry["metadata_source_result_key"] = metadata_source_result_key
        entries_payload.append((candidate, final_run_dir, entry))

    for logical_key, loser_list in duplicates.items():
        chosen_target = selected_targets.get(logical_key)
        for loser in loser_list:
            if dry_run:
                removed_duplicates += int(loser.source_run_dir.exists())
                continue
            if chosen_target is not None and loser.source_run_dir.exists() and loser.source_run_dir.resolve() != chosen_target.resolve():
                shutil.rmtree(loser.source_run_dir)
                removed_duplicates += 1

    if not dry_run:
        for root in LEGACY_SOURCE_ROOTS:
            remove_empty_dirs(root)

    payload = build_results_card_skeleton(benchmark_card)
    payload["metadata_cards"] = [str(path) for path in metadata_card_paths if path.exists()]
    for candidate, _, entry in entries_payload:
        upsert_result_entry(
            payload,
            task_scope=entry["task_scope"],
            task_name=candidate.task_name,
            emb_name=candidate.emb_name,
            model_name=candidate.model_name,
            split_variant=candidate.split_variant,
            result_entry=entry,
        )
    if not dry_run:
        write_results_card(results_card_path, payload)

    return {
        "selected_runs": len(selected),
        "moved_runs": moved,
        "already_canonical": already_canonical,
        "removed_duplicate_runs": removed_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move result artifacts into results/predictions/<mode>/<split>/<model>/<task>/<foundation> "
            "and rebuild results_card.json from the moved artifacts."
        )
    )
    parser.add_argument(
        "--benchmark-card",
        type=Path,
        default=PROJECT_ROOT / "benchmark_card.json",
        help="Benchmark card used for task metadata.",
    )
    parser.add_argument(
        "--metadata-card",
        dest="metadata_cards",
        action="append",
        type=Path,
        default=None,
        help="Card whose training_results entries provide params metadata. Repeatable.",
    )
    parser.add_argument(
        "--results-card",
        type=Path,
        default=PROJECT_ROOT / "results_card.json",
        help="Where to write the rebuilt results card.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the reorganization without moving files or writing JSON.",
    )
    args = parser.parse_args()

    metadata_cards = args.metadata_cards or [
        PROJECT_ROOT / "benchmark_card.json",
        PROJECT_ROOT / "benchmark_card_mean_pooled_mlp.json",
    ]
    stats = reorganize_results(
        benchmark_card_path=args.benchmark_card,
        metadata_card_paths=metadata_cards,
        results_card_path=args.results_card,
        dry_run=args.dry_run,
    )
    print(
        "Reorganized results: "
        f"selected_runs={stats['selected_runs']} "
        f"moved_runs={stats['moved_runs']} "
        f"already_canonical={stats['already_canonical']} "
        f"removed_duplicate_runs={stats['removed_duplicate_runs']}"
    )


if __name__ == "__main__":
    main()
