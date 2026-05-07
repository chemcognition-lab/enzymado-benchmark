#!/usr/bin/env python3
"""Sync training_results entries between benchmark cards.

Typical use:

    python scripts/results/sync_training_results.py \
        --source-card /path/to/other/repo/benchmark_card.json \
        --target-card benchmark_card.json \
        --source-root /path/to/other/repo \
        --target-root . \
        --copy-artifacts

When ``--copy-artifacts`` is enabled, the script copies each referenced run
directory into ``target-root`` and rewrites the merged card entries so their
artifact paths point at the local copy.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path


ARTIFACT_FIELDS = (
    "train_pred_file",
    "val_pred_file",
    "test_pred_file",
    "model_file",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def resolve_artifact_path(raw_path: str, source_root: Path) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return source_root / path_obj


def relative_artifact_path(raw_path: str, resolved_path: Path, source_root: Path) -> Path:
    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        return path_obj
    try:
        return resolved_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"Absolute artifact path '{resolved_path}' is outside source root '{source_root}'. "
            "Pass --source-root that contains the run outputs, or fix the source card first."
        ) from exc


def copy_run_dir(source_dir: Path, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def sync_entry_artifacts(
    entry: dict,
    source_root: Path,
    target_root: Path,
    *,
    copy_artifacts: bool,
    absolute_paths: bool,
    dry_run: bool,
    copied_run_dirs: set[Path],
) -> dict:
    updated_entry = copy.deepcopy(entry)
    artifact_plan: list[tuple[str, Path, Path]] = []

    for field in ARTIFACT_FIELDS:
        raw_value = updated_entry.get(field)
        if not raw_value:
            continue
        resolved_source_path = resolve_artifact_path(str(raw_value), source_root)
        relative_path = relative_artifact_path(str(raw_value), resolved_source_path, source_root)
        artifact_plan.append((field, resolved_source_path, relative_path))

    if not copy_artifacts:
        return updated_entry

    for _field, resolved_source_path, relative_path in artifact_plan:
        source_run_dir = resolved_source_path.parent
        target_run_dir = (target_root / relative_path).parent
        if source_run_dir in copied_run_dirs:
            continue
        if not source_run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {source_run_dir}")
        copy_run_dir(source_run_dir, target_run_dir, dry_run=dry_run)
        copied_run_dirs.add(source_run_dir)

    for field, _resolved_source_path, relative_path in artifact_plan:
        target_path = target_root / relative_path
        updated_entry[field] = str(target_path if absolute_paths else relative_path)

    return updated_entry


def sync_training_results(
    source_card_path: Path,
    target_card_path: Path,
    *,
    source_root: Path | None = None,
    target_root: Path | None = None,
    copy_artifacts: bool = False,
    overwrite: bool = False,
    only_tasks: set[str] | None = None,
    only_keys: set[str] | None = None,
    absolute_paths: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    source_card_path = source_card_path.resolve()
    target_card_path = target_card_path.resolve()
    source_root = (source_root or source_card_path.parent).resolve()
    target_root = (target_root or target_card_path.parent).resolve()
    only_tasks = only_tasks or set()
    only_keys = only_keys or set()

    source_card = read_json(source_card_path)
    target_card = read_json(target_card_path)

    source_properties = source_card.get("properties", {})
    target_properties = target_card.setdefault("properties", {})
    copied_run_dirs: set[Path] = set()
    stats = {
        "added": 0,
        "replaced": 0,
        "skipped_existing": 0,
        "skipped_missing_task": 0,
        "copied_run_dirs": 0,
    }

    for task_name, source_prop in source_properties.items():
        if only_tasks and task_name not in only_tasks:
            continue

        source_results = source_prop.get("training_results") or {}
        if not source_results:
            continue

        target_prop = target_properties.get(task_name)
        if target_prop is None:
            print(
                f"[warn] Skipping task '{task_name}' because it does not exist in target card.",
                file=sys.stderr,
            )
            stats["skipped_missing_task"] += len(source_results)
            continue

        target_results = target_prop.setdefault("training_results", {})

        for result_key, source_entry in source_results.items():
            if only_keys and result_key not in only_keys:
                continue

            if result_key in target_results and not overwrite:
                stats["skipped_existing"] += 1
                continue

            merged_entry = sync_entry_artifacts(
                source_entry,
                source_root,
                target_root,
                copy_artifacts=copy_artifacts,
                absolute_paths=absolute_paths,
                dry_run=dry_run,
                copied_run_dirs=copied_run_dirs,
            )

            if result_key in target_results:
                stats["replaced"] += 1
            else:
                stats["added"] += 1
            target_results[result_key] = merged_entry

    stats["copied_run_dirs"] = len(copied_run_dirs)

    if not dry_run and (stats["added"] or stats["replaced"]):
        write_json(target_card, target_card_path)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge training_results from one benchmark card into another and "
            "optionally copy the referenced run outputs."
        )
    )
    parser.add_argument("--source-card", required=True, help="Source benchmark_card.json")
    parser.add_argument("--target-card", required=True, help="Destination benchmark_card.json")
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Root directory used to resolve source artifact paths. "
            "Defaults to the directory that contains --source-card."
        ),
    )
    parser.add_argument(
        "--target-root",
        default=None,
        help=(
            "Root directory used for copied artifacts and rewritten paths. "
            "Defaults to the directory that contains --target-card."
        ),
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help=(
            "Copy each referenced run directory into target-root and rewrite "
            "artifact paths in the merged entries."
        ),
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store rewritten artifact paths as absolute paths instead of relative ones.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target entries when the same training_results key already exists.",
    )
    parser.add_argument(
        "--only-task",
        action="append",
        default=[],
        help="Restrict syncing to one task name. Repeatable.",
    )
    parser.add_argument(
        "--only-key",
        action="append",
        default=[],
        help="Restrict syncing to one training_results key. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the target card or copying files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stats = sync_training_results(
        Path(args.source_card),
        Path(args.target_card),
        source_root=Path(args.source_root) if args.source_root else None,
        target_root=Path(args.target_root) if args.target_root else None,
        copy_artifacts=args.copy_artifacts,
        overwrite=args.overwrite,
        only_tasks=set(args.only_task),
        only_keys=set(args.only_key),
        absolute_paths=args.absolute_paths,
        dry_run=args.dry_run,
    )

    print(
        "Sync summary: "
        f"added={stats['added']} "
        f"replaced={stats['replaced']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"skipped_missing_task={stats['skipped_missing_task']} "
        f"copied_run_dirs={stats['copied_run_dirs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
