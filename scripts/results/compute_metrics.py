#!/usr/bin/env python
"""Compute primary metrics directly from saved prediction artifacts.

The benchmark card determines each task type when training metadata is absent.
Classification is reduced to compact per-row confusion counts that are cached
for the bootstrap pass, avoiding dense EC concatenation.

Example:
    python scripts/results/compute_metrics.py --filter-task ec_number
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _repo_root() -> pathlib.Path:
    """Return the repository root without relying on the launch directory."""
    return pathlib.Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root()))

from src.evaluation import (  # noqa: E402
    binary_statistics,
    discover,
    loaders,
    metrics_registry,
    statistics_cache,
    task_types,
)


def _compute_metrics(
    experiment: discover.Experiment,
    task_type: str,
    predictions_root: pathlib.Path,
    statistics_root: pathlib.Path,
) -> dict[str, dict[str, object]]:
    """Compute every available split for one experiment."""
    metrics_by_split: dict[str, dict[str, object]] = {}
    for split_name in ("train", "val", "test"):
        if task_type == "classification":
            counts = statistics_cache.load_or_create_counts(
                statistics_root,
                predictions_root,
                experiment.run_dir,
                split_name,
            )
            if counts is not None:
                metrics_by_split[split_name] = binary_statistics.evaluate(
                    counts,
                    split=split_name,
                )
            continue

        payload = loaders.load_split_predictions(experiment.run_dir, split_name)
        if payload is None:
            continue
        metrics_by_split[split_name] = metrics_registry.evaluate_predictions(
            payload.y_true,
            payload.predictions,
            task_type,
            payload.target_mask,
            split=split_name,
        )
    return metrics_by_split


def _parse_args() -> argparse.Namespace:
    """Parse command-line configuration."""
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=pathlib.Path,
        default=repo / "results" / "predictions",
    )
    parser.add_argument(
        "--statistics-root",
        type=pathlib.Path,
        default=repo / "results" / "evaluation_statistics",
    )
    parser.add_argument(
        "--benchmark-card",
        type=pathlib.Path,
        default=repo / "benchmark_card.json",
    )
    parser.add_argument(
        "--filter-task",
        action="append",
        default=None,
        help="Only compute metrics for this task. May be repeated.",
    )
    parser.add_argument(
        "--filter-pool-type",
        action="append",
        choices=("attention", "no_attention"),
        default=None,
        help="Only compute metrics for this pooling type. May be repeated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute runs that already contain metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Compute and persist metrics for discovered experiments."""
    args = _parse_args()
    experiments = discover.discover_experiments(args.predictions_root)
    if args.filter_pool_type:
        selected_pools = set(args.filter_pool_type)
        experiments = [
            experiment
            for experiment in experiments
            if experiment.pool_type in selected_pools
        ]
    if args.filter_task:
        selected_tasks = set(args.filter_task)
        before = len(experiments)
        experiments = [
            experiment
            for experiment in experiments
            if experiment.task in selected_tasks
        ]
        selected = ", ".join(sorted(selected_tasks))
        print(f"Filtering to task(s): {selected} ({len(experiments)}/{before})")
    print(f"Discovered {len(experiments)} experiments under {args.predictions_root}")

    written = 0
    skipped = 0
    for index, experiment in enumerate(experiments, start=1):
        label = (
            f"[{index}/{len(experiments)}] "
            f"{experiment.embedding_model}/{experiment.task}"
        )
        metrics_path = experiment.run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            skipped += 1
            continue

        task_type = task_types.resolve(
            experiment.task,
            args.benchmark_card,
            loaders.read_hparams(experiment.run_dir),
        )
        if task_type is None:
            print(f"{label}: task type is unavailable; skipping")
            continue

        metrics = _compute_metrics(
            experiment,
            task_type,
            args.predictions_root,
            args.statistics_root,
        )
        if not metrics:
            print(f"{label}: no raw predictions found; skipping")
            continue

        loaders.write_metrics(experiment.run_dir, metrics)
        written += 1
        summary = ", ".join(
            f"{split}.n={block.get('n')}" for split, block in metrics.items()
        )
        print(f"{label} ({task_type}): {summary}")

    print(f"\nDone. {written} metrics.json written, {skipped} skipped.")


if __name__ == "__main__":
    main()
