#!/usr/bin/env python
# ruff: noqa: E402
"""Compute metrics directly from saved prediction artifacts.

The benchmark card determines each task type when training metadata is absent.
Classification ranking metrics are calculated from the saved continuous
positive-class scores.

EC-number is evaluated separately at each hierarchy level. Threshold metrics
use validation-calibrated operating points; AUROC/AUPRC remain threshold-free.

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

from src.evaluation import (
    discover,
    ec_evaluation,
    loaders,
    metrics_registry,
    task_types,
)


def _compute_metrics(
    experiment: discover.Experiment,
    task_type: str,
    *,
    ec_threshold_grid_size: int,
) -> dict[str, dict[str, object]]:
    """Compute every available split for one experiment."""
    if experiment.task == "ec_number":
        calibrations = ec_evaluation.calibrate_thresholds(
            experiment.run_dir,
            grid_size=ec_threshold_grid_size,
        )
        return {
            split_name: ec_evaluation.evaluate_split(
                experiment.run_dir,
                split_name,
                calibrations,
            )
            for split_name in ("val", "test")
        }

    metrics_by_split: dict[str, dict[str, object]] = {}
    for split_name in ("train", "val", "test"):
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
        "--skip-existing",
        action="store_true",
        help="Do not recompute runs that already contain metrics.json.",
    )
    parser.add_argument(
        "--ec-threshold-grid-size",
        type=int,
        default=ec_evaluation.DEFAULT_THRESHOLD_GRID_SIZE,
        help="Number of validation thresholds considered for each EC level.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def _select_shard(
    experiments: list[discover.Experiment],
    shard_index: int,
    shard_count: int,
) -> list[discover.Experiment]:
    """Select a deterministic experiment shard for Slurm arrays."""
    if shard_count <= 0:
        raise ValueError(f"shard-count must be positive, got {shard_count}.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard-index must be in [0, {shard_count}), got {shard_index}."
        )
    return [
        experiment
        for index, experiment in enumerate(experiments)
        if index % shard_count == shard_index
    ]


def main() -> None:
    """Compute and persist metrics for discovered experiments."""
    args = _parse_args()
    experiments = discover.discover_experiments(args.predictions_root)
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
    unsharded_count = len(experiments)
    experiments = _select_shard(
        experiments,
        args.shard_index,
        args.shard_count,
    )
    print(
        f"Shard {args.shard_index + 1}/{args.shard_count}: "
        f"{len(experiments)}/{unsharded_count} experiments"
    )
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
            ec_threshold_grid_size=args.ec_threshold_grid_size,
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
