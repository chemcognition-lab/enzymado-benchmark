#!/usr/bin/env python
"""Compute primary metrics.json for every discovered experiment.

Training only writes raw predictions (safetensors) plus hparams.json — it no
longer computes metrics inline. This script is the first evaluation pass: for
each experiment it reads the raw train/val/test predictions and hparams.json,
infers task_type, and writes metrics.json.

scripts/results/compute_bootstrap_metrics.py runs after this script and only
augments the metrics.json written here with a `test_bootstrap` confidence-
interval block — it does not compute the primary metrics itself.

Usage:
    python scripts/results/compute_metrics.py
    python scripts/results/compute_metrics.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.artifacts import write_metrics_file
from src.evaluation.discover import discover_experiments
from src.evaluation.loaders import derive_task_type, load_split_predictions, read_hparams
from src.evaluation.metrics_registry import evaluate_predictions


def _compute_experiment_metrics(run_dir: Path, task_type: str) -> dict[str, dict]:
    """Compute metrics for every split with raw predictions present in run_dir."""
    metrics_by_split: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        payload = load_split_predictions(run_dir, split_name)
        if payload is None:
            continue
        metrics_by_split[split_name] = evaluate_predictions(
            payload.y_true,
            payload.predictions,
            task_type,
            payload.target_mask,
            split=split_name,
        )
    return metrics_by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=REPO / "results" / "predictions")
    parser.add_argument(
        "--filter-task",
        action="append",
        default=None,
        help="Only compute metrics for this task name. May be provided multiple times.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute if metrics.json already exists for a run.",
    )
    args = parser.parse_args()

    experiments = discover_experiments(args.predictions_root)
    if args.filter_task:
        selected_tasks = set(args.filter_task)
        before = len(experiments)
        experiments = [e for e in experiments if e.task in selected_tasks]
        print(
            f"Filtering to task(s): {', '.join(sorted(selected_tasks))} "
            f"({len(experiments)}/{before} experiments)"
        )
    print(f"Discovered {len(experiments)} experiments under {args.predictions_root}")

    n_written = 0
    n_skipped = 0
    for idx, exp in enumerate(experiments, start=1):
        label = f"[{idx}/{len(experiments)}] {exp.embedding_model}/{exp.task}"
        metrics_path = exp.run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            n_skipped += 1
            continue

        task_type = derive_task_type(read_hparams(exp.run_dir))
        if task_type is None:
            print(f"{label}: missing task_type in hparams.json; skipping")
            continue

        metrics_by_split = _compute_experiment_metrics(exp.run_dir, task_type)
        if not metrics_by_split:
            print(f"{label}: no raw predictions found; skipping")
            continue

        write_metrics_file(exp.run_dir, metrics_by_split)
        n_written += 1
        splits_summary = ", ".join(f"{split}.n={block.get('n')}" for split, block in metrics_by_split.items())
        print(f"{label} ({task_type}): {splits_summary}")

    print(f"\nDone. {n_written} metrics.json written, {n_skipped} skipped (already existed).")


if __name__ == "__main__":
    main()
