#!/usr/bin/env python
"""Bootstrap test-set metrics for every discovered experiment.

For each experiment we:
  - generate a (n_replicates, n_units) resample-index matrix shared by every
    embedding model on the same task signature (so matchups are paired);
  - evaluate every metric in the appropriate task-type suite per replicate;
  - write the raw replicate arrays under results/bootstrap/<run_path>/test.npz;
  - augment the experiment's metrics.json with a `test_bootstrap` block holding
    mean / std / CI per metric.

Usage:
    python scripts/compute_bootstrap_metrics.py
    python scripts/compute_bootstrap_metrics.py --n-replicates 200 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.bootstrap import (
    bootstrap_replicates,
    make_resample_indices,
    n_resample_units,
    summarise_replicates,
)
from src.evaluation.bootstrap_fast import (
    bootstrap_classification_fast,
    is_fast_classification_eligible,
    precompute_binary_micro_indicators,
)
from src.evaluation.discover import discover_experiments, task_signature
from src.evaluation.loaders import (
    load_test_predictions,
    read_metrics,
    write_metrics,
)
from src.evaluation.metrics import PRIMARY_METRIC, metric_set_for


def _resolve_task_type(run_dir: Path) -> str | None:
    md = read_metrics(run_dir)
    for split in ("test", "val", "train"):
        block = md.get(split)
        if isinstance(block, dict) and block.get("task_type") in {"regression", "classification"}:
            return block["task_type"]
    return None


def _per_task_seed(seed_root: int, signature: tuple) -> int:
    """Hash signature deterministically into an int32 seed."""
    rng = np.random.default_rng(seed_root)
    payload = json.dumps(signature, separators=(",", ":"), sort_keys=False).encode()
    salt = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)
    return int(rng.integers(0, 2**31 - 1)) ^ salt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=REPO / "results" / "predictions")
    parser.add_argument("--bootstrap-root", type=Path, default=REPO / "results" / "bootstrap")
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--filter-task",
        action="append",
        default=None,
        help=(
            "Only compute bootstrap metrics for this task name. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute if results/bootstrap/<run>/test.npz already exists.",
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
    if not experiments:
        return

    # Group by task signature so all experiments on the same task share resamples.
    by_task: dict[tuple, list] = defaultdict(list)
    for e in experiments:
        by_task[task_signature(e)].append(e)

    n_tasks_total = len(by_task)
    args.bootstrap_root.mkdir(parents=True, exist_ok=True)

    for task_idx, (signature, exps) in enumerate(sorted(by_task.items()), start=1):
        # Find a reference n_units from the first experiment with predictions.
        ref_n: int | None = None
        ref_payload = None
        for e in exps:
            payload = load_test_predictions(e.run_dir)
            if payload is None:
                continue
            ref_n = n_resample_units(payload)
            ref_payload = payload
            break
        if ref_n is None or ref_n == 0:
            print(f"[{task_idx}/{n_tasks_total}] {signature}: no test predictions; skipping")
            continue

        seed = _per_task_seed(args.seed, signature)
        indices = make_resample_indices(ref_n, args.n_replicates, seed)
        print(
            f"[{task_idx}/{n_tasks_total}] task={signature} "
            f"n_units={ref_n} n_exps={len(exps)} seed={seed}"
        )

        for e in exps:
            run_relative = e.run_dir.relative_to(args.predictions_root)
            bs_dir = args.bootstrap_root / run_relative
            bs_path = bs_dir / "test.npz"
            if args.skip_existing and bs_path.exists():
                print(f"    skip-existing: {run_relative}")
                continue

            task_type = _resolve_task_type(e.run_dir)
            if task_type is None:
                print(f"    {e.embedding_model}: missing task_type in metrics.json; skipping")
                continue

            payload = load_test_predictions(e.run_dir)
            if payload is None:
                print(f"    {e.embedding_model}: missing test predictions; skipping")
                continue
            if n_resample_units(payload) != ref_n:
                print(
                    f"    {e.embedding_model}: n_units mismatch "
                    f"({n_resample_units(payload)} vs {ref_n}); skipping"
                )
                continue

            metric_set = metric_set_for(task_type)
            t0 = time.perf_counter()
            if task_type == "classification" and is_fast_classification_eligible(payload):
                # Pre-aggregate per-row TP/FP/FN/TN once, then each replicate is
                # a single sum + four arithmetic ops. Bit-exact with the generic
                # path for binary classification (incl. multilabel/per-residue).
                precomputed = precompute_binary_micro_indicators(payload)
                replicates = bootstrap_classification_fast(precomputed, indices)
                fast_path = True
            else:
                replicates = bootstrap_replicates(payload, indices, metric_set)
                fast_path = False
            elapsed = time.perf_counter() - t0
            summary = summarise_replicates(replicates, confidence_level=args.confidence_level)

            bs_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(bs_path, **{k: v.astype(np.float64) for k, v in replicates.items()})

            md = read_metrics(e.run_dir)
            md["test_bootstrap"] = {
                "n_replicates": args.n_replicates,
                "confidence_level": args.confidence_level,
                "primary_metric": PRIMARY_METRIC[task_type],
                "task_type": task_type,
                "seed": seed,
                "metrics": summary,
            }
            write_metrics(e.run_dir, md)

            path_tag = "fast" if fast_path else "generic"
            print(
                f"    {e.embedding_model}: {task_type} "
                f"primary={PRIMARY_METRIC[task_type]}={summary[PRIMARY_METRIC[task_type]]['mean']:.4f} "
                f"({summary[PRIMARY_METRIC[task_type]]['ci_lo']:.4f}, "
                f"{summary[PRIMARY_METRIC[task_type]]['ci_hi']:.4f}) "
                f"in {elapsed:.1f}s [{path_tag}]"
            )

    print("\nBootstrap complete.")


if __name__ == "__main__":
    main()
