#!/usr/bin/env python
"""Bootstrap test metrics directly from prediction artifacts.

Classification uses per-row TP/FP/FN/TN statistics from the cache and builds
a missing entry from the predictions. Regression uses the raw payload. Existing
metrics are never required: the test point estimate is recomputed before
bootstrap metadata is written, so this script alone produces both.

Example:
    python scripts/results/compute_bootstrap_metrics.py --filter-task ec_number --n-replicates 1000
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np


def _repo_root() -> pathlib.Path:
    """Return the repository root without relying on the launch directory."""
    return pathlib.Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root() / "src"))

from enzymado.evaluation import (  # noqa: E402, metrics_registry
    binary_statistics,
    discover,
    loaders,
    metrics_registry,
    statistics_cache,
    task_types,
)
from enzymado.evaluation import bootstrap as bootstrap_lib  # noqa: E402


def _task_seed(seed_root: int, signature: tuple[str, ...]) -> int:
    """Derive a deterministic int32 seed from a task signature."""
    generator = np.random.default_rng(seed_root)
    payload = json.dumps(signature, separators=(",", ":")).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    salt = int.from_bytes(digest, "little") % (2**31 - 1)
    return int(generator.integers(0, 2**31 - 1)) ^ salt


def _classification_evaluation(
    experiment: discover.Experiment,
    indices: binary_statistics.CountArray,
    predictions_root: pathlib.Path,
    statistics_root: pathlib.Path,
) -> tuple[dict[str, object], binary_statistics.MetricArrays, int] | None:
    """Calculate point and bootstrap classification metrics from counts."""
    counts = statistics_cache.load_or_create_counts(
        statistics_root,
        predictions_root,
        experiment.run_dir,
        "test",
    )
    if counts is None:
        return None
    point_metrics = metrics_registry.evaluate(counts, split="test")
    replicates = metrics_registry.bootstrap(counts, indices)
    # Confusion counts cannot express a ranking metric. Where the artifact
    # stores probabilities, auroc/auprc are recomputed from additive score
    # histograms so replicates stay a weighted sum rather than a per-replicate
    # sort; the remaining metrics keep their count fast path.
    blocks = loaders.load_split_blocks(experiment.run_dir, "test")
    histograms = binary_statistics.ranking_histogram_blocks(blocks) if blocks else None
    if histograms is not None:
        point_metrics |= metrics_registry.evaluate_ranking(histograms)
        point_metrics["ranking_source"] = "proba"
        replicates = replicates | metrics_registry.bootstrap_ranking(
            histograms,
            indices,
        )
    else:
        point_metrics["ranking_source"] = "hard_labels"
    return point_metrics, replicates, counts.shape[0]


def _regression_evaluation(
    experiment: discover.Experiment,
    indices: binary_statistics.CountArray,
) -> tuple[dict[str, object], dict[str, np.ndarray], int] | None:
    """Calculate point and bootstrap regression metrics from raw arrays."""
    payload = loaders.load_test_predictions(experiment.run_dir)
    if payload is None:
        return None
    point_metrics = metrics_registry.evaluate_predictions(
        payload.y_true,
        payload.predictions,
        "regression",
        payload.target_mask,
        split="test",
    )
    specs = metrics_registry.metric_specs("regression")
    replicates = bootstrap_lib.bootstrap_replicates(
        payload,
        indices,
        specs,
    )
    return point_metrics, replicates, bootstrap_lib.n_resample_units(payload)


def _n_units(
    experiment: discover.Experiment,
    task_type: str,
    predictions_root: pathlib.Path,
    statistics_root: pathlib.Path,
) -> int | None:
    """Load only enough evaluation data to determine the resampling size."""
    if task_type == "classification":
        counts = statistics_cache.load_or_create_counts(
            statistics_root,
            predictions_root,
            experiment.run_dir,
            "test",
        )
        return None if counts is None else counts.shape[0]

    payload = loaders.load_test_predictions(experiment.run_dir)
    return None if payload is None else bootstrap_lib.n_resample_units(payload)


def _test_sample_identity(
    experiment: discover.Experiment,
) -> tuple[int, str] | None:
    """Read the ordered test-sample fingerprint from training metadata."""
    return loaders.read_sample_identity(experiment.run_dir, "test")


def _independent_seed(
    seed_root: int,
    signature: tuple[str, ...],
    relative_run: pathlib.Path,
) -> int:
    """Derive a deterministic fallback seed for one unpaired experiment."""
    return _task_seed(seed_root, (*signature, relative_run.as_posix()))


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
        "--bootstrap-root",
        type=pathlib.Path,
        default=repo / "results" / "bootstrap",
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
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--filter-task",
        action="append",
        default=None,
        help="Only bootstrap this task. May be repeated.",
    )
    parser.add_argument(
        "--filter-pool-type",
        action="append",
        choices=("attention", "no_attention"),
        default=None,
        help="Only bootstrap this pooling type. May be repeated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute an existing bootstrap test.npz.",
    )
    return parser.parse_args()


def _select_experiments(
    experiments: list[discover.Experiment],
    selected_tasks: list[str] | None,
) -> list[discover.Experiment]:
    """Apply an optional task filter and report its effect."""
    if not selected_tasks:
        return experiments
    selected = set(selected_tasks)
    filtered = [item for item in experiments if item.task in selected]
    names = ", ".join(sorted(selected))
    print(f"Filtering to task(s): {names} ({len(filtered)}/{len(experiments)})")
    return filtered


def main() -> None:
    """Compute bootstrap distributions and confidence intervals."""
    args = _parse_args()
    experiments = discover.discover_experiments(args.predictions_root)
    if args.filter_pool_type:
        selected_pools = set(args.filter_pool_type)
        experiments = [
            experiment
            for experiment in experiments
            if experiment.pool_type in selected_pools
        ]
    experiments = _select_experiments(experiments, args.filter_task)
    print(f"Discovered {len(experiments)} experiments under {args.predictions_root}")
    if not experiments:
        return

    grouped: dict[tuple[str, ...], list[discover.Experiment]] = collections.defaultdict(
        list
    )
    for experiment in experiments:
        grouped[discover.task_signature(experiment)].append(experiment)

    args.bootstrap_root.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    for task_index, (signature, task_experiments) in enumerate(
        sorted(grouped.items()),
        start=1,
    ):
        seed = _task_seed(args.seed, signature)
        reference_identity: tuple[int, str] | None = None
        shared_indices: binary_statistics.CountArray | None = None
        print(
            f"[{task_index}/{len(grouped)}] task={signature} "
            f"n_exps={len(task_experiments)} seed={seed}"
        )

        for experiment in task_experiments:
            relative_run = experiment.run_dir.relative_to(args.predictions_root)
            bootstrap_dir = args.bootstrap_root / relative_run
            bootstrap_path = bootstrap_dir / "test.npz"
            if args.skip_existing and bootstrap_path.exists():
                print(f"    skip-existing: {relative_run}")
                continue

            task_type = task_types.resolve(
                experiment.task,
                args.benchmark_card,
                loaders.read_hparams(experiment.run_dir),
            )
            if task_type is None:
                print(f"    {experiment.embedding_model}: unknown task type; skipping")
                skipped.append(f"{experiment.task}/{experiment.embedding_model}")
                continue

            n_units = _n_units(
                experiment,
                task_type,
                args.predictions_root,
                args.statistics_root,
            )
            if n_units is None or not n_units:
                print(
                    f"    {experiment.embedding_model}: no test predictions; skipping"
                )
                skipped.append(f"{experiment.task}/{experiment.embedding_model}")
                continue
            identity = _test_sample_identity(experiment)
            if identity is not None and identity[0] != n_units:
                raise ValueError(
                    f"Sample identity metadata reports {identity[0]} rows but "
                    f"predictions contain {n_units} for {relative_run}."
                )
            if reference_identity is None and identity is not None:
                reference_identity = identity
                shared_indices = bootstrap_lib.make_resample_indices(
                    n_units,
                    args.n_replicates,
                    seed,
                )

            paired_by_sample_id = (
                identity is not None and identity == reference_identity
            )
            if paired_by_sample_id:
                if shared_indices is None:
                    raise RuntimeError("Paired bootstrap indices were not initialized.")
                indices = shared_indices
                identity_status = "verified"
                resample_seed = seed
            else:
                resample_seed = _independent_seed(
                    args.seed,
                    signature,
                    relative_run,
                )
                indices = bootstrap_lib.make_resample_indices(
                    n_units,
                    args.n_replicates,
                    resample_seed,
                )
                identity_status = (
                    "missing" if identity is None else "different_from_reference"
                )
                print(
                    f"    {experiment.embedding_model}: sample IDs are "
                    f"{identity_status}; using an independent resample"
                )

            started = time.perf_counter()
            if task_type == "classification":
                evaluation = _classification_evaluation(
                    experiment,
                    indices,
                    args.predictions_root,
                    args.statistics_root,
                )
            else:
                evaluation = _regression_evaluation(experiment, indices)
            if evaluation is None:
                print(f"    {experiment.embedding_model}: evaluation failed; skipping")
                continue

            point_metrics, replicates, evaluated_n = evaluation
            if evaluated_n != n_units:
                raise ValueError(
                    f"Evaluation rows changed from {n_units} to {evaluated_n} for "
                    f"{relative_run}."
                )
            summary = bootstrap_lib.summarise_replicates(
                replicates,
                confidence_level=args.confidence_level,
            )
            primary_metric = metrics_registry.primary_metric_for(task_type)
            if primary_metric is None:
                raise ValueError(f"No primary metric registered for {task_type}.")

            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                bootstrap_path,
                **{
                    name: values.astype(np.float64)
                    for name, values in replicates.items()
                },
            )

            metrics = loaders.read_metrics(experiment.run_dir)
            metrics["test"] = point_metrics
            metrics["test_bootstrap"] = {
                "n_replicates": args.n_replicates,
                "confidence_level": args.confidence_level,
                "primary_metric": primary_metric,
                "task_type": task_type,
                "seed": resample_seed,
                "n_units": n_units,
                "sample_identity_status": identity_status,
                "sample_identity_sha256": identity[1] if identity else None,
                "paired_by_sample_id": paired_by_sample_id,
                "paired_by_position": paired_by_sample_id,
                "metrics": summary,
            }
            loaders.write_metrics(experiment.run_dir, metrics)

            elapsed = time.perf_counter() - started
            primary_summary = summary[primary_metric]
            print(
                f"    {experiment.embedding_model}: {task_type} "
                f"{primary_metric}={primary_summary['mean']:.4f} "
                f"({primary_summary['ci_lo']:.4f}, "
                f"{primary_summary['ci_hi']:.4f}) in {elapsed:.1f}s"
            )

    if skipped:
        # An experiment that produced no bootstrap leaves a hole in the
        # results tree. Printing it and exiting 0 makes the gap invisible to
        # the scheduler and to any downstream check.
        raise SystemExit(
            f"{len(skipped)} experiment(s) produced no bootstrap: "
            + ", ".join(skipped)
        )
    print("\nBootstrap complete.")


if __name__ == "__main__":
    main()
