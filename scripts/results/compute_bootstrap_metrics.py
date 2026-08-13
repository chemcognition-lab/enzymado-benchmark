#!/usr/bin/env python
# ruff: noqa: E402
"""Bootstrap test metrics directly from prediction artifacts.

General tasks recompute every registered metric from raw predictions. EC-number
and residue-level tasks bootstrap only threshold metrics by protein and retain
the exact point AUROC/AUPRC already stored in metrics.json.

Example:
    python scripts/results/compute_bootstrap_metrics.py --filter-task ec_number
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


sys.path.insert(0, str(_repo_root()))

from src.evaluation import (
    binary_statistics,
    discover,
    ec_evaluation,
    loaders,
    metrics_registry,
    task_types,
)
from src.evaluation import (
    bootstrap as bootstrap_lib,
)


def _task_seed(seed_root: int, signature: tuple[str, ...]) -> int:
    """Derive a deterministic int32 seed from a task signature."""
    generator = np.random.default_rng(seed_root)
    payload = json.dumps(signature, separators=(",", ":")).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    salt = int.from_bytes(digest, "little") % (2**31 - 1)
    return int(generator.integers(0, 2**31 - 1)) ^ salt


def _label_kind(experiment: discover.Experiment) -> str | None:
    """Read the structured label kind recorded during training."""
    task_config = loaders.read_hparams(experiment.run_dir).get("task_config")
    if not isinstance(task_config, dict):
        return None
    label_kind = task_config.get("label_kind")
    return label_kind if isinstance(label_kind, str) else None


def _evaluation(
    experiment: discover.Experiment,
    indices: np.ndarray,
    task_type: str,
) -> tuple[dict[str, object], dict[str, np.ndarray], int] | None:
    """Calculate point and bootstrap metrics from raw prediction scores."""
    if experiment.task == "ec_number":
        metrics = loaders.read_metrics(experiment.run_dir)
        point_metrics = metrics.get("test")
        if not isinstance(point_metrics, dict):
            raise ValueError(
                f"Level-aware EC metrics must be computed before bootstrap: "
                f"{experiment.run_dir}."
            )
        raw_thresholds = point_metrics.get("thresholds")
        if not isinstance(raw_thresholds, dict):
            raise ValueError(
                f"Missing validation EC thresholds in {experiment.run_dir}."
            )
        thresholds: dict[str, float] = {}
        for level_name in ec_evaluation.LEVEL_NAMES:
            value = raw_thresholds.get(level_name)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Missing threshold for {level_name} in {experiment.run_dir}."
                )
            thresholds[level_name] = float(value)
        replicates = ec_evaluation.bootstrap_split(
            experiment.run_dir,
            indices,
            thresholds,
        )
        return point_metrics, replicates, int(indices.shape[1])

    if _label_kind(experiment) == "residue_binary_vector":
        metrics = loaders.read_metrics(experiment.run_dir)
        point_metrics = metrics.get("test")
        if not isinstance(point_metrics, dict):
            raise ValueError(
                f"Point metrics must be computed before residue bootstrap: "
                f"{experiment.run_dir}."
            )
        blocks = loaders.load_split_blocks(experiment.run_dir, "test")
        if blocks is None:
            return None
        counts = binary_statistics.count_blocks(blocks, threshold=0.5)
        replicates = binary_statistics.bootstrap(counts, indices)
        return point_metrics, replicates, int(counts.shape[0])

    payload = loaders.load_test_predictions(experiment.run_dir)
    if payload is None:
        return None
    point_metrics = metrics_registry.evaluate_predictions(
        payload.y_true,
        payload.predictions,
        task_type,
        payload.target_mask,
        split="test",
    )
    metric_set = metrics_registry.metric_set_for(task_type)
    replicates = bootstrap_lib.bootstrap_replicates(
        payload,
        indices,
        metric_set,
        task_type,
    )
    return point_metrics, replicates, bootstrap_lib.n_resample_units(payload)


def _n_units(
    experiment: discover.Experiment,
) -> int | None:
    """Load only enough evaluation data to determine the resampling size."""
    identity = loaders.read_sample_identity(experiment.run_dir, "test")
    if identity is not None:
        return identity[0]
    if experiment.task == "ec_number":
        return ec_evaluation.n_resample_units(experiment.run_dir)
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
        "--benchmark-card",
        type=pathlib.Path,
        default=repo / "benchmark_card.json",
    )
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--task-scope",
        choices=sorted(discover.TASK_SCOPES),
        default=None,
        help="Only bootstrap experiments from this task scope.",
    )
    parser.add_argument(
        "--filter-task",
        action="append",
        default=None,
        help="Only bootstrap this task. May be repeated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute an existing bootstrap test.npz.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def _select_experiments(
    experiments: list[discover.Experiment],
    selected_tasks: list[str] | None,
    task_scope: str | None = None,
) -> list[discover.Experiment]:
    """Apply an optional task filter and report its effect."""
    filtered = experiments
    if task_scope is not None:
        filtered = [item for item in filtered if item.task_scope == task_scope]
        print(
            f"Filtering to task scope: {task_scope} "
            f"({len(filtered)}/{len(experiments)})"
        )
    if selected_tasks:
        selected = set(selected_tasks)
        before = len(filtered)
        filtered = [item for item in filtered if item.task in selected]
        names = ", ".join(sorted(selected))
        print(f"Filtering to task(s): {names} ({len(filtered)}/{before})")
    return filtered


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
    """Compute bootstrap distributions and confidence intervals."""
    args = _parse_args()
    experiments = discover.discover_experiments(args.predictions_root)
    experiments = _select_experiments(
        experiments,
        args.filter_task,
        args.task_scope,
    )
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
    if not experiments:
        return

    grouped: dict[tuple[str, ...], list[discover.Experiment]] = collections.defaultdict(
        list
    )
    for experiment in experiments:
        grouped[discover.task_signature(experiment)].append(experiment)

    args.bootstrap_root.mkdir(parents=True, exist_ok=True)
    for task_index, (signature, task_experiments) in enumerate(
        sorted(grouped.items()),
        start=1,
    ):
        seed = _task_seed(args.seed, signature)
        reference_identity: tuple[int, str] | None = None
        shared_indices: np.ndarray | None = None
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
                continue

            n_units = _n_units(experiment)
            if n_units is None or not n_units:
                print(
                    f"    {experiment.embedding_model}: no test predictions; skipping"
                )
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
                    raise RuntimeError(
                        "Paired bootstrap indices were not initialized."
                    )
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
            evaluation = _evaluation(experiment, indices, task_type)
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
                "ranking_metrics_bootstrapped": (
                    experiment.task != "ec_number"
                    and _label_kind(experiment) != "residue_binary_vector"
                ),
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

    print("\nBootstrap complete.")


if __name__ == "__main__":
    main()
