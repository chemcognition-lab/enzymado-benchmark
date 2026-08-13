#!/usr/bin/env python
# ruff: noqa: E402
"""Rate every complete embedding, model, and pooling baseline together.

For each task scope and split, a participant is the full combination of one
embedding model, downstream model family, and pooling strategy. Every pair of
participants competes on every complete task using paired bootstrap samples.
Regression outcomes combine Kendall tau, MAE, RMSE, and R2 with respective
weights of 40%, 20%, 20%, and 20%.

Usage:
    python scripts/results/compute_ratings.py --task-scope real_case
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation import (
    discover,
    loaders,
    metrics_registry,
    rating_lib,
)

DEFAULT_FOUNDATION_MODELS: tuple[str, ...] = (
    "ankh3_large",
    "ankh3_xl",
    "ankh_base",
    "ankh_large",
    "esm2_3B",
    "esm2_650M",
    "esm3_open",
    "esmc_300m",
    "esmc_600m",
    "profluent_e1_300m",
    "profluent_e1_600m",
    "prost5",
    "prot5_xl_half_uniref50",
    "prot5_xl_uniref50",
)

BASELINE_SPECS: tuple[tuple[str, str], ...] = (
    ("no_attention", "linear"),
    ("no_attention", "mlp"),
    ("no_attention", "xgboost"),
    ("attention", "linear"),
    ("attention", "mlp"),
)

REGRESSION_METRIC_WEIGHTS: dict[str, float] = {
    "kendall_tau": 0.4,
    "mae": 0.2,
    "rmse": 0.2,
    "r2": 0.2,
}


def _baseline_id(exp: discover.Experiment) -> str:
    """Return the stable identifier for one complete benchmark baseline."""
    return f"{exp.embedding_model}::{exp.model_family}::{exp.pool_type}"


def _expected_baselines(embedding_models: set[str]) -> set[str]:
    """Build the full embedding by downstream-model baseline matrix."""
    return {
        f"{embedding_model}::{model_family}::{pool_type}"
        for embedding_model in embedding_models
        for pool_type, model_family in BASELINE_SPECS
    }


def _baseline_metadata(baselines: set[str]) -> dict[str, dict[str, str]]:
    """Describe the components of every baseline identifier."""
    metadata = {}
    for baseline in sorted(baselines):
        embedding_model, model_family, pool_type = baseline.split("::")
        metadata[baseline] = {
            "embedding_model": embedding_model,
            "model_family": model_family,
            "pool_type": pool_type,
        }
    return metadata


def _task_type_of(exp: discover.Experiment) -> str | None:
    metrics = loaders.read_metrics(exp.run_dir)
    test = metrics.get("test", {})
    return test.get("task_type")


def _bootstrap_pairing(exp: discover.Experiment) -> tuple[object, ...] | None:
    """Return metadata proving that bootstrap replicate positions are shared."""
    metrics = loaders.read_metrics(exp.run_dir)
    metadata = metrics.get("test_bootstrap")
    if not isinstance(metadata, dict) or not metadata.get("paired_by_position"):
        return None
    return (
        metadata.get("seed"),
        metadata.get("n_replicates"),
        metadata.get("n_units"),
        metadata.get("sample_identity_sha256"),
    )


def _load_distribution(
    exp: discover.Experiment,
    bootstrap_root: Path,
    predictions_root: Path,
    metric_name: str,
) -> np.ndarray | None:
    bootstrap_path = (
        bootstrap_root / exp.run_dir.relative_to(predictions_root) / "test.npz"
    )
    if not bootstrap_path.exists():
        return None
    with np.load(bootstrap_path) as bootstrap:
        if metric_name not in bootstrap.files:
            return None
        data = bootstrap[metric_name].astype(np.float64)
    return data if np.isfinite(data).sum() >= 2 else None


def _metric_weights(task_type: str) -> dict[str, float]:
    match task_type:
        case "regression":
            return dict(REGRESSION_METRIC_WEIGHTS)
        case "classification":
            primary_metric = metrics_registry.primary_metric_for(task_type)
            return {primary_metric: 1.0}
        case _:
            raise ValueError(f"Unsupported task type '{task_type}'.")


def _paired_outcome(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    sign: int,
    alpha: float,
) -> tuple[float, int]:
    finite = np.isfinite(arr_a) & np.isfinite(arr_b)
    score_a = arr_a[finite] * sign
    score_b = arr_b[finite] * sign
    outcome_a = rating_lib.determine_outcome_raw(
        score_b,
        score_a,
        alpha=alpha,
    )
    return outcome_a, int(finite.sum())


def _build_weighted_matches_for_task(
    arrays_by_metric: dict[str, dict[str, np.ndarray]],
    metric_weights: dict[str, float],
    metric_signs: dict[str, int],
    alpha: float,
) -> tuple[list[tuple[str, str, float]], dict[str, dict]]:
    """Combine metric-specific Wilcoxon outcomes into pairwise task matches."""
    baselines = sorted(next(iter(arrays_by_metric.values())))
    matches: list[tuple[str, str, float]] = []
    details: dict[str, dict] = {}
    for index, baseline_a in enumerate(baselines):
        for baseline_b in baselines[index + 1 :]:
            weighted_outcome = 0.0
            metric_details = {}
            for metric_name, weight in metric_weights.items():
                arrays = arrays_by_metric[metric_name]
                outcome_a, n_paired = _paired_outcome(
                    arrays[baseline_a],
                    arrays[baseline_b],
                    metric_signs[metric_name],
                    alpha,
                )
                weighted_outcome += weight * outcome_a
                metric_details[metric_name] = {
                    "weight": weight,
                    "sign": metric_signs[metric_name],
                    "outcome_a": outcome_a,
                    "mean_a": float(np.nanmean(arrays[baseline_a])),
                    "mean_b": float(np.nanmean(arrays[baseline_b])),
                    "n_paired_replicates": n_paired,
                }
            matches.append((baseline_a, baseline_b, weighted_outcome))
            details[f"{baseline_a}__vs__{baseline_b}"] = {
                "baseline_a": baseline_a,
                "baseline_b": baseline_b,
                "weighted_outcome_a": weighted_outcome,
                "metric_outcomes": metric_details,
            }
    return matches, details


def _points_by_baseline(
    matches: list[tuple[str, str, float]],
    baselines: set[str],
) -> dict[str, float]:
    points = {baseline: 0.0 for baseline in baselines}
    for baseline_a, baseline_b, outcome_a in matches:
        points[baseline_a] += outcome_a
        points[baseline_b] += 1.0 - outcome_a
    return dict(sorted(points.items()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=REPO / "results" / "predictions",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=REPO / "results" / "bootstrap",
    )
    parser.add_argument(
        "--ratings-root",
        type=Path,
        default=REPO / "results" / "ratings",
    )
    parser.add_argument(
        "--task-scope",
        choices=sorted(discover.TASK_SCOPES),
        default="real_case",
    )
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--anchor-rating", type=float, default=1500.0)
    parser.add_argument(
        "--foundation-models",
        nargs="+",
        default=list(DEFAULT_FOUNDATION_MODELS),
        help="Embedding models required in every baseline and task.",
    )
    return parser.parse_args()


def main() -> None:
    """Fit one complete-baseline leaderboard for every selected split."""
    args = _parse_args()
    embedding_models = set(args.foundation_models)
    expected_baselines = _expected_baselines(embedding_models)
    experiments = [
        exp
        for exp in discover.discover_experiments(args.predictions_root)
        if exp.task_scope == args.task_scope
        and (args.split is None or exp.split in args.split)
    ]

    by_split: dict[str, list[discover.Experiment]] = defaultdict(list)
    for exp in experiments:
        by_split[exp.split].append(exp)

    args.ratings_root.mkdir(parents=True, exist_ok=True)
    if not by_split:
        print("No experiments found; nothing to rate.")
        return

    for split, split_experiments in sorted(by_split.items()):
        by_task: dict[str, list[discover.Experiment]] = defaultdict(list)
        for exp in split_experiments:
            by_task[exp.task].append(exp)

        all_matches: list[tuple[str, str, float]] = []
        per_task_report: dict[str, dict] = {}
        skipped_tasks: list[dict] = []

        for task, task_experiments in sorted(by_task.items()):
            experiment_by_baseline = {
                _baseline_id(exp): exp
                for exp in task_experiments
                if exp.embedding_model in embedding_models
            }
            missing = sorted(expected_baselines - experiment_by_baseline.keys())
            if missing:
                skipped_tasks.append(
                    {
                        "task": task,
                        "reason": "missing_baselines",
                        "missing": missing,
                    }
                )
                continue

            pairing = {
                _bootstrap_pairing(experiment_by_baseline[baseline])
                for baseline in expected_baselines
            }
            if len(pairing) != 1 or None in pairing:
                skipped_tasks.append(
                    {
                        "task": task,
                        "reason": "bootstraps_not_paired_across_all_baselines",
                        "pairing_signatures": [
                            list(item) if item is not None else None
                            for item in sorted(pairing, key=str)
                        ],
                    }
                )
                continue

            task_type = next(
                (
                    task_type
                    for exp in task_experiments
                    if (task_type := _task_type_of(exp)) is not None
                ),
                None,
            )
            if task_type is None:
                skipped_tasks.append({"task": task, "reason": "no_task_type"})
                continue

            metric_weights = _metric_weights(task_type)
            signs = metrics_registry.higher_is_better_for(task_type)
            metric_signs = {
                metric_name: signs[metric_name]
                for metric_name in metric_weights
            }
            arrays_by_metric: dict[str, dict[str, np.ndarray]] = {
                metric_name: {} for metric_name in metric_weights
            }
            for baseline in expected_baselines:
                exp = experiment_by_baseline[baseline]
                for metric_name in metric_weights:
                    distribution = _load_distribution(
                        exp,
                        args.bootstrap_root,
                        args.predictions_root,
                        metric_name,
                    )
                    if distribution is not None:
                        arrays_by_metric[metric_name][baseline] = distribution

            missing_by_metric = {
                metric_name: sorted(expected_baselines - arrays)
                for metric_name, arrays in arrays_by_metric.items()
                if len(arrays) != len(expected_baselines)
            }
            if missing_by_metric:
                skipped_tasks.append(
                    {
                        "task": task,
                        "reason": "missing_bootstrap_arrays",
                        "missing_by_metric": missing_by_metric,
                    }
                )
                continue

            matches, details = _build_weighted_matches_for_task(
                arrays_by_metric,
                metric_weights,
                metric_signs,
                args.alpha,
            )
            all_matches.extend(matches)
            per_task_report[task] = {
                "task_type": task_type,
                "bootstrap_pairing_signature": list(next(iter(pairing))),
                "metrics_used": list(metric_weights),
                "metric_weights": metric_weights,
                "metric_signs": metric_signs,
                "n_replicates_by_metric": {
                    metric_name: int(next(iter(arrays.values())).size)
                    for metric_name, arrays in arrays_by_metric.items()
                },
                "n_matches": len(matches),
                "n_metric_matches": len(matches) * len(metric_weights),
                "weighted_points_per_baseline": _points_by_baseline(
                    matches,
                    expected_baselines,
                ),
                "mean_by_metric": {
                    metric_name: {
                        baseline: float(np.nanmean(data))
                        for baseline, data in sorted(arrays.items())
                    }
                    for metric_name, arrays in arrays_by_metric.items()
                },
                "pairwise_outcomes": details,
            }

        stem = f"{args.task_scope}__{split}__combined"
        if not per_task_report:
            print(
                f"{stem}: 0 tasks complete with all "
                f"{len(expected_baselines)} baselines."
            )
            continue

        anchor = sorted(expected_baselines)[0]
        ratings = rating_lib.fit_batch_mle(
            all_matches,
            sorted(expected_baselines),
            anchor_model=anchor,
            anchor_rating=args.anchor_rating,
        )
        ranked = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
        payload = {
            "task_scope": args.task_scope,
            "split": split,
            "tournament": "all_complete_baselines",
            "baseline_definition": "embedding_model + model_family + pool_type",
            "baseline_specs": [
                {"pool_type": pool_type, "model_family": model_family}
                for pool_type, model_family in BASELINE_SPECS
            ],
            "n_expected_foundation_models": len(embedding_models),
            "expected_foundation_models": sorted(embedding_models),
            "n_expected_baselines": len(expected_baselines),
            "expected_baselines": sorted(expected_baselines),
            "baseline_metadata": _baseline_metadata(expected_baselines),
            "n_tasks_used": len(per_task_report),
            "tasks_used": sorted(per_task_report),
            "skipped_tasks": skipped_tasks,
            "n_matches": len(all_matches),
            "n_metric_matches": sum(
                report["n_metric_matches"]
                for report in per_task_report.values()
            ),
            "metric_weights_by_task_type": {
                task_type: _metric_weights(task_type)
                for task_type in sorted(
                    {
                        report["task_type"]
                        for report in per_task_report.values()
                    }
                )
            },
            "n_baselines": len(expected_baselines),
            "anchor_baseline": anchor,
            "anchor_rating": args.anchor_rating,
            "ratings": dict(ranked),
            "per_task_report": per_task_report,
        }
        output_path = args.ratings_root / f"{stem}.json"
        output_path.write_text(json.dumps(payload, indent=4) + "\n")
        print(
            f"{stem}: {len(per_task_report)} tasks, "
            f"{len(expected_baselines)} baselines, {len(all_matches)} matches, "
            f"top: {ranked[0][0]}={ranked[0][1]:.1f}"
        )

    print("\nRatings complete.")


if __name__ == "__main__":
    main()
