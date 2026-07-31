#!/usr/bin/env python
"""Compute Bradley-Terry MLE ratings per leaderboard from saved bootstrap arrays.

For each (track, split, model_family) leaderboard:
  - Take only tasks where every expected foundation model has a bootstrap file
    (default: 14 — same set used in scripts/slurm/run_training_cluster.slurm).
  - For each such task, build all pair (a, b) matchups using the Wilcoxon
    signed-rank test on the paired bootstrap distributions of the primary metric.
  - Solve Bradley-Terry MLE once per leaderboard; the alphabetically-first
    embedding model is fixed at 1500 to gauge the scale.

Outputs `results/ratings/<track>__<split>__<model_family>.json`.

Usage:
    python scripts/results/compute_ratings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.discover import (
    Experiment,
    TrackKey,
    discover_experiments,
    group_into_tracks,
    load_real_case_tasks,
)
from src.evaluation.loaders import read_metrics
from src.evaluation.metrics_registry import higher_is_better_for, primary_metric_for
from src.evaluation.rating_lib import determine_outcome_raw, fit_batch_mle


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


def _task_type_of(exp: Experiment) -> str | None:
    md = read_metrics(exp.run_dir)
    test = md.get("test", {})
    return test.get("task_type")


def _load_primary_distribution(
    exp: Experiment,
    bootstrap_root: Path,
    predictions_root: Path,
    primary_metric: str,
) -> np.ndarray | None:
    metrics = read_metrics(exp.run_dir)
    bootstrap_metadata = metrics.get("test_bootstrap")
    if (
        isinstance(bootstrap_metadata, dict)
        and bootstrap_metadata.get("paired_by_position") is False
    ):
        return None

    bs_path = bootstrap_root / exp.run_dir.relative_to(predictions_root) / "test.npz"
    if not bs_path.exists():
        return None
    arr = np.load(bs_path)
    if primary_metric not in arr.files:
        return None
    data = arr[primary_metric].astype(np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.size < 2:
            return None
        return finite
    return data


def _build_matches_for_task(
    arrays: dict[str, np.ndarray],
    sign: int,
    alpha: float,
) -> tuple[list[tuple[str, str, float]], dict[tuple[str, str], dict]]:
    """Pairwise Wilcoxon outcomes within one task.

    `sign` is +1 for higher-is-better, -1 for lower-is-better.
    Returns the match list (one entry per ordered pair (a, b) with the outcome
    *for a*) and a per-pair detail dict for traceability.
    """
    models = sorted(arrays.keys())
    matches: list[tuple[str, str, float]] = []
    details: dict[tuple[str, str], dict] = {}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            arr_a = arrays[a] * sign
            arr_b = arrays[b] * sign
            outcome_a = determine_outcome_raw(arr_b, arr_a, alpha=alpha)
            matches.append((a, b, outcome_a))
            details[(a, b)] = {
                "outcome_a": outcome_a,
                "mean_a": float(arrays[a].mean()),
                "mean_b": float(arrays[b].mean()),
            }
    return matches, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=REPO / "results" / "predictions")
    parser.add_argument("--bootstrap-root", type=Path, default=REPO / "results" / "bootstrap")
    parser.add_argument("--ratings-root", type=Path, default=REPO / "results" / "ratings")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--anchor-rating", type=float, default=1500.0)
    parser.add_argument(
        "--foundation-models",
        nargs="+",
        default=list(DEFAULT_FOUNDATION_MODELS),
        help="Embedding models that must all be present for a task to contribute matches.",
    )
    args = parser.parse_args()

    expected_models = set(args.foundation_models)
    n_expected = len(expected_models)

    real_case_tasks = load_real_case_tasks()
    experiments = discover_experiments(args.predictions_root)
    tracks = group_into_tracks(experiments, real_case_tasks)
    args.ratings_root.mkdir(parents=True, exist_ok=True)

    if not tracks:
        print("No experiments found; nothing to rate.")
        return

    for track_key, exps in sorted(
        tracks.items(),
        key=lambda kv: (kv[0].track, kv[0].split, kv[0].model_family),
    ):
        # Group within the track by task.
        by_task: dict[str, list[Experiment]] = defaultdict(list)
        for e in exps:
            by_task[e.task].append(e)

        track_matches: list[tuple[str, str, float]] = []
        track_models: set[str] = set()
        per_task_report: dict[str, dict] = {}
        skipped_tasks: list[dict] = []

        for task, task_exps in sorted(by_task.items()):
            embeddings_present = {e.embedding_model for e in task_exps}
            missing = sorted(expected_models - embeddings_present)
            if missing:
                skipped_tasks.append(
                    {
                        "task": task,
                        "reason": "missing_foundation_models",
                        "missing": missing,
                        "present": sorted(embeddings_present),
                    }
                )
                continue

            # Resolve task_type from any experiment in this task.
            task_type: str | None = None
            for e in task_exps:
                tt = _task_type_of(e)
                if tt is not None:
                    task_type = tt
                    break
            if task_type is None:
                skipped_tasks.append({"task": task, "reason": "no_task_type"})
                continue
            primary_metric = primary_metric_for(task_type)
            sign = higher_is_better_for(task_type)[primary_metric]

            arrays: dict[str, np.ndarray] = {}
            for e in task_exps:
                if e.embedding_model not in expected_models:
                    continue
                dist = _load_primary_distribution(
                    e,
                    bootstrap_root=args.bootstrap_root,
                    predictions_root=args.predictions_root,
                    primary_metric=primary_metric,
                )
                if dist is not None:
                    arrays[e.embedding_model] = dist

            if len(arrays) != n_expected:
                skipped_tasks.append(
                    {
                        "task": task,
                        "reason": "missing_bootstrap_arrays",
                        "missing": sorted(expected_models - arrays.keys()),
                    }
                )
                continue

            matches, details = _build_matches_for_task(arrays, sign=sign, alpha=args.alpha)
            track_matches.extend(matches)
            track_models.update(arrays.keys())
            per_task_report[task] = {
                "task_type": task_type,
                "primary_metric": primary_metric,
                "sign": sign,
                "n_replicates": int(next(iter(arrays.values())).size),
                "n_matches": len(matches),
                "wins_per_model": {
                    m: sum(1 for a, b, o in matches if (a == m and o == 1.0) or (b == m and o == 0.0))
                    for m in sorted(arrays.keys())
                },
                "ties_per_model": {
                    m: sum(1 for a, b, o in matches if (a == m or b == m) and o == 0.5)
                    for m in sorted(arrays.keys())
                },
                "mean_primary": {m: float(arr.mean()) for m, arr in arrays.items()},
            }

        if not per_task_report:
            print(
                f"{track_key.filename_stem()}: 0 tasks complete with all "
                f"{n_expected} foundation models — no leaderboard written."
            )
            if skipped_tasks:
                payload = {
                    "track": track_key.track,
                    "split": track_key.split,
                    "model_family": track_key.model_family,
                    "n_expected_foundation_models": n_expected,
                    "skipped_tasks": skipped_tasks,
                    "ratings": {},
                    "n_matches": 0,
                    "n_models": 0,
                }
                out_path = args.ratings_root / f"{track_key.filename_stem()}.json"
                out_path.write_text(json.dumps(payload, indent=4) + "\n")
            continue

        anchor = sorted(track_models)[0]
        ratings = fit_batch_mle(
            track_matches,
            sorted(track_models),
            anchor_model=anchor,
            anchor_rating=args.anchor_rating,
        )
        ranked = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)

        payload = {
            "track": track_key.track,
            "split": track_key.split,
            "model_family": track_key.model_family,
            "n_expected_foundation_models": n_expected,
            "expected_foundation_models": sorted(expected_models),
            "n_tasks_used": len(per_task_report),
            "tasks_used": sorted(per_task_report.keys()),
            "skipped_tasks": skipped_tasks,
            "n_matches": len(track_matches),
            "n_models": len(track_models),
            "anchor_model": anchor,
            "anchor_rating": args.anchor_rating,
            "ratings": dict(ranked),
            "per_task_report": per_task_report,
        }
        out_path = args.ratings_root / f"{track_key.filename_stem()}.json"
        out_path.write_text(json.dumps(payload, indent=4) + "\n")
        print(
            f"{track_key.filename_stem()}: {len(per_task_report)} tasks, "
            f"{len(track_matches)} matches, top: {ranked[0][0]}={ranked[0][1]:.1f}"
        )

    print("\nRatings complete.")


if __name__ == "__main__":
    main()
