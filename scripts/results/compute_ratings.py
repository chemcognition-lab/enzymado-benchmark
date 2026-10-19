#!/usr/bin/env python
"""Compute two global Bradley-Terry leaderboards from bootstrap battles.

Example:
    python scripts/results/compute_ratings.py

Test run:
    python scripts/results/compute_ratings.py --n-battles 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado.evaluation import discover, metrics_registry, rating_lib  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Source:
    """One predictions tree and its mirrored bootstrap tree."""

    predictions_root: Path
    bootstrap_root: Path
    task_scope: str
    pool_types: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Cell:
    """One task, split, and metric matrix used in every battle round."""

    task: str
    split: str
    metric: str
    weight: float
    scores: np.ndarray
    sigma: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class Exclusion:
    """One metric cell withheld from the leaderboard, with the evidence for it."""

    task: str
    split: str
    metric: str
    rule: str
    measured: float
    threshold: float


def _foundation_models() -> tuple[str, ...]:
    """Return the benchmark's required embedding models."""
    return (
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


def _reference_embeddings() -> tuple[str, ...]:
    """Return the representations built without a PLM that compete as references.

    ``noplm`` holds ProtParam descriptors (pooled) and a residue one-hot, so it
    shows how much of each task needs no language model at all.
    """
    return ("noplm",)


# Official task weights of the property leaderboard, ordered by how directly a task
# tests knowledge of enzymes, in five tiers one equal step apart: what reaction an
# enzyme catalyzes (EC number) and which residues catalyze it (active sites); how
# fast it turns over (kcat) and where it binds (binding sites); substrate recognition
# under turnover (KM) and the ionization of its catalytic groups (pH optimum);
# folding stability, which is protein biophysics in general (ddG, dTm); and last
# the tasks that say least about the enzyme itself: BioLiP binding, which the
# ligand descriptors largely decide, and solubility, an expression property. The
# real-case datasets keep equal weights.
_TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "properties": {
        "active_sites": 1.0,
        "ec_number": 1.0,
        "kcat": 0.8,
        "binding_site": 0.8,
        "km": 0.6,
        "ph_opt": 0.6,
        "ddg": 0.4,
        "dtm": 0.4,
        "binding": 0.2,
        "solubility": 0.2,
    },
}


def _task_weights(scope: str, tasks: list[str]) -> dict[str, float]:
    """Return each task's leaderboard weight; a scope without a table weighs tasks equally."""
    table = _TASK_WEIGHTS.get(scope)
    if table is None:
        return {task: 1.0 for task in tasks}
    missing = sorted(set(tasks) - set(table))
    if missing:
        raise ValueError(f"No task weight for {missing} in the {scope} leaderboard.")
    return {task: table[task] for task in tasks}


def _competitor_name(exp: discover.Experiment) -> str:
    """Identify a complete baseline, including architecture and embedding."""
    return f"{exp.embedding_model}/{exp.model_family}/{exp.pool_type}"


def _expected_competitors() -> list[str]:
    """Return the 75 baselines that must compete in both leaderboards."""
    configurations = (
        ("attention", "linear"),
        ("attention", "mlp"),
        ("no_attention", "linear"),
        ("no_attention", "mlp"),
        ("no_attention", "xgboost"),
    )
    return sorted(
        f"{embedding}/{family}/{pool}"
        for embedding in (*_foundation_models(), *_reference_embeddings())
        for pool, family in configurations
    )


def _task_types(card: dict, scope: str) -> dict[str, str]:
    """Read expected tasks and types from the benchmark card."""
    return {task: config["task_type"] for task, config in card[scope].items()}


def _class_prevalences(card: dict, scope: str) -> dict[tuple[str, str], float]:
    """Read the declared positive-class prevalence of every classification split."""
    return {
        (task, split): float(prevalence)
        for task, config in card[scope].items()
        for split, prevalence in config.get("class_prevalence", {}).items()
    }


def _load_source(
    source: Source,
    task_types: dict[str, str],
    n_battles: int,
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> None:
    """Load every required metric array from one source tree."""
    for exp in discover.discover_experiments(source.predictions_root):
        if exp.task_scope != source.task_scope:
            continue
        if exp.pool_type not in source.pool_types:
            continue
        if exp.task not in task_types:
            continue

        competitor = _competitor_name(exp)
        relative_run = exp.run_dir.relative_to(source.predictions_root)
        path = source.bootstrap_root / relative_run / "test.npz"
        directions = metrics_registry.higher_is_better_for(task_types[exp.task])
        with np.load(path) as archive:
            for metric, direction in directions.items():
                values = np.asarray(archive[metric], dtype=np.float64)
                if values.shape != (n_battles,):
                    raise ValueError(
                        f"Expected {n_battles} values in {path}:{metric}; "
                        f"found {values.shape}."
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite values in {path}:{metric}.")
                key = (exp.task, exp.split, metric)
                if competitor in arrays[key]:
                    raise ValueError(f"Duplicate competitor {competitor} in {key}.")
                arrays[key][competitor] = values * direction


def _metric_weights(
    task_type: str,
    kendall_weight: float,
    excluded: frozenset[str] = frozenset(),
    label: str = "",
) -> dict[str, float]:
    """Normalize the surviving metric weights to equal influence per task cell."""
    raw = metrics_registry.rating_weights_for(task_type)
    if "kendall_tau" in raw:
        raw["kendall_tau"] = kendall_weight
    kept = {
        name: weight
        for name, weight in raw.items()
        if weight > 0.0 and name not in excluded
    }
    total = sum(kept.values())
    if total <= 0.0:
        raise ValueError(f"Every rated metric of {label or task_type} was excluded.")
    return {name: weight / total for name, weight in kept.items()}


def _prevalence_exclusions(
    rule: dict,
    task_types: dict[str, str],
    split_weights: dict[str, float],
    prevalences: dict[tuple[str, str], float],
) -> list[Exclusion]:
    """Withhold metrics whose scale is dominated by the majority class."""
    threshold = float(rule["threshold"])
    found: list[Exclusion] = []
    for task, task_type in sorted(task_types.items()):
        if task_type != "classification":
            continue
        for split in split_weights:
            if (task, split) not in prevalences:
                raise ValueError(
                    f"The benchmark card declares no class_prevalence for "
                    f"{task}/{split}, so min_positive_prevalence cannot be applied."
                )
            prevalence = prevalences[task, split]
            if prevalence >= threshold:
                continue
            found.extend(
                Exclusion(
                    task=task,
                    split=split,
                    metric=metric,
                    rule="min_positive_prevalence",
                    measured=prevalence,
                    threshold=threshold,
                )
                for metric in rule["metrics"]
            )
    return found


def _ceiling_exclusions(
    rule: dict,
    task_types: dict[str, str],
    split_weights: dict[str, float],
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> list[Exclusion]:
    """Withhold metrics whose best competitor never clears the no-skill mark."""
    found: list[Exclusion] = []
    for task, task_type in sorted(task_types.items()):
        directions = metrics_registry.higher_is_better_for(task_type)
        for split in split_weights:
            for metric, threshold in rule["thresholds"].items():
                if metric not in directions:
                    continue
                if directions[metric] != 1:
                    raise ValueError(
                        f"A degenerate_ceiling threshold needs a higher-is-better "
                        f"metric; '{metric}' is not one."
                    )
                loaded = arrays.get((task, split, metric))
                if not loaded:
                    continue
                ceiling = max(float(values.mean()) for values in loaded.values())
                if ceiling < float(threshold):
                    found.append(
                        Exclusion(
                            task=task,
                            split=split,
                            metric=metric,
                            rule="degenerate_ceiling",
                            measured=ceiling,
                            threshold=float(threshold),
                        )
                    )
    return found


def _exclusions(
    rules: dict,
    task_types: dict[str, str],
    split_weights: dict[str, float],
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]],
    prevalences: dict[tuple[str, str], float],
) -> list[Exclusion]:
    """Apply every rating-eligibility rule declared in the benchmark card."""
    found: list[Exclusion] = []
    if "min_positive_prevalence" in rules:
        found += _prevalence_exclusions(
            rules["min_positive_prevalence"],
            task_types,
            split_weights,
            prevalences,
        )
    if "degenerate_ceiling" in rules:
        found += _ceiling_exclusions(
            rules["degenerate_ceiling"],
            task_types,
            split_weights,
            arrays,
        )
    return found


def _load_tie_margins(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    """Read the training-noise SD of every task, split, metric, head and pooling."""
    margins: dict[tuple[str, str, str, str, str], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["task"],
                row["split"],
                row["metric"],
                row["model_family"],
                row["pool_type"],
            )
            sigma = float(row["sigma"])
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError(f"Invalid sigma {sigma} for {key} in {path}.")
            if key in margins:
                raise ValueError(f"Duplicate tie margin {key} in {path}.")
            margins[key] = sigma
    return margins


def _cell_sigma(
    task: str,
    split: str,
    metric: str,
    competitors: list[str],
    margins: dict[tuple[str, str, str, str, str], float],
) -> np.ndarray:
    """Look up the training-noise SD of every competitor in one metric cell."""
    sigma = np.empty(len(competitors), dtype=np.float64)
    missing = set()
    for index, name in enumerate(competitors):
        _, family, pool = name.split("/")
        key = (task, split, metric, family, pool)
        if key not in margins:
            missing.add(key)
            continue
        sigma[index] = margins[key]
    if missing:
        raise ValueError(f"No tie margin for {sorted(missing)}.")
    return sigma


def _build_cells(
    sources: tuple[Source, ...],
    task_types: dict[str, str],
    split_weights: dict[str, float],
    n_battles: int,
    kendall_weight: float,
    rules: dict,
    prevalences: dict[tuple[str, str], float],
    margins: dict[tuple[str, str, str, str, str], float] | None = None,
    task_weights: dict[str, float] | None = None,
) -> tuple[list[str], list[Cell], list[Exclusion]]:
    """Load and validate a complete leaderboard design."""
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for source in sources:
        _load_source(source, task_types, n_battles, arrays)

    exclusions = _exclusions(rules, task_types, split_weights, arrays, prevalences)
    dropped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for exclusion in exclusions:
        dropped[exclusion.task, exclusion.split].add(exclusion.metric)

    competitors = _expected_competitors()
    expected = set(competitors)
    cells: list[Cell] = []
    for task, task_type in sorted(task_types.items()):
        for split, split_weight in split_weights.items():
            metric_weights = _metric_weights(
                task_type,
                kendall_weight,
                frozenset(dropped[task, split]),
                label=f"{task}/{split}",
            )
            for metric, metric_weight in metric_weights.items():
                key = (task, split, metric)
                present = set(arrays[key])
                if present != expected:
                    missing = sorted(expected - present)
                    extra = sorted(present - expected)
                    raise ValueError(
                        f"Incomplete rating cell {key}: missing={missing}, "
                        f"extra={extra}."
                    )
                scores = np.stack([arrays[key][name] for name in competitors])
                sigma = (
                    None
                    if margins is None
                    else _cell_sigma(task, split, metric, competitors, margins)
                )
                cells.append(
                    Cell(
                        task=task,
                        split=split,
                        metric=metric,
                        weight=(
                            (1.0 if task_weights is None else task_weights[task])
                            * split_weight
                            * metric_weight
                        ),
                        scores=scores,
                        sigma=sigma,
                    )
                )
    return competitors, cells, exclusions


def _pair_indices(n_models: int) -> tuple[np.ndarray, np.ndarray]:
    """Return every unordered competitor pair once."""
    pair_a, pair_b = np.triu_indices(n_models, k=1)
    return pair_a.astype(np.int64), pair_b.astype(np.int64)


def _battle_outcomes(
    cells: list[Cell],
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    n_battles: int,
    tie_z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate all weighted metric battles within each bootstrap round.

    A cell that carries training-noise SDs scores a pair as a tie when the gap
    is within tie_z * sqrt(sigma_a**2 + sigma_b**2): a retrained pair could
    swap order by that much. With zero SDs this is the strict comparison.
    """
    weighted_wins = np.zeros((n_battles, pair_a.size), dtype=np.float64)
    pair_weight = np.zeros(pair_a.size, dtype=np.float64)
    for cell in cells:
        scores_a = cell.scores[pair_a]
        scores_b = cell.scores[pair_b]
        if cell.sigma is None:
            outcomes = (scores_a > scores_b).astype(np.float64)
            outcomes += 0.5 * (scores_a == scores_b)
        else:
            margin = tie_z * np.hypot(cell.sigma[pair_a], cell.sigma[pair_b])
            gap = scores_a - scores_b
            outcomes = (gap > margin[:, None]).astype(np.float64)
            outcomes += 0.5 * (np.abs(gap) <= margin[:, None])
        weighted_wins += cell.weight * outcomes.T
        pair_weight += cell.weight
    return weighted_wins / pair_weight, pair_weight


def _fit_rounds(
    outcomes: np.ndarray,
    pair_weight: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    competitors: list[str],
    center_rating: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one rating vector per bootstrap round plus the pooled vector."""
    samples = np.empty((outcomes.shape[0], len(competitors)), dtype=np.float64)
    for index, round_outcomes in enumerate(outcomes):
        ratings = rating_lib.fit_weighted_mle(
            pair_a,
            pair_b,
            round_outcomes,
            pair_weight,
            competitors,
            center_rating=center_rating,
        )
        samples[index] = [ratings[name] for name in competitors]
        if (index + 1) % 100 == 0:
            print(f"  fitted {index + 1}/{outcomes.shape[0]} battle rounds")

    pooled = rating_lib.fit_weighted_mle(
        pair_a,
        pair_b,
        outcomes.mean(axis=0),
        pair_weight,
        competitors,
        center_rating=center_rating,
    )
    point = np.array([pooled[name] for name in competitors], dtype=np.float64)
    return point, samples


def _ranking_rows(
    competitors: list[str],
    point: np.ndarray,
    samples: np.ndarray,
) -> list[dict[str, object]]:
    """Build sorted, traceable leaderboard rows."""
    standard_errors = samples.std(axis=0, ddof=1)
    ci_lo, ci_hi = np.percentile(samples, (2.5, 97.5), axis=0)
    order = np.argsort(-point)
    rows: list[dict[str, object]] = []
    for rank, index in enumerate(order, start=1):
        embedding, family, pool = competitors[index].split("/")
        rows.append(
            {
                "rank": rank,
                "competitor": competitors[index],
                "embedding_model": embedding,
                "model_family": family,
                "pool_type": pool,
                "rating": float(point[index]),
                "bootstrap_se": float(standard_errors[index]),
                "ci_lo": float(ci_lo[index]),
                "ci_hi": float(ci_hi[index]),
            }
        )
    return rows


def _source_payload(sources: tuple[Source, ...]) -> list[dict[str, object]]:
    """Serialize rating input roots for provenance."""
    return [
        {
            "predictions_root": str(source.predictions_root),
            "bootstrap_root": str(source.bootstrap_root),
            "task_scope": source.task_scope,
            "pool_types": list(source.pool_types),
        }
        for source in sources
    ]


def _write_outputs(
    name: str,
    output_root: Path,
    payload: dict[str, object],
    rows: list[dict[str, object]],
    competitors: list[str],
    point: np.ndarray,
    samples: np.ndarray,
) -> None:
    """Write one JSON/CSV leaderboard and its bootstrap rating matrix."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{name}.json").write_text(json.dumps(payload, indent=4) + "\n")
    with (output_root / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_root / f"{name}_rating_bootstrap.npz",
        competitors=np.asarray(competitors),
        point_ratings=point,
        bootstrap_ratings=samples,
    )


def _compute_leaderboard(
    name: str,
    sources: tuple[Source, ...],
    task_types: dict[str, str],
    split_weights: dict[str, float],
    rules: dict,
    prevalences: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> None:
    """Compute, summarize, and persist one global leaderboard."""
    print(f"Building {name} leaderboard")
    started = time.perf_counter()
    margins = (
        None if args.tie_margins is None else _load_tie_margins(args.tie_margins)
    )
    task_weights = _task_weights(name, sorted(task_types))
    competitors, cells, exclusions = _build_cells(
        sources,
        task_types,
        split_weights,
        args.n_battles,
        args.kendall_weight,
        rules,
        prevalences,
        margins,
        task_weights,
    )
    for exclusion in exclusions:
        print(
            f"  excluded {exclusion.task}/{exclusion.split}/{exclusion.metric}"
            f" ({exclusion.rule}: {exclusion.measured:.6g}"
            f" vs {exclusion.threshold:g})"
        )
    pair_a, pair_b = _pair_indices(len(competitors))
    outcomes, pair_weight = _battle_outcomes(
        cells,
        pair_a,
        pair_b,
        args.n_battles,
        args.tie_z,
    )
    point, samples = _fit_rounds(
        outcomes,
        pair_weight,
        pair_a,
        pair_b,
        competitors,
        args.center_rating,
    )
    rows = _ranking_rows(competitors, point, samples)
    raw_per_round = len(cells) * pair_a.size
    payload = {
        "leaderboard": name,
        "method": "weighted_bradley_terry_bootstrap_battles",
        "n_competitors": len(competitors),
        "n_tasks": len(task_types),
        "tasks": sorted(task_types),
        "n_metric_cells": len(cells),
        "n_bootstrap_battle_rounds": args.n_battles,
        "raw_pairwise_battles_per_round": raw_per_round,
        "raw_pairwise_battles_total": raw_per_round * args.n_battles,
        "effective_pair_weight_per_round": float(pair_weight[0]),
        "task_weights": task_weights,
        "split_weights": split_weights,
        "metric_weights": {
            "kendall_tau": args.kendall_weight,
            "other_metrics": 1.0,
            "normalization": (
                "within each task and split, over the metrics that survive "
                "rating_exclusions"
            ),
        },
        "rating_exclusions": {
            "rules": rules,
            "excluded_cells": [
                dataclasses.asdict(exclusion) for exclusion in exclusions
            ],
        },
        "tie_margins": (
            None
            if args.tie_margins is None
            else {
                "path": str(args.tie_margins),
                "sha256": hashlib.sha256(args.tie_margins.read_bytes()).hexdigest(),
                "z": args.tie_z,
                "rule": "tie when |gap| <= z * sqrt(sigma_a^2 + sigma_b^2)",
            }
        ),
        "rating_center": args.center_rating,
        "bootstrap_se_definition": "SD of ratings across bootstrap battle rounds",
        "sources": _source_payload(sources),
        "ranking": rows,
    }
    _write_outputs(
        name,
        args.ratings_root,
        payload,
        rows,
        competitors,
        point,
        samples,
    )
    elapsed = time.perf_counter() - started
    print(
        f"{name}: {len(rows)} competitors, {len(cells)} metric cells, "
        f"top={rows[0]['competitor']} ({rows[0]['rating']:.1f}), "
        f"elapsed={elapsed:.1f}s"
    )


def _parse_args() -> argparse.Namespace:
    """Parse the two-leaderboard configuration."""
    repo = _REPO
    predictions = repo / "results" / "predictions"
    bootstrap = repo / "results" / "bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-card", type=Path, default=repo / "benchmark_card.json"
    )
    parser.add_argument(
        "--properties-predictions-root",
        type=Path,
        default=predictions,
    )
    parser.add_argument("--properties-bootstrap-root", type=Path, default=bootstrap)
    parser.add_argument(
        "--real-case-no-attention-predictions-root",
        type=Path,
        default=predictions,
    )
    parser.add_argument(
        "--real-case-no-attention-bootstrap-root",
        type=Path,
        default=bootstrap,
    )
    parser.add_argument(
        "--real-case-attention-predictions-root",
        type=Path,
        default=predictions,
    )
    parser.add_argument(
        "--real-case-attention-bootstrap-root",
        type=Path,
        default=bootstrap,
    )
    parser.add_argument(
        "--ratings-root",
        type=Path,
        default=repo / "results" / "ratings" / "seed_42_two_leaderboards",
    )
    parser.add_argument("--n-battles", type=int, default=1000)
    parser.add_argument(
        "--kendall-weight",
        type=float,
        default=metrics_registry.rating_weights_for("regression")["kendall_tau"],
    )
    parser.add_argument("--random-weight", type=float, default=0.4)
    parser.add_argument("--similarity-weight", type=float, default=0.6)
    parser.add_argument("--center-rating", type=float, default=1500.0)
    parser.add_argument(
        "--tie-margins",
        type=Path,
        default=None,
        help=(
            "CSV of training-noise SDs (task, split, metric, model_family, "
            "pool_type, sigma). Battles within the margin count as ties."
        ),
    )
    parser.add_argument("--tie-z", type=float, default=1.96)
    return parser.parse_args()


def main() -> None:
    """Compute exactly one Properties and one Real_case leaderboard."""
    args = _parse_args()
    properties_sources = (
        Source(
            predictions_root=args.properties_predictions_root,
            bootstrap_root=args.properties_bootstrap_root,
            task_scope="properties",
            pool_types=("attention", "no_attention"),
        ),
    )
    real_case_sources = (
        Source(
            predictions_root=args.real_case_no_attention_predictions_root,
            bootstrap_root=args.real_case_no_attention_bootstrap_root,
            task_scope="real_case",
            pool_types=("no_attention",),
        ),
        Source(
            predictions_root=args.real_case_attention_predictions_root,
            bootstrap_root=args.real_case_attention_bootstrap_root,
            task_scope="real_case",
            pool_types=("attention",),
        ),
    )
    card = json.loads(args.benchmark_card.read_text())
    rules = card.get("rating_exclusions", {})
    _compute_leaderboard(
        "properties",
        properties_sources,
        _task_types(card, "properties"),
        {"random": args.random_weight, "similarity": args.similarity_weight},
        rules,
        _class_prevalences(card, "properties"),
        args,
    )
    _compute_leaderboard(
        "real_case",
        real_case_sources,
        _task_types(card, "real_case"),
        {"random": 1.0},
        rules,
        _class_prevalences(card, "real_case"),
        args,
    )
    print("Two leaderboards complete.")


if __name__ == "__main__":
    main()
