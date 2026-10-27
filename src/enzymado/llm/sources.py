"""Read the benchmark artifacts behind the assistant's cards and comparisons.

Everything here is torch-free and network-free, and reads files tracked in the
repository, so the cards always match the published results:

* per-run test metrics and settings: ``results/predictions/<scope>/<pool>/<split>/
  <head>/<task>/<embedding>/metrics.json`` and ``hparams.json``;
* the two Bradley-Terry leaderboards:
  ``results/ratings/seed_42_two_leaderboards/{properties,real_case}.csv``;
* the training-noise SD of every rated cell: ``results/noise_seeds/tie_margins_v4.csv``;
* task metadata: ``benchmark_card.json``, each task's ``dataset_card.json`` and the
  task registry;
* PLM facts (parameters, width, architecture, objective):
  ``scripts/figures/plm_attributes.csv``.

The primary metric is the paper's: Kendall's tau for regression and AUPRC for
classification, the metrics epochs are selected on (``metrics_registry`` flags MAE
and F1 instead). The task registry is loaded from its source file alone, because
importing ``enzymado.data_loading`` or ``enzymado.evaluation.metrics_registry``
pulls in torch.

Example:
    python -c "from enzymado.llm import sources; print(len(sources.load_records()))"
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
from pathlib import Path
import sys
import typing

import pandas as pd

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
REPO_ROOT = _P.project_root
PREDICTIONS_DIR = _P.results_dir / "predictions"
RATINGS_DIR = _P.results_dir / "ratings" / "seed_42_two_leaderboards"
PLM_ATTRIBUTES = _P.scripts_dir / "figures" / "plm_attributes.csv"
BENCHMARK_CARD = REPO_ROOT / "benchmark_card.json"
TASK_REGISTRY_FILE = _P.src_dir / "enzymado" / "data_loading" / "data_registry.py"

NOPLM = "noplm"
NOPLM_FACTS = {
    "dim": 36,
    "params": None,
    "arch": "no PLM",
    "objective": (
        "36 Biopython ProtParam descriptors per protein, and a 20-letter residue "
        "one-hot for attention pooling and the residue tasks"
    ),
    "extra": "reference baseline",
}
LEADERBOARDS = ("properties", "real_case")
PRIMARY_METRIC = {"regression": "kendall_tau", "classification": "auprc"}
HIGHER_IS_BETTER = {"kendall_tau": 1, "auprc": 1}

# The five baselines every representation is trained with: (pool, head) pairs.
BASELINES: list[dict[str, str]] = [
    {"id": "linear", "pool": "no_attention", "head": "linear",
     "title": "Linear head, mean pooling"},
    {"id": "mlp", "pool": "no_attention", "head": "mlp",
     "title": "MLP head, mean pooling"},
    {"id": "xgboost", "pool": "no_attention", "head": "xgboost",
     "title": "XGBoost head, mean pooling"},
    {"id": "attention_linear", "pool": "attention", "head": "linear",
     "title": "Linear head, attention pooling"},
    {"id": "attention_mlp", "pool": "attention", "head": "mlp",
     "title": "MLP head, attention pooling"},
]  # fmt: skip

SPLITS: list[dict[str, str]] = [
    {"id": "random", "title": "Random split"},
    {"id": "similarity", "title": "Similarity split (MMseqs2 clusters)"},
]


def config_label(head: str, pool: str) -> str:
    """Short name of a baseline, e.g. ``mlp`` or ``mlp + attention``."""
    return head if pool == "no_attention" else f"{head} + attention"


def split_order(split: str) -> int:
    return 0 if split == "random" else 1


@functools.lru_cache(maxsize=1)
def load_task_registry() -> dict[str, typing.Any]:
    """``TASK_REGISTRY`` (name -> TaskConfig), loaded without importing torch.

    The module must be in ``sys.modules`` before it runs so that its dataclasses
    and enums can resolve their own module.
    """
    name = "_enzymado_llm_task_registry"
    spec = importlib.util.spec_from_file_location(name, TASK_REGISTRY_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {TASK_REGISTRY_FILE}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.TASK_REGISTRY


@functools.lru_cache(maxsize=1)
def load_models() -> dict[str, dict[str, typing.Any]]:
    """Facts per representation: the 14 PLMs, then the no-PLM reference.

    ``params`` is in millions (encoder only for the T5 models), ``dim`` the
    embedding width. Cached and shared: treat the result as read-only.
    """
    attr = pd.read_csv(PLM_ATTRIBUTES).copy()
    models = {
        row.emb: {
            "dim": int(row.dim),
            "params": int(row.params),
            "arch": row.arch,
            "objective": row.objective,
            "extra": row.extra,
        }
        for row in attr.itertuples()
    }
    models[NOPLM] = dict(NOPLM_FACTS)
    return models


def format_params(millions: int | None) -> str:
    if millions is None:
        return "n/a"
    return f"{millions / 1000:.2f}B" if millions >= 1000 else f"{millions}M"


def load_benchmark_card() -> dict[str, typing.Any]:
    return json.loads(BENCHMARK_CARD.read_text())


def tie_z() -> float:
    """The z of the noise threshold z * sqrt(sigma_a^2 + sigma_b^2)."""
    return float(load_benchmark_card()["rating"]["tie_margins"]["z"])


def dataset_stats_for(task_cfg: typing.Any) -> dict[str, typing.Any] | None:
    """The ``dataset_card.json`` next to a task's processed CSVs, if present."""
    if not task_cfg.dataset_paths:
        return None
    any_csv = Path(next(iter(task_cfg.dataset_paths.values())))
    stats_path = any_csv.parent / "dataset_card.json"
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def benchmark_entry_for(
    card: dict[str, typing.Any], task: str
) -> dict[str, typing.Any] | None:
    """The ``benchmark_card.json`` entry for a task, in either scope."""
    for scope in LEADERBOARDS:
        if task in card.get(scope, {}):
            return card[scope][task]
    return None


def _is_finite(value: typing.Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


@functools.lru_cache(maxsize=1)
def load_records() -> list[dict[str, typing.Any]]:
    """One record per published run: its identity and its test primary metric.

    Cached and shared across callers, which read but never mutate the records.
    """
    records = []
    for path in sorted(PREDICTIONS_DIR.glob("*/*/*/*/*/*/metrics.json")):
        scope, pool, split, head, task, emb = path.parent.relative_to(
            PREDICTIONS_DIR
        ).parts
        test = json.loads(path.read_text()).get("test") or {}
        task_type = test.get("task_type")
        if not task_type:
            continue
        primary = PRIMARY_METRIC[task_type]
        value = test.get(primary)
        hparams_path = path.with_name("hparams.json")
        hparams = json.loads(hparams_path.read_text()) if hparams_path.exists() else {}
        records.append(
            {
                "scope": scope,
                "pool": pool,
                "split": split,
                "head": head,
                "config": config_label(head, pool),
                "task": task,
                "emb": emb,
                "task_type": task_type,
                "model_name": hparams.get("model_name"),
                "params": hparams.get("params") or {},
                "primary_metric": primary,
                "primary_sign": HIGHER_IS_BETTER[primary],
                "primary_value": value if _is_finite(value) else None,
                "n_test": test.get("n"),
            }
        )
    return records


@functools.lru_cache(maxsize=2)
def load_ratings(leaderboard: str) -> pd.DataFrame:
    """One leaderboard, best first: competitor, rating, bootstrap SE, 95% interval."""
    ratings = pd.read_csv(RATINGS_DIR / f"{leaderboard}.csv").copy()
    return ratings.sort_values("rating", ascending=False).reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def load_noise() -> dict[tuple[str, str, str, str, str], float]:
    """Training-noise SD per (task, split, metric, head, pool)."""
    path = REPO_ROOT / load_benchmark_card()["rating"]["tie_margins"]["path"]
    margins = pd.read_csv(path).copy()
    return {
        (r.task, r.split, r.metric, r.model_family, r.pool_type): float(r.sigma)
        for r in margins.itertuples()
    }


def noise_threshold(a: dict[str, typing.Any], b: dict[str, typing.Any]) -> float | None:
    """Gap below which two runs of one task, split and metric are a tie.

    ``a`` and ``b`` are records; the threshold is z * sqrt(sigma_a^2 + sigma_b^2)
    with each sigma taken for that run's head and pooling, exactly as the
    leaderboards count ties. ``None`` when a sigma is missing.
    """
    noise = load_noise()
    sigmas = [
        noise.get((r["task"], r["split"], r["primary_metric"], r["head"], r["pool"]))
        for r in (a, b)
    ]
    if any(s is None for s in sigmas):
        return None
    return tie_z() * math.sqrt(sum(s**2 for s in sigmas))


def _group(records: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record[k] for k in keys), []).append(record)
    return grouped


def _best(records: list[dict]) -> dict:
    sign = records[0]["primary_sign"]
    return max(records, key=lambda r: sign * r["primary_value"])


def best_per_task(records: list[dict], emb: str) -> dict[tuple[str, str], dict]:
    """``(split, task) -> best record`` of one representation across baselines."""
    mine = [r for r in records if r["emb"] == emb and r["primary_value"] is not None]
    return {key: _best(group) for key, group in _group(mine, "split", "task").items()}


def best_by_task_for_model(records: list[dict], emb: str) -> dict[str, list[dict]]:
    """Per split, a representation's best test score per task and its baseline."""
    out: dict[str, list[dict]] = {}
    for (split, task), best in sorted(best_per_task(records, emb).items()):
        out.setdefault(split, []).append(
            {
                "task": task,
                "metric": best["primary_metric"],
                "value": round(best["primary_value"], 4),
                "config": best["config"],
                "n_test": best["n_test"],
            }
        )
    return out


def baseline_task_table(
    records: list[dict], pool: str, head: str
) -> dict[str, list[dict]]:
    """Per split, one baseline's mean and best test score per task over the PLMs."""
    mine = [
        r
        for r in records
        if r["pool"] == pool
        and r["head"] == head
        and r["emb"] != NOPLM
        and r["primary_value"] is not None
    ]
    out: dict[str, list[dict]] = {}
    for (split, task), group in sorted(_group(mine, "split", "task").items()):
        values = [r["primary_value"] for r in group]
        out.setdefault(split, []).append(
            {
                "task": task,
                "metric": group[0]["primary_metric"],
                "mean": round(sum(values) / len(values), 4),
                "best": round(_best(group)["primary_value"], 4),
                "best_emb": _best(group)["emb"],
                "n_models": len(values),
            }
        )
    return out


def task_leaderboard(records: list[dict], task: str) -> dict[str, list[dict]]:
    """Per split, every representation ranked by its best score on one task.

    ``within_noise`` marks a representation whose gap to the leader is no larger
    than the noise threshold between the two winning baselines.
    """
    mine = [r for r in records if r["task"] == task and r["primary_value"] is not None]
    out: dict[str, list[dict]] = {}
    for (split,), group in sorted(
        _group(mine, "split").items(), key=lambda kv: split_order(kv[0][0])
    ):
        sign = group[0]["primary_sign"]
        best = [_best(g) for g in _group(group, "emb").values()]
        ranked = sorted(best, key=lambda r: sign * r["primary_value"], reverse=True)
        leader = ranked[0]
        rows = []
        for i, record in enumerate(ranked, start=1):
            gap = sign * (leader["primary_value"] - record["primary_value"])
            threshold = noise_threshold(leader, record)
            rows.append(
                {
                    "rank": i,
                    "emb": record["emb"],
                    "metric": record["primary_metric"],
                    "value": round(record["primary_value"], 4),
                    "config": record["config"],
                    "gap": round(gap, 4),
                    "within_noise": threshold is not None and gap <= threshold,
                }
            )
        out[split] = rows
    return out


def split_gap_table(records: list[dict]) -> list[dict]:
    """Per property task: mean test score of the PLMs under each split, and the drop.

    Averaged over every PLM, head and pooling, as in the paper's split ablation.
    """
    scored = [
        r
        for r in records
        if r["primary_value"] is not None
        and r["scope"] == "properties"
        and r["emb"] != NOPLM
    ]
    rows = []
    for (task,), group in sorted(_group(scored, "task").items()):
        by_split = _group(group, "split")
        random_runs, similarity_runs = (
            by_split.get(("random",)),
            by_split.get(("similarity",)),
        )
        if not random_runs or not similarity_runs:
            continue
        sign = group[0]["primary_sign"]
        random_mean = sum(r["primary_value"] for r in random_runs) / len(random_runs)
        similarity_mean = sum(r["primary_value"] for r in similarity_runs) / len(
            similarity_runs
        )
        rows.append(
            {
                "task": task,
                "metric": group[0]["primary_metric"],
                "random": round(random_mean, 4),
                "similarity": round(similarity_mean, 4),
                "harder_by": round(sign * (random_mean - similarity_mean), 4),
            }
        )
    return rows


def model_standing(emb: str) -> dict[str, dict[str, typing.Any]]:
    """Per leaderboard: a representation's five baselines with rating and interval.

    ``best`` is its highest-rated baseline with that baseline's rank among all
    competitors; ``mean`` the mean rating of its baselines.
    """
    out = {}
    for leaderboard in LEADERBOARDS:
        ratings = load_ratings(leaderboard)
        ranked = ratings.assign(rank=range(1, len(ratings) + 1))
        mine = ranked[ranked.embedding_model == emb]
        if mine.empty:
            continue
        rows = [
            {
                "config": config_label(r.model_family, r.pool_type),
                "rating": round(float(r.rating), 1),
                "ci_lo": round(float(r.ci_lo), 1),
                "ci_hi": round(float(r.ci_hi), 1),
                "rank": int(r.rank),
            }
            for r in mine.itertuples()
        ]
        out[leaderboard] = {
            "rows": rows,
            "best": rows[0],
            "mean": round(float(mine.rating.mean()), 1),
            "n_competitors": len(ratings),
        }
    return out
