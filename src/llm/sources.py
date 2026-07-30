"""Data loaders and aggregators for the RAG card generator (the "auto" layer).

Everything here is deliberately **torch-free** so the generator runs in the base
project environment (no `train`/`embeddings` extras required):

* Task configs come from ``data_loading/data_registry.py``, loaded standalone via
  ``importlib`` so we never trigger ``data_loading/__init__.py`` (which imports
  torch/rdkit).
* Metric conventions come from ``evaluation/metrics_registry.py`` (numpy/scipy
  only).
* All benchmark numbers *and* the exact per-run hyperparameters come from
  ``results_card.json`` at the repo root -- the artifact that is regenerated
  whenever results change. Nothing here is hand-copied.

These functions return plain dicts/lists that the Jinja templates render.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

EMBEDDINGS_MANIFEST = REPO_ROOT / "embeddings_models_output.json"
RESULTS_CARD = REPO_ROOT / "results_card.json"
BENCHMARK_CARD = REPO_ROOT / "benchmark_card.json"

# results_card.json uses "phopt" while the registry key is "ph_opt".
REGISTRY_TO_RESULTS_TASK = {"ph_opt": "phopt"}
RESULTS_TO_REGISTRY_TASK = {v: k for k, v in REGISTRY_TO_RESULTS_TASK.items()}

# The five prediction-head baselines, as (embedding_mode, model_family) pairs.
BASELINES: list[dict[str, str]] = [
    {"id": "linear", "embedding_mode": "no_attention", "model_family": "linear",
     "title": "Linear head (no attention)"},
    {"id": "mlp", "embedding_mode": "no_attention", "model_family": "mlp",
     "title": "MLP head (no attention)"},
    {"id": "xgboost", "embedding_mode": "no_attention", "model_family": "xgboost",
     "title": "XGBoost head (no attention)"},
    {"id": "attention_linear", "embedding_mode": "attention", "model_family": "linear",
     "title": "Attention-pooled linear head"},
    {"id": "attention_mlp", "embedding_mode": "attention", "model_family": "mlp",
     "title": "Attention-pooled MLP head"},
]

# The two data splits. results_card buckets "similarity" under "loco".
SPLITS: list[dict[str, str]] = [
    {"id": "random", "split_family": "random", "title": "Random split"},
    {"id": "similarity", "split_family": "loco", "title": "Similarity (MMseqs2) split"},
]


# ---------------------------------------------------------------------------
# Torch-free module loading
# ---------------------------------------------------------------------------

def _load_standalone(mod_name: str, rel_path: str):
    """Execute a single source file as a module without importing its package.

    Registering the module in ``sys.modules`` *before* executing it is required
    so that dataclass/enum machinery can resolve ``cls.__module__`` back to a
    real module object.
    """
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_task_registry() -> dict[str, Any]:
    """Return ``TASK_REGISTRY`` (name -> TaskConfig) without importing torch."""
    module = _load_standalone("_rag_data_registry", "src/data_loading/data_registry.py")
    return module.TASK_REGISTRY


def _metric_conventions():
    """Return (primary_metric_for, higher_is_better_for) from the registry."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from evaluation.metrics_registry import higher_is_better_for, primary_metric_for
    return primary_metric_for, higher_is_better_for


# ---------------------------------------------------------------------------
# Static/manifest sources
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_models() -> dict[str, dict[str, Any]]:
    """Foundation-model auto facts: embedding dim + shard path per model key.

    Cached: the manifest is read once per process. Treat the result as read-only
    (it is shared across callers). A regenerated manifest requires a fresh process.
    """
    data = json.loads(EMBEDDINGS_MANIFEST.read_text())
    return {
        key: {"dim": val["dim"], "shard_path": val["path"]}
        for key, val in data.items()
    }


def load_benchmark_card() -> dict[str, Any]:
    return json.loads(BENCHMARK_CARD.read_text())


def dataset_stats_for(task_cfg) -> dict[str, Any] | None:
    """Load ``dataset_card.json`` living next to a task's processed CSVs.

    Keyed off the CSV's parent directory (handles ``ddg`` -> ``ddg_mut``).
    Only the 10 ``properties`` tasks have one; real_case returns ``None``.
    """
    if not task_cfg.dataset_paths:
        return None
    any_csv = next(iter(task_cfg.dataset_paths.values()))
    stats_path = Path(any_csv).parent / "dataset_card.json"
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def benchmark_entry_for(card: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    """Return the benchmark_card.json entry for a task (either scope)."""
    for scope in ("properties", "real_case"):
        if task_name in card.get(scope, {}):
            return card[scope][task_name]
    return None


# ---------------------------------------------------------------------------
# results_card.json -> tidy records (the numeric auto layer)
# ---------------------------------------------------------------------------

def iter_raw_entries():
    """Yield every leaf entry in results_card.json (even runs without metrics)."""
    card = json.loads(RESULTS_CARD.read_text())
    for by_split in card.get("predictions", {}).values():
        for by_family in by_split.values():
            for by_task in by_family.values():
                for by_emb in by_task.values():
                    for entry in by_emb.values():
                        if entry:
                            yield entry


@functools.lru_cache(maxsize=1)
def load_records() -> list[dict[str, Any]]:
    """Flatten results_card.json into tidy per-run records.

    Each record captures the identity of a run plus its held-out (test) primary
    metric, so downstream aggregations stay simple.

    Cached: parsed once per process and shared across callers, which read but never
    mutate the records. A regenerated results_card requires a fresh process.
    """
    primary_metric_for, higher_is_better_for = _metric_conventions()
    card = json.loads(RESULTS_CARD.read_text())
    predictions = card.get("predictions", {})

    records: list[dict[str, Any]] = []
    for embedding_mode, by_split in predictions.items():
        for split_family, by_family in by_split.items():
            for model_family, by_task in by_family.items():
                for task, by_emb in by_task.items():
                    for emb, entry in by_emb.items():
                        metrics = (entry or {}).get("metrics") or {}
                        test = metrics.get("test") or {}
                        task_type = test.get("task_type")
                        if not task_type:
                            continue
                        primary = primary_metric_for(task_type)
                        value = test.get(primary)
                        signs = higher_is_better_for(task_type)
                        records.append({
                            "embedding_mode": embedding_mode,
                            "split_family": split_family,
                            "model_family": model_family,
                            "task": task,
                            "emb": emb,
                            "model_name": entry.get("model_name"),
                            "task_scope": entry.get("task_scope", "properties"),
                            "task_type": task_type,
                            "params": entry.get("params") or {},
                            "primary_metric": primary,
                            "primary_sign": signs.get(primary, 1),
                            "primary_value": value if _is_finite(value) else None,
                            "n_test": test.get("n"),
                        })
    return records


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _avg_ranks(values: list[float], sign: int) -> list[float]:
    """Competition-style ranks with averaged ties; 1 = best (sign-aware)."""
    order = sorted(range(len(values)), key=lambda i: sign * values[i], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sign * values[order[j + 1]] == sign * values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _group(records, *keys):
    grouped: dict[tuple, list] = {}
    for rec in records:
        grouped.setdefault(tuple(rec[k] for k in keys), []).append(rec)
    return grouped


def _rank_score(rank: float, n: int) -> float:
    """Normalize a within-group rank to a 0-1 score, 1 = best (fraction of field beaten).

    ``(n - rank) / (n - 1)``: first of any field scores 1.0, last scores 0.0, and the
    midpoint is 0.5 regardless of ``n``. This is what makes scores from a 2-model group
    and a 14-model group comparable — a raw rank is not, since 2nd of 2 is *last* while
    2nd of 14 is near the top. Averaged ties land between their neighbours' scores.
    """
    return (n - rank) / (n - 1)


def model_standing(records: list[dict[str, Any]]) -> dict[str, dict[tuple, dict]]:
    """Mean competition score of each embedding within (embedding_mode, split_family).

    For every (embedding_mode, split_family, model_family, task) group we rank the
    competing embeddings by the primary metric, convert each rank to a size-normalized
    ``score`` (:func:`_rank_score`, 1 = best), then average those scores per embedding
    across all groups sharing an (embedding_mode, split_family).

    ``score`` is the headline number *because groups are not the same size*: sparse
    groups (some tasks ran only 2-5 of the 14 models) hand their members a flattering
    raw rank that is not comparable with a 14-way group's. ``mean_rank`` is still
    reported alongside it for continuity, but it is only meaningful within one group.
    This is a local proxy for the deferred Bradley-Terry Elo.
    """
    scored = [r for r in records if r["primary_value"] is not None]
    groups = _group(scored, "embedding_mode", "split_family", "model_family", "task")

    # accumulate: standing[emb][(embedding_mode, split_family)] = [(rank, score)...]
    acc: dict[str, dict[tuple, list[tuple[float, float]]]] = {}
    for (emode, sfam, _mf, _task), recs in groups.items():
        if len(recs) < 2:  # a one-horse race has no ranking information
            continue
        values = [r["primary_value"] for r in recs]
        ranks = _avg_ranks(values, recs[0]["primary_sign"])
        for rec, rank in zip(recs, ranks):
            acc.setdefault(rec["emb"], {}).setdefault((emode, sfam), []).append(
                (rank, _rank_score(rank, len(recs))))

    standing: dict[str, dict[tuple, dict]] = {}
    for emb, buckets in acc.items():
        for bucket, pairs in buckets.items():
            standing.setdefault(emb, {})[bucket] = {
                "score": round(sum(s for _, s in pairs) / len(pairs), 3),
                "mean_rank": round(sum(r for r, _ in pairs) / len(pairs), 2),
                "n": len(pairs),
            }
    return standing


def best_by_task_for_model(records, emb) -> dict[str, list[dict]]:
    """Per split_family, this model's best test primary per task (across heads)."""
    mine = [r for r in records if r["emb"] == emb and r["primary_value"] is not None]
    out: dict[str, list[dict]] = {}
    for (sfam, task), recs in _group(mine, "split_family", "task").items():
        sign = recs[0]["primary_sign"]
        best = max(recs, key=lambda r: sign * r["primary_value"])
        out.setdefault(sfam, []).append({
            "task": _display_task(task),
            "metric": best["primary_metric"],
            "value": round(best["primary_value"], 4),
            "head": best["model_name"],
            "n_test": best["n_test"],
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["task"])
    return out


def baseline_task_table(records, embedding_mode, model_family) -> dict[str, list[dict]]:
    """Per split_family, this head's mean/best test primary per task (across models)."""
    mine = [
        r for r in records
        if r["embedding_mode"] == embedding_mode
        and r["model_family"] == model_family
        and r["primary_value"] is not None
    ]
    out: dict[str, list[dict]] = {}
    for (sfam, task), recs in _group(mine, "split_family", "task").items():
        sign = recs[0]["primary_sign"]
        values = [r["primary_value"] for r in recs]
        best = max(values) if sign > 0 else min(values)
        out.setdefault(sfam, []).append({
            "task": _display_task(task),
            "metric": recs[0]["primary_metric"],
            "mean": round(sum(values) / len(values), 4),
            "best": round(best, 4),
            "n_models": len(values),
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["task"])
    return out


def task_leaderboard(records, results_task) -> dict[str, list[dict]]:
    """Per split_family, rank embeddings by their best test primary on this task."""
    mine = [r for r in records if r["task"] == results_task and r["primary_value"] is not None]
    out: dict[str, list[dict]] = {}
    for (sfam,), recs in _group(mine, "split_family").items():
        sign = recs[0]["primary_sign"]
        per_emb: dict[str, dict] = {}
        for rec in recs:
            cur = per_emb.get(rec["emb"])
            if cur is None or sign * rec["primary_value"] > sign * cur["primary_value"]:
                per_emb[rec["emb"]] = rec
        ranked = sorted(per_emb.values(), key=lambda r: sign * r["primary_value"], reverse=True)
        out[sfam] = [{
            "rank": i + 1,
            "emb": rec["emb"],
            "metric": rec["primary_metric"],
            "value": round(rec["primary_value"], 4),
            "head": rec["model_name"],
        } for i, rec in enumerate(ranked)]
    return out


def split_gap_table(records) -> list[dict]:
    """Per properties task: mean test primary under random vs similarity + delta.

    Aggregated across all embeddings and heads. Only ``properties`` tasks have
    both splits; real_case is random-only and excluded.
    """
    scored = [
        r for r in records
        if r["primary_value"] is not None and r["task_scope"] == "properties"
    ]
    rows = []
    for (task,), recs in _group(scored, "task").items():
        by_split = _group(recs, "split_family")
        rnd = by_split.get(("random",))
        loco = by_split.get(("loco",))
        if not rnd or not loco:
            continue
        metric = recs[0]["primary_metric"]
        sign = recs[0]["primary_sign"]
        rnd_mean = sum(r["primary_value"] for r in rnd) / len(rnd)
        loco_mean = sum(r["primary_value"] for r in loco) / len(loco)
        # "harder" delta: positive means similarity split is worse.
        delta = sign * (rnd_mean - loco_mean)
        rows.append({
            "task": _display_task(task),
            "metric": metric,
            "random": round(rnd_mean, 4),
            "similarity": round(loco_mean, 4),
            "harder_by": round(delta, 4),
        })
    rows.sort(key=lambda r: r["task"])
    return rows


def _display_task(results_task: str) -> str:
    """Map a results_card task name back to its registry-canonical name."""
    return RESULTS_TO_REGISTRY_TASK.get(results_task, results_task)


def results_task_name(registry_name: str) -> str:
    return REGISTRY_TO_RESULTS_TASK.get(registry_name, registry_name)
