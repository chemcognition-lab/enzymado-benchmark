#!/usr/bin/env python
"""Training-noise SD of every rated cell, for the tie margins of compute_ratings.py.

For each task, split, metric, head and pooling, sigma is the median over PLMs of the
standard deviation of that PLM's seeds, each divided by

    c_n = sqrt(median(chi2_{n-1}) / (n - 1))

so that under normal noise the estimate targets sigma itself (c_3 = 0.83, c_2 = 0.67).
The median keeps one PLM with a collapsed seed from setting the margin of its whole
configuration, which a pooled variance cannot. Seed 42 is the published tree; the
other seeds are retrainings of the same runs.

Regression runs whose training R2 falls below a threshold (default -1, a training
RMSE above sqrt(2) times the target SD) are left out of sigma only; the leaderboard
keeps them. A PLM counts when at least two of its seeds remain. A configuration with
no PLM left borrows the sigma of the same task, split, head and metric without
attention, so a broken model does not earn ties from its own spread. A cell without
seed runs gets sigma 0 when its head is deterministic in the seed (mean-pooled
XGBoost on the scalar tasks).

    python scripts/results/compute_tie_margins.py
    python scripts/results/compute_tie_margins.py --out /tmp/tie_margins.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

_REPO = Path(__file__).resolve().parents[2]
RATED = {
    "regression": ("mae", "rmse", "kendall_tau"),
    "classification": ("accuracy", "precision", "recall", "f1", "auroc", "auprc"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-root", type=Path, default=_REPO / "results" / "predictions",
                        help="Published runs, used as seed 42.")
    parser.add_argument("--seed-root", type=Path, action="append", default=None,
                        help="Predictions tree of one extra seed (repeatable).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV (default: the path in the card's rating.tie_margins).")
    parser.add_argument("--benchmark-card", type=Path, default=_REPO / "benchmark_card.json")
    parser.add_argument("--exploded-train-r2", type=float, default=-1.0)
    return parser.parse_args()


def consistency(n: int) -> float:
    """Median of the sample SD of n normal draws, in units of sigma."""
    return float(np.sqrt(chi2.median(n - 1) / (n - 1)))


def main() -> None:
    args = parse_args()
    seed_roots = args.seed_root or [
        _REPO / "results" / "noise_seeds" / "seed_43" / "predictions",
        _REPO / "results" / "noise_seeds" / "seed_44" / "predictions",
    ]
    trees = {"published": args.predictions_root, **{str(p): p for p in seed_roots}}
    out = args.out
    if out is None:
        card = json.loads(args.benchmark_card.read_text())
        out = _REPO / card["rating"]["tie_margins"]["path"]

    runs = collections.defaultdict(dict)  # (pool, split, head, task) -> PLM -> tree -> test metrics
    exploded = set()
    cells = set()
    for name, tree in trees.items():
        for path in tree.glob("*/*/*/*/*/*/metrics.json"):
            _, pool, split, head, task, emb = path.parent.relative_to(tree).parts
            key = (pool, split, head, task)
            if name == "published":
                cells.add(key)
            metrics = json.loads(path.read_text())
            runs[key].setdefault(emb, {})[name] = metrics["test"]
            train_r2 = metrics.get("train", {}).get("r2", 0.0)
            if metrics["test"]["task_type"] == "regression" and train_r2 < args.exploded_train_r2:
                exploded.add(key + (emb, name))

    rows = []
    for key in sorted(cells):
        pool, split, head, task = key
        complete, dropped = {}, 0
        for emb, by_tree in runs[key].items():
            if len(by_tree) < len(trees):
                continue
            kept = {t: m for t, m in by_tree.items() if key + (emb, t) not in exploded}
            dropped += len(by_tree) - len(kept)
            if len(kept) >= 2:
                complete[emb] = kept
        task_type = next(iter(runs[key].values()))["published"]["task_type"]
        for metric in RATED[task_type]:
            if complete:
                groups = [np.array([m[metric] for m in kept.values()]) for kept in complete.values()]
                sigma = float(np.median([g.std(ddof=1) / consistency(len(g)) for g in groups]))
            elif head == "xgboost" and pool == "no_attention":
                sigma = 0.0
            else:
                sigma = float("nan")
            rows.append(dict(task=task, split=split, metric=metric, model_family=head, pool_type=pool,
                             sigma=sigma, n_plms=len(complete), n_runs_dropped=dropped, source="measured"))

    index = {(r["task"], r["split"], r["metric"], r["model_family"], r["pool_type"]): r for r in rows}
    for row in rows:
        if not np.isfinite(row["sigma"]) and row["pool_type"] == "attention":
            donor = index.get((row["task"], row["split"], row["metric"], row["model_family"], "no_attention"))
            if donor and np.isfinite(donor["sigma"]):
                row["sigma"], row["source"] = donor["sigma"], "no_attention_fallback"
    missing = [r for r in rows if not np.isfinite(r["sigma"])]
    if missing:
        raise ValueError(f"No sigma for {len(missing)} cells, e.g. {missing[0]}.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fallbacks = sum(r["source"] != "measured" for r in rows)
    print(f"{len(rows)} cells, {fallbacks} fallbacks, {len(exploded)} exploded runs left out -> {out}")


if __name__ == "__main__":
    main()
