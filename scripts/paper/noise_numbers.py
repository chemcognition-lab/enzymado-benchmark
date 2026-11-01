#!/usr/bin/env python
"""Training-noise numbers of the paper, from the seed trees and the tie margins.

Prints the extra seed runs, the exploded runs left out of sigma, the noise table
(sigma, between-PLM SD, their ratio, and the share of PLM pairs within the margin),
the median sigma per head and pooling, the collapsed EC attention seed, the
fallback margins, and the PLMs drawn for the attention, EC and residue seeds.
With --fresh-margins it also compares a fresh compute_tie_margins.py output with
the published table.

Example:
    python scripts/paper/noise_numbers.py
    python scripts/results/compute_tie_margins.py --out /tmp/tie_margins.csv
    python scripts/paper/noise_numbers.py --fresh-margins /tmp/tie_margins.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from enzymado import paths_and_constants

_P = paths_and_constants.Paths
MARGINS = _P.results_dir / "noise_seeds" / "tie_margins_v4.csv"
KEY = ["task", "split", "metric", "model_family", "pool_type"]
PRIMARY = {"regression": "kendall_tau", "classification": "auprc"}
C3 = 0.8326  # c_n for three seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-margins", type=Path, default=None)
    parser.add_argument("--exploded-train-r2", type=float, default=-1.0)
    return parser.parse_args()


def load_runs() -> pd.DataFrame:
    trees = {42: _P.results_dir / "predictions"}
    trees |= {
        s: _P.results_dir / "noise_seeds" / f"seed_{s}" / "predictions"
        for s in (43, 44)
    }
    rows = []
    for seed, tree in trees.items():
        for path in tree.glob("*/*/*/*/*/*/metrics.json"):
            scope, pool, split, head, task, emb = path.parent.relative_to(tree).parts
            run = json.loads(path.read_text())
            test = run["test"]
            rows.append(
                dict(
                    seed=seed, scope=scope, pool=pool, split=split, head=head,
                    task=task, emb=emb, task_type=test["task_type"],
                    train_r2=run.get("train", {}).get("r2"),
                    primary=test[PRIMARY[test["task_type"]]],
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def seed_runs(runs: pd.DataFrame, r2_floor: float) -> None:
    extra = runs[runs.seed != 42]
    print(
        f"runs per seed {runs.groupby('seed').size().to_dict()}, extra {len(extra):,}"
    )
    print(extra.groupby(["scope", "pool", "head"]).size().to_dict())
    exploded = runs[(runs.task_type == "regression") & (runs.train_r2 < r2_floor)]
    seeded = extra[["scope", "pool", "split", "head", "task", "emb"]].drop_duplicates()
    within = exploded.merge(seeded)
    glucosidase = within.task.str.contains("glucosidase").sum()
    print(
        f"exploded (train R2 < {r2_floor:g}): {len(exploded)} in all, {len(within)} "
        f"among PLMs with seed runs, {glucosidase} of them on glucosidase"
    )
    drawn = sorted(extra[extra.pool == "attention"].emb.unique())
    print("PLMs drawn for the attention, EC and residue seeds:", drawn)


def noise_table(
    runs: pd.DataFrame, margins: pd.DataFrame, real_tasks: set[str]
) -> None:
    """Medians over tasks and splits of the primary metric, per head and pooling."""
    primary = (
        margins[margins.metric.isin(PRIMARY.values())]
        .set_index(KEY[:2] + KEY[3:])
        .sigma
    )
    plm = runs[(runs.seed == 42) & (runs.emb != "noplm") & (runs["head"] != "xgboost")]
    rows = []
    for (task, split, head, pool), group in plm.groupby(
        ["task", "split", "head", "pool"]
    ):
        sigma = primary[task, split, head, pool]
        between = group.primary.std(ddof=1)
        margin = 1.96 * np.sqrt(2) * sigma
        pairs = itertools.combinations(group.primary, 2)
        tied = np.mean([abs(a - b) < margin for a, b in pairs])
        scope = "real_case" if task in real_tasks else "properties"
        rows.append(
            dict(
                scope=scope, head=head, pool=pool, sigma=sigma, between=between,
                ratio=sigma / between, tied=tied,
            )
        )  # fmt: skip
    table = pd.DataFrame(rows).groupby(["scope", "head", "pool"]).median()
    print(table.round(3).to_string())


def median_sigma(margins: pd.DataFrame, real_tasks: set[str]) -> None:
    neural = margins[margins.model_family != "xgboost"]
    primary = neural[neural.metric.isin(PRIMARY.values())]
    for label, df in (
        ("primary metric, both leaderboards", primary),
        ("primary metric, properties", primary[~primary.task.isin(real_tasks)]),
    ):
        sigma = df.groupby(["model_family", "pool_type"]).sigma.median()
        print(f"median sigma, {label}: {sigma.round(4).to_dict()}")


def ec_attention(runs: pd.DataFrame, margins: pd.DataFrame) -> None:
    """The EC attention MLP cell, where one seed collapses."""
    ec = runs[
        (runs.task == "ec_number")
        & (runs.pool == "attention")
        & (runs["head"] == "mlp")
    ]
    keep = (margins.task == "ec_number") & (margins.pool_type == "attention")
    keep &= (margins.model_family == "mlp") & (margins.metric == "auprc")
    published = margins[keep].set_index("split").sigma
    for split, group in ec.groupby("split"):
        scores = group.pivot(index="emb", columns="seed", values="primary").dropna()
        pooled = np.sqrt(scores.var(axis=1, ddof=1).mean())
        median = np.median(scores.std(axis=1, ddof=1) / C3)
        print(
            f"{split}: pooled sigma {pooled:.4f}, median {median:.5f}, "
            f"published {published[split]:.5f}"
        )
        print(scores.round(3).to_string())


def main() -> None:
    args = parse_args()
    margins = pd.read_csv(MARGINS)
    runs = load_runs()
    real_tasks = set(runs[runs.scope == "real_case"].task)
    print(f"tie margins: {len(margins)} cells")
    fallback = margins[margins.source != "measured"]
    print("fallback margins:", fallback[KEY + ["sigma"]].round(4).values.tolist())
    if args.fresh_margins is not None:
        fresh = margins.merge(
            pd.read_csv(args.fresh_margins), on=KEY, suffixes=("", "_new")
        )
        print(
            "fresh vs published: max |diff| "
            f"{(fresh.sigma - fresh.sigma_new).abs().max():.2e}"
        )
    print()
    seed_runs(runs, args.exploded_train_r2)
    print("\nNoise table (primary metric):")
    noise_table(runs, margins, real_tasks)
    print()
    median_sigma(margins, real_tasks)
    print("\nEC number, attention, MLP:")
    ec_attention(runs, margins)
    xgb = runs[(runs["head"] == "xgboost") & (runs.seed != 42)]
    print(
        "\nXGBoost seed runs:", xgb.groupby("task").emb.unique().map(sorted).to_dict()
    )


if __name__ == "__main__":
    main()
