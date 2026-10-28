#!/usr/bin/env python
"""Check that PLMs within one configuration share their training noise.

The tie margin gives every PLM of a head, pooling, task, split and metric the same
sigma. With three seeds the standard deviation of each PLM varies widely even when
the true noise is identical, so the observed spread of log SD across PLMs is
compared with a simulation where 14 PLMs share one sigma. Uses the configurations
where all 14 PLMs have three seeds: mean-pooled linear and MLP on the scalar
property tasks and the real-case datasets. Writes the observed spread per cell and
the simulated distribution, pooled over cells, for the noise figure.

Example:
    python scripts/paper/shared_sigma.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
RATED = {
    "regression": ("kendall_tau", "mae"),
    "classification": ("auroc", "auprc", "f1", "precision", "recall"),
}
SEEDS = (42, 43, 44)
N_SIM = 5000
BINS = np.round(np.arange(0, 2.001, 0.02), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_P.results_dir / "paper")
    return parser.parse_args()


def load_seed_scores() -> dict[tuple[str, ...], dict[str, dict[int, dict]]]:
    """Test metrics per configuration, PLM and seed; exploded runs left out."""
    trees = {42: _P.results_dir / "predictions"}
    trees |= {
        s: _P.results_dir / "noise_seeds" / f"seed_{s}" / "predictions"
        for s in SEEDS[1:]
    }
    runs = collections.defaultdict(lambda: collections.defaultdict(dict))
    for seed, tree in trees.items():
        for path in tree.glob("*/no_attention/*/*/*/*/metrics.json"):
            scope, _pool, split, head, task, emb = path.parent.relative_to(tree).parts
            if head not in ("linear", "mlp") or emb == "noplm":
                continue
            run = json.loads(path.read_text())
            train_r2 = run.get("train", {}).get("r2", 0)  # seed runs may lack it
            if run["test"]["task_type"] == "regression" and train_r2 < -1:
                continue
            runs[(scope, split, head, task)][emb][seed] = run["test"]
    return runs


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)
    null_counts = np.zeros(len(BINS) - 1)
    attr = pd.read_csv(
        _P.scripts_dir / "figures" / "plm_attributes.csv", index_col="emb"
    )
    rows = []
    for key, by_emb in load_seed_scores().items():
        complete = {e: s for e, s in by_emb.items() if len(s) == len(SEEDS)}
        if len(complete) < 14:
            continue
        task_type = next(iter(complete.values()))[42]["task_type"]
        for metric in RATED[task_type]:
            sd = {
                e: np.std([s[k][metric] for k in SEEDS], ddof=1)
                for e, s in complete.items()
            }
            observed = np.log(np.maximum(list(sd.values()), 1e-12)).std()
            draws = rng.standard_normal((N_SIM, len(sd), len(SEEDS)))
            null = np.log(draws.std(axis=2, ddof=1)).std(axis=1)  # 14 PLMs, one sigma
            null_counts += np.histogram(null, BINS)[0]
            rows.append(
                dict(
                    scope=key[0], head=key[2], metric=metric, sd=sd, observed=observed,
                    null_median=np.median(null), p=(null >= observed).mean(),
                )
            )  # fmt: skip
    cells = pd.DataFrame(rows)
    print(f"configurations x metrics with all 14 PLMs: {len(cells)}")
    print(
        f"SD of log seed-SD across PLMs: observed {cells.observed.median():.2f}, "
        f"shared-sigma simulation {cells.null_median.median():.2f}"
    )
    print(
        f"cells with more spread than chance (p < 0.05): {(cells.p < 0.05).mean():.0%}"
    )
    print(
        cells.assign(above=cells.p < 0.05)
        .groupby(["scope", "head"])
        .above.mean()
        .round(2)
    )

    ranks = collections.defaultdict(list)
    for sd in cells.sd:
        for rank, emb in enumerate(sorted(sd, key=sd.get), start=1):
            ranks[emb].append(rank)
    mean_rank = pd.Series({e: np.mean(v) for e, v in ranks.items()}).sort_values()
    print("\nmean noise rank (1 = least noisy, 14 = noisiest, 7.5 if equal):")
    print(pd.DataFrame({"rank": mean_rank, "dim": attr.dim}).round(2).to_string())
    rho = stats.spearmanr(attr.dim.reindex(mean_rank.index), mean_rank)
    print(f"Spearman(width, noise rank) rho {rho.statistic:.2f} (p {rho.pvalue:.3f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.drop(columns="sd").to_csv(
        args.out_dir / "shared_sigma_cells.csv", index=False
    )
    density = null_counts / null_counts.sum() / np.diff(BINS)
    pd.DataFrame(dict(lo=BINS[:-1], hi=BINS[1:], density=density)).to_csv(
        args.out_dir / "shared_sigma_null.csv", index=False
    )
    print(f"wrote shared_sigma_cells.csv and shared_sigma_null.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
