#!/usr/bin/env python
"""Check that seed noise is normal, as the tie margin 1.96 sqrt(2) sigma assumes.

Every PLM with at least two kept seeds gives one gap d per pair of seeds. Under
normal noise with the cell's sigma, z = |d| / (sqrt(2) sigma) is half-normal, so 5%
of the pairs exceed 1.96. sigma comes from the same seeds, so the expected shares
come from a simulation with the same PLMs and seed counts per cell, normal noise,
one shared sigma, and the median estimator of compute_tie_margins.py. Uses every
rated metric with a nonzero sigma; exploded runs are left out as in the margins.

Writes the share of pairs beyond each threshold, observed and simulated, per group.

Example:
    python scripts/paper/noise_normality.py
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats

from enzymado import paths_and_constants

_P = paths_and_constants.Paths
RATED = {
    "regression": ("kendall_tau", "mae"),
    "classification": ("auroc", "auprc", "f1", "precision", "recall"),
}
PRIMARY = ("kendall_tau", "auprc")
SEEDS = (42, 43, 44)
RESIDUE_EC = ("ec_number", "active_sites", "binding_site")
CUTS = (1.96, 2.58, 3.29)  # two-sided 5%, 1%, 0.1%
GRID = np.round(np.arange(0, 6.01, 0.02), 2)
N_SIM = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_P.results_dir / "paper")
    return parser.parse_args()


def c_n(n: int) -> float:
    return float(np.sqrt(stats.chi2.median(n - 1) / (n - 1)))


def load_cells() -> dict[tuple[str, ...], dict[str, list[dict]]]:
    """Kept test metrics per (pool, split, head, task) and PLM, as in the margins."""
    trees = {42: _P.results_dir / "predictions"}
    trees |= {
        s: _P.results_dir / "noise_seeds" / f"seed_{s}" / "predictions"
        for s in SEEDS[1:]
    }
    runs = collections.defaultdict(dict)
    exploded = set()
    for seed, tree in trees.items():
        for path in tree.glob("*/*/*/*/*/*/metrics.json"):
            _, pool, split, head, task, emb = path.parent.relative_to(tree).parts
            if emb == "noplm":
                continue
            run = json.loads(path.read_text())
            runs[(pool, split, head, task, emb)][seed] = run["test"]
            train_r2 = run.get("train", {}).get("r2", 0.0)
            if run["test"]["task_type"] == "regression" and train_r2 < -1:
                exploded.add((pool, split, head, task, emb, seed))
    cells = collections.defaultdict(dict)
    for key, by_seed in runs.items():
        if len(by_seed) < len(SEEDS):
            continue
        kept = [t for s, t in by_seed.items() if key + (s,) not in exploded]
        if len(kept) >= 2:
            cells[key[:4]][key[4]] = kept
    return cells


def pair_z(groups: list[np.ndarray], sigma: np.ndarray) -> np.ndarray:
    """|d| / (sqrt(2) sigma) for every seed pair; groups are (..., PLMs, seeds)."""
    gaps = [
        np.abs(g[..., a] - g[..., b])
        for g in groups
        for a, b in itertools.combinations(range(g.shape[-1]), 2)
    ]
    return np.concatenate(gaps, axis=-1) / (np.sqrt(2) * sigma[..., None])


def beyond(z: np.ndarray) -> np.ndarray:
    """Share of z above each threshold of GRID."""
    return 1 - np.searchsorted(np.sort(z), GRID, side="right") / len(z)


def median_sigma(groups: list[np.ndarray]) -> np.ndarray:
    """Median over PLMs of seed SD / c_n; groups are (..., PLMs, seeds)."""
    sds = [g.std(axis=-1, ddof=1) / c_n(g.shape[-1]) for g in groups]
    return np.median(np.concatenate(sds, axis=-1), axis=-1)


def main() -> None:
    args = parse_args()
    started = time.time()
    cells = load_cells()
    print(f"cells with seed runs: {len(cells)} ({time.time() - started:.0f} s)")

    rng = np.random.default_rng(0)
    rows, curves = [], collections.defaultdict(lambda: np.zeros(len(GRID)))
    for (pool, split, head, task), by_emb in sorted(cells.items()):
        task_type = next(iter(by_emb.values()))[0]["task_type"]
        drawn = pool == "attention" or task in RESIDUE_EC
        group = "4 PLMs" if drawn else "14 PLMs"
        by_size = collections.defaultdict(list)
        for kept in by_emb.values():
            by_size[len(kept)].append(kept)
        for metric in RATED[task_type]:
            groups = [
                np.array([[t[metric] for t in kept] for kept in plms])
                for plms in by_size.values()
            ]
            sigma = median_sigma(groups)
            if not sigma > 1e-9:  # deterministic cells have no shape to test
                continue
            z = pair_z(groups, sigma)
            null_groups = [rng.standard_normal((N_SIM, *g.shape)) for g in groups]
            null_share = beyond(pair_z(null_groups, median_sigma(null_groups)).ravel())
            curves[group] += len(z) * null_share
            curves["all"] += len(z) * null_share
            rows += [
                dict(
                    pool=pool, split=split, head=head, task=task, metric=metric,
                    group=group, z=v,
                )
                for v in z
            ]  # fmt: skip
    pairs = pd.DataFrame(rows)
    n_cells = len(pairs.groupby(["pool", "split", "head", "task", "metric"]))
    print(
        f"cells x metrics {n_cells}, seed pairs {len(pairs):,} "
        f"({time.time() - started:.0f} s)"
    )

    sizes = pairs.groupby("group").size().to_dict() | {"all": len(pairs)}
    normal = {g: curve / sizes[g] for g, curve in curves.items()}
    print("\nshare of seed pairs beyond each threshold, observed against normal:")
    for name, subset in (
        ("all", pairs),
        ("primary metric", pairs[pairs.metric.isin(PRIMARY)]),
        *((g, pairs[pairs.group == g]) for g in ("14 PLMs", "4 PLMs")),
        *((m, pairs[pairs.metric == m]) for m in itertools.chain(*RATED.values())),
    ):
        shares = "  ".join(f">{c}: {(subset.z > c).mean():.1%}" for c in CUTS)
        print(f"  {name:<15} pairs {len(subset):>5}  {shares}")
    for g, share in normal.items():
        at = "  ".join(f">{c}: {share[np.searchsorted(GRID, c)]:.1%}" for c in CUTS)
        print(f"  normal, {g:<7} {at}")

    print("\nquantiles of z (half-normal: 50% 0.67, 90% 1.64, 95% 1.96, 99% 2.58)")
    print(f"  {pairs.z.quantile([0.5, 0.9, 0.95, 0.99]).round(2).to_dict()}")
    print("largest gaps:")
    print(pairs.sort_values("z", ascending=False).head(8).round(1).to_string())

    curve = pd.concat(
        pd.DataFrame(
            dict(
                group=g, t=GRID, observed=beyond(subset.z.to_numpy()), normal=normal[g]
            )
        )
        for g, subset in (("all", pairs), *pairs.groupby("group"))
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.out_dir / "noise_normality_curve.csv", index=False)
    print(f"wrote noise_normality_curve.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
