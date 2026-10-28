#!/usr/bin/env python
"""Paper numbers from the test metrics of the published runs (seed 42).

Prints the split penalty and how the two split orderings correlate, the heatmap
rows, the split, pooling and head ablations, the gains over the 2020 ProtT5
models, the no-PLM gaps, the real-case medians, and EC accuracy under imbalance.
Scores are the primary metric: Kendall's tau for regression, AUPRC for
classification.

Example:
    python scripts/paper/score_numbers.py
"""

from __future__ import annotations

import json

import pandas as pd
from scipy import stats

from enzymado import paths_and_constants, vis

_P = paths_and_constants.Paths()
PROT5_2020 = ("prot5_xl_uniref50", "prot5_xl_half_uniref50")


def split_penalty(plm: pd.DataFrame) -> None:
    mlp = plm[(plm["head"] == "mlp") & (plm.pool == "no_attention")]
    family = mlp.groupby(["task_type", "emb", "split"]).value.mean().unstack()
    family["drop"] = family.random - family.similarity
    for task_type in ("regression", "classification"):
        f = family.loc[task_type]
        rho = stats.spearmanr(f.random, f.similarity)
        print(
            f"{task_type}: drop {f['drop'].min():.4f} ({f['drop'].idxmin()}) to "
            f"{f['drop'].max():.4f} ({f['drop'].idxmax()}), "
            f"ratio {f['drop'].max() / f['drop'].min():.2f}, "
            f"mean {f['drop'].mean():.4f}, all positive {bool((f['drop'] > 0).all())}"
            f" | Spearman rho {rho.statistic:.3f} (p {rho.pvalue:.3f})"
        )
    rank = mlp.assign(rank=mlp.groupby(["task", "split"]).value.rank(ascending=False))
    mean_rank = rank.groupby(["emb", "split"])["rank"].mean().unstack()
    rho = stats.spearmanr(mean_rank.random, mean_rank.similarity)
    print(f"mean rank over tasks: rho {rho.statistic:.3f} (p {rho.pvalue:.3f})")


def ablations(plm: pd.DataFrame) -> None:
    split = plm.pivot_table(index="task", columns="split", values="value")
    split["change"] = split.similarity - split.random
    print("Split ablation, all heads and poolings:")
    print(split.round(3).to_string())

    mean_pool = plm[plm.pool == "no_attention"]
    print(
        "\nHeads under mean pooling:",
        mean_pool.groupby("head").value.mean().round(3).to_dict(),
    )
    print(mean_pool.groupby(["task_type", "head"]).value.mean().round(3).to_dict())
    heads = mean_pool.pivot_table(index="task", columns="head", values="value")
    print(heads.round(3).to_string())
    print("best head:", heads.idxmax(axis=1).to_dict())

    neural = plm[plm["head"].isin(["linear", "mlp"])]
    print(
        "\nPooling, linear and MLP:",
        neural.groupby("pool").value.mean().round(3).to_dict(),
    )
    pool = neural.pivot_table(index="task", columns="pool", values="value")
    pool["change"] = pool.attention - pool.no_attention
    print(pool.round(3).to_string())
    print(f"tasks where attention loses: {(pool.change < 0).sum()} of {len(pool)}")
    by_emb = neural.pivot_table(index="emb", columns="pool", values="value")
    change = (by_emb.attention - by_emb.no_attention).sort_values()
    print("attention change by embedding:", change.round(3).to_dict())
    ec = plm[
        (plm.task == "ec_number") & (plm.pool == "attention") & (plm["head"] == "mlp")
    ]
    print(
        "EC attention MLP:",
        ec.sort_values("value")[["split", "emb", "value"]].round(3).values.tolist(),
    )


def heatmap_rows(plm: pd.DataFrame, noplm: pd.DataFrame) -> None:
    """Rows of the heatmap: MLP, mean pooling, similarity split."""
    keep = (
        (plm["head"] == "mlp")
        & (plm.pool == "no_attention")
        & (plm.split == "similarity")
    )
    grid = plm[keep].pivot(index="task", columns="emb", values="value")
    for task, row in grid.iterrows():
        print(
            f"{task:13s} leader {row.idxmax():22s} {row.max():.3f}, "
            f"lowest {row.min():.3f} ({row.idxmin()})"
        )
    print("distinct leaders:", grid.idxmax(axis=1).nunique())
    keep = (
        (noplm["head"] == "mlp")
        & (noplm.pool == "no_attention")
        & (noplm.split == "similarity")
    )
    reference = noplm[keep].set_index("task").value
    print(
        "rows no-PLM leads:",
        [t for t in grid.index if reference[t] > grid.loc[t].max()],
    )


def prot5_gains(plm: pd.DataFrame) -> None:
    """Best of the 14 PLMs against the best of the two 2020 ProtT5 models."""
    best = plm.groupby(["task", "split"]).value.max()
    old = plm[plm.emb.isin(PROT5_2020)].groupby(["task", "split"]).value.max()
    gain = (best - old).unstack()
    print(gain.round(3).to_string())
    for split in gain.columns:
        g = gain[split]
        print(
            f"{split}: median {g.median():.4f}, max {g.max():.3f} ({g.idxmax()}), "
            f"below 0.02 on {(g < 0.02).sum()}"
        )


def noplm_gaps(plm: pd.DataFrame, noplm: pd.DataFrame) -> None:
    for (task, split), group in plm.groupby(["task", "split"]):
        ref = noplm[(noplm.task == task) & (noplm.split == split)]
        top = ref.loc[ref.value.idxmax()]
        print(
            f"{task:13s} {split:10s} no-PLM {top.value:.3f} ({top['head']}/{top.pool}) "
            f"best PLM {group.value.max():.3f} gap {group.value.max() - top.value:+.3f}"
        )
    for split in ("random", "similarity"):
        binding = plm[(plm.task == "binding") & (plm.split == split)]
        keep = (noplm.task == "binding") & (noplm.split == split)
        linear = noplm[
            keep & (noplm["head"] == "linear") & (noplm.pool == "no_attention")
        ]
        value = linear.value.iloc[0]
        plm_linear = binding[
            (binding["head"] == "linear") & (binding.pool == "no_attention")
        ]
        print(
            f"binding {split}: no-PLM linear {value:.3f}, best PLM linear "
            f"{plm_linear.value.max():.3f}, PLM models below it "
            f"{(binding.value < value).sum()} of {len(binding)}"
        )


def real_case(scores: pd.DataFrame) -> None:
    for task, group in scores[scores.scope == "real_case"].groupby("task"):
        plm = group[group.emb != "noplm"]
        medians = plm.groupby(["head", "pool"]).value.median().round(3).to_dict()
        best_noplm = group[group.emb == "noplm"].value.max()
        print(
            f"{task:30s} best PLM {plm.value.max():.3f}, no-PLM {best_noplm:.3f}, "
            f"gap {plm.value.max() - best_noplm:+.3f} | medians {medians}"
        )


def ec_accuracy() -> None:
    """EC accuracy against the all-negative classifier, from the run metrics."""
    root = _P.results_dir / "predictions"
    rows = []
    for path in root.glob("properties/*/*/*/ec_number/*/metrics.json"):
        pool, split, head, _task, emb = path.parent.relative_to(
            root / "properties"
        ).parts
        test = json.loads(path.read_text())["test"]
        rows.append(
            dict(
                split=split,
                emb=emb,
                head=head,
                pool=pool,
                accuracy=test["accuracy"],
                auprc=test["auprc"],
            )
        )
    ec = pd.DataFrame(rows)
    plm = ec[ec.emb != "noplm"]
    print(
        f"{len(plm)} PLM runs: accuracy {plm.accuracy.min():.5f} to "
        f"{plm.accuracy.max():.5f}, "
        f"AUPRC {plm.auprc.min():.3f} to {plm.auprc.max():.3f}"
    )
    card = json.loads(
        (
            _P.results_dir / "ratings" / "seed_42_two_leaderboards" / "properties.json"
        ).read_text()
    )
    cells = card["rating_exclusions"]["excluded_cells"]
    prevalence = {c["split"]: c["measured"] for c in cells if c["task"] == "ec_number"}
    for split, p in prevalence.items():
        below = plm[(plm.split == split) & (plm.accuracy < 1 - p)]
        print(
            f"{split}: positives {p:.4%}, trivial accuracy {1 - p:.5f}, "
            f"PLM runs below it {len(below)}"
        )


def main() -> None:
    scores = vis.primary_scores(plm_only=False)
    properties = scores[scores.scope == "properties"]
    plm = properties[properties.emb != "noplm"]
    noplm = properties[properties.emb == "noplm"]
    print(f"runs: {scores.groupby('scope').size().to_dict()}, PLMs {plm.emb.nunique()}")
    print("\nSplit penalty, MLP with mean pooling:")
    split_penalty(plm)
    print()
    ablations(plm)
    print("\nHeatmap rows:")
    heatmap_rows(plm, noplm)
    print("\nBest PLM against the 2020 ProtT5 models:")
    prot5_gains(plm)
    print("\nNo-PLM against the PLMs:")
    noplm_gaps(plm, noplm)
    print("\nReal-case datasets:")
    real_case(scores)
    print("\nEC accuracy:")
    ec_accuracy()


if __name__ == "__main__":
    main()
