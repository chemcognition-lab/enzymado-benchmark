#!/usr/bin/env python
"""Paired bootstrap tests of the best PLM against the other 13, per task and split.

Uses the heatmap baseline (MLP head, mean pooling) and the primary metric. The
1,000 stored bootstrap replicates share their resample indices across models, so
the per-replicate difference best - other is a paired bootstrap of the gap. A PLM
cannot be separated from the best when:

    test noise      the 95% percentile interval of the differences contains 0;
    Holm            its two-sided bootstrap p, 2 min(P(d <= 0), P(d >= 0)), Holm-
                    adjusted over the 13 comparisons, is at least 0.05;
    + training      the gap is within 1.96 sqrt(var_boot + 2 sigma^2), sigma being
                    the tie-margin SD of the baseline on that task, split, metric.

Writes the summary per task and split, the detail per pair, and the LaTeX table.

Example:
    python scripts/paper/pairwise_significance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
REGRESSION = ("kcat", "km", "ph_opt", "dtm", "ddg")
TASKS = (
    *REGRESSION,
    "binding",
    "ec_number",
    "active_sites",
    "binding_site",
    "solubility",
)
TASK_TEX = {
    "kcat": r"$k_\text{cat}$",
    "km": r"$K_\text{M}$",
    "ph_opt": r"$\text{pH}_\text{opt}$",
    "dtm": r"$\Delta T_\text{m}$",
    "ddg": r"$\Delta\Delta G$",
    "binding": "binding",
    "ec_number": "EC number",
    "active_sites": "active sites",
    "binding_site": "binding site",
    "solubility": "solubility",
}
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_P.results_dir / "paper")
    return parser.parse_args()


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p)
    stepped = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, stepped)
    return adjusted


def compare(task: str, split: str, sigma: pd.Series) -> tuple[dict, list[dict]]:
    """Test the best PLM of one task and split against each other PLM."""
    metric = "kendall_tau" if task in REGRESSION else "auprc"
    rel = Path("properties", "no_attention", split, "mlp", task)
    runs = sorted(
        p.name
        for p in (_P.results_dir / "predictions" / rel).iterdir()
        if p.name != "noplm"
    )
    point = {
        e: json.loads(
            (_P.results_dir / "predictions" / rel / e / "metrics.json").read_text()
        )["test"][metric]
        for e in runs
    }
    boot = {
        e: np.load(_P.results_dir / "bootstrap" / rel / e / "test.npz")[metric]
        for e in runs
    }
    best = max(point, key=point.get)
    s = sigma[task, split, metric, "mlp", "no_attention"]
    others = [e for e in runs if e != best]
    diffs = [boot[best] - boot[e] for e in others]
    p = np.array([2 * min((d <= 0).mean(), (d >= 0).mean()) for d in diffs])
    p_holm = holm(np.maximum(p, 1 / len(diffs[0])))  # a bootstrap p is at least 1/B
    pairs = []
    for e, d, pe, ph in zip(others, diffs, p, p_holm):
        lo, hi = np.percentile(d, [2.5, 97.5])
        gap = point[best] - point[e]
        half = 1.96 * np.sqrt(d.var(ddof=1) + 2 * s**2)  # same baseline, same sigma
        pairs.append(
            dict(
                task=task, split=split, best=best, other=e, gap=gap, ci_lo=lo,
                ci_hi=hi, p=pe, p_holm=ph, half_with_training=half,
                tied_test=lo <= 0 <= hi, tied_holm=ph >= ALPHA,
                tied_test_train=gap <= half,
            )
        )  # fmt: skip
    df = pd.DataFrame(pairs)
    summary = dict(
        task=task, split=split, metric=metric, best=best,
        best_value=round(point[best], 3), tied_test=int(df.tied_test.sum()),
        tied_holm=int(df.tied_holm.sum()),
        tied_test_train=int(df.tied_test_train.sum()),
        sigma=round(s, 4),
    )  # fmt: skip
    return summary, pairs


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Split} & \textbf{Best PLM} & \textbf{Score} & "
        r"\textbf{Test} & \textbf{Holm} & \textbf{+Train} \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        task = TASK_TEX[row.task] if row.split == "random" else ""
        name = row.best.replace("_", r"\_")
        lines.append(
            f"{task} & {row.split} & \\texttt{{{name}}} & {row.best_value:.3f} & "
            f"{row.tied_test} & {row.tied_holm} & {row.tied_test_train} \\\\"
        )
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    args = parse_args()
    margins = pd.read_csv(_P.results_dir / "noise_seeds" / "tie_margins_v4.csv")
    sigma = margins.set_index(
        ["task", "split", "metric", "model_family", "pool_type"]
    ).sigma
    rows, pairs = [], []
    for task in TASKS:
        for split in ("random", "similarity"):
            summary, detail = compare(task, split, sigma)
            rows.append(summary)
            pairs += detail
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    counts = ("tied_test", "tied_holm", "tied_test_train")
    print(
        "\nmedian PLMs not separable from the best, of 13:",
        {c: float(summary[c].median()) for c in counts},
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "pairwise_significance.csv", index=False)
    pd.DataFrame(pairs).to_csv(
        args.out_dir / "pairwise_significance_pairs.csv", index=False
    )
    (args.out_dir / "pairwise_significance_table.tex").write_text(latex_table(summary))
    print(f"wrote the summary, the pairs and the LaTeX table to {args.out_dir}")


if __name__ == "__main__":
    main()
