#!/usr/bin/env python
"""Export the leaderboards and every published run as plain CSV tables.

Writes leaderboard_properties.csv and leaderboard_real_case.csv (PLM, predictor
head, Elo, bootstrap standard deviation) and all_runs.csv (one row per run with
all its test metrics) from results/ratings and results/predictions.

Example:
    python scripts/results/export_tables.py

Test run:
    python scripts/results/export_tables.py --out /tmp/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado import vis  # noqa: E402

RATINGS = _REPO / "results" / "ratings" / "seed_42_two_leaderboards"
PREDICTIONS = _REPO / "results" / "predictions"
PLM = vis.PLM_LABEL | {"noplm": "no-PLM"}
HEAD = {
    ("mlp", "no_attention"): "MLP",
    ("xgboost", "no_attention"): "XGBoost",
    ("linear", "no_attention"): "Linear",
    ("mlp", "attention"): "MLP + attention",
    ("linear", "attention"): "Linear + attention",
}
METRICS = [
    "kendall_tau",
    "mae",
    "rmse",
    "r2",
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "accuracy",
]


def leaderboard(board: str) -> pd.DataFrame:
    lb = vis.ratings(board)
    boot = np.load(RATINGS / f"{board}_rating_bootstrap.npz", allow_pickle=True)
    sd = pd.Series(boot["bootstrap_ratings"].std(axis=0, ddof=1), boot["competitors"])
    # The published SE is the SD of the bootstrap replicates.
    assert np.allclose(lb.bootstrap_se, sd[lb.competitor], atol=0.05)
    out = pd.DataFrame(
        {
            "PLM": lb.embedding_model.map(PLM),
            "Predictor Head": [HEAD[k] for k in zip(lb.model_family, lb.pool_type)],
            "Elo": lb.rating.round(1),
            "Standard Dev": lb.bootstrap_se.round(1),
        }
    )
    assert out.notna().all().all()
    return out.sort_values("Elo", ascending=False)


def all_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(PREDICTIONS.glob("*/*/*/*/*/*/metrics.json")):
        scope, pool, split, head, task, emb = f.parent.relative_to(PREDICTIONS).parts
        test = json.loads(f.read_text())["test"]
        rows.append(
            {
                "PLM": PLM[emb],
                "Predictor Head": HEAD[head, pool],
                "Task": task,
                "Split": split,
                "Scope": scope,
                "Task Type": test["task_type"],
                "n_test": test["n"],
            }
            | {m: test.get(m) for m in METRICS}
        )
    df = pd.DataFrame(rows)
    assert not df.duplicated(["PLM", "Predictor Head", "Task", "Split", "Scope"]).any()
    df[METRICS] = df[METRICS].round(4)
    return df.sort_values(["Scope", "Task", "Split", "PLM", "Predictor Head"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for board in ("properties", "real_case"):
        lb = leaderboard(board)
        lb.to_csv(out / f"leaderboard_{board}.csv", index=False)
        print(f"leaderboard_{board}.csv: {len(lb)} competitors")
    runs = all_runs()
    runs.to_csv(out / "all_runs.csv", index=False)
    print(f"all_runs.csv: {len(runs)} runs")
    print(runs.groupby(["Scope", "Split"]).size().to_string())


if __name__ == "__main__":
    main()
