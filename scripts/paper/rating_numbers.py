#!/usr/bin/env python
"""Paper numbers from the two leaderboards, their sensitivity runs and win shares.

Reruns compute_ratings.py with the official settings (which must reproduce the
published ratings), with tie margins at z = 0, 0.98 and 3.92, and with equal task
weights. It then prints the leader's interval and bootstrap ranks, the spread of
heads and poolings, the weighted tie share at each z, the per-task win shares, the
release-date and PLM-attribute correlations, and the top of the real-case board.
Real case = case studies (the code says real_case, the docs case studies).

Example:
    python scripts/paper/rating_numbers.py
    python scripts/paper/rating_numbers.py --skip-variants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from enzymado import paths_and_constants

_P = paths_and_constants.Paths
sys.path.insert(0, str(_P.scripts_dir / "results"))
sys.path.insert(0, str(_P.scripts_dir / "figures"))

import compute_ratings  # noqa: E402
import question_figures  # noqa: E402

CARD = _P.project_root / "benchmark_card.json"
PUBLISHED = _P.results_dir / "ratings" / "seed_42_two_leaderboards"
SCOPES = ("properties", "real_case")
Z_VARIANTS = (0.0, 0.98, 3.92)
RECENT = pd.Timestamp("2024-12-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=_P.results_dir / "paper" / "ratings"
    )
    parser.add_argument(
        "--skip-variants",
        action="store_true",
        help="Reuse the variant leaderboards already in --out-dir.",
    )
    return parser.parse_args()


def run_variants(out_dir: Path) -> None:
    card = json.loads(CARD.read_text())
    del card["rating"]["task_weights"]["properties"]
    equal_card = out_dir / "card_equal_tasks.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    equal_card.write_text(json.dumps(card, indent=2))
    variants = {"official": [], "equal_tasks": ["--benchmark-card", str(equal_card)]}
    variants |= {f"z{z}": ["--tie-z", str(z)] for z in Z_VARIANTS}
    script = _P.scripts_dir / "results" / "compute_ratings.py"
    for name, extra in variants.items():
        started = time.time()
        command = [sys.executable, str(script), "--ratings-root", str(out_dir / name)]
        subprocess.run(command + extra, check=True, capture_output=True)
        print(f"variant {name}: {time.time() - started:.0f} s")


def load_cells(scope: str) -> tuple[list[str], list[compute_ratings.Cell]]:
    """Build the rating cells of one leaderboard exactly as compute_ratings does."""
    card = json.loads(CARD.read_text())
    rating = card["rating"]
    predictions, bootstrap = (
        _P.results_dir / "predictions",
        _P.results_dir / "bootstrap",
    )
    match scope:
        case "properties":
            pools = [("attention", "no_attention")]
        case "real_case":
            pools = [("no_attention",), ("attention",)]
        case _:
            raise ValueError(f"Unknown scope {scope}.")
    sources = tuple(
        compute_ratings.Source(predictions, bootstrap, scope, p) for p in pools
    )
    task_types = compute_ratings._task_types(card, scope)
    competitors, cells, _ = compute_ratings._build_cells(
        sources,
        task_types,
        {k: float(v) for k, v in rating["split_weights"][scope].items()},
        1000,
        rating["metric_weights"],
        rating.get("exclusions", {}),
        compute_ratings._class_prevalences(card, scope),
        compute_ratings._load_tie_margins(
            _P.project_root / rating["tie_margins"]["path"]
        ),
        compute_ratings._task_weights(rating, scope, sorted(task_types)),
    )
    return competitors, cells


def tie_share(cells: list[compute_ratings.Cell], z: float) -> float:
    """Share of matches within the margin, weighted like the battles."""
    total = weight = 0.0
    for cell in cells:
        margin = z * np.hypot(cell.sigma[:, None], cell.sigma[None, :])
        gap = cell.scores[:, None, :] - cell.scores[None, :, :]
        tie = (np.abs(gap) <= margin[:, :, None]).mean(axis=2)
        np.fill_diagonal(tie, np.nan)
        total += cell.weight * np.nanmean(tie)
        weight += cell.weight
    return total / weight


def win_shares(
    competitors: list[str], cells: list[compute_ratings.Cell], z: float
) -> pd.DataFrame:
    """Mean match outcome of each competitor per task, weighted by metric weight."""
    rows = []
    for cell in cells:
        margin = z * np.hypot(cell.sigma[:, None], cell.sigma[None, :])
        gap = cell.scores[:, None, :] - cell.scores[None, :, :]
        tie = np.abs(gap) <= margin[:, :, None]
        win = (gap > margin[:, :, None]).mean(axis=2) + 0.5 * tie.mean(axis=2)
        np.fill_diagonal(win, np.nan)
        share = np.nanmean(win, axis=1)
        rows += [
            dict(task=cell.task, competitor=c, share=s, weight=cell.weight)
            for c, s in zip(competitors, share)
        ]
    df = pd.DataFrame(rows)
    df["weighted"] = df.share * df.weight
    by_task = df.groupby(["competitor", "task"])[["weighted", "weight"]].sum()
    return (by_task.weighted / by_task.weight).unstack()


def leaderboard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("rating", ascending=False, ignore_index=True)
    df["rank"] = df.index + 1
    df["cfg"] = df.model_family + "/" + df.pool_type
    return df


def leader(board: pd.DataFrame) -> None:
    top = board.iloc[0]
    print(f"{top.competitor}: {top.rating:.1f} ({top.ci_lo:.1f} to {top.ci_hi:.1f})")
    reach = board[(board["rank"] > 1) & (board.ci_hi >= top.ci_lo)]
    print(
        f"intervals reaching its lower bound: {len(reach)} {reach.competitor.tolist()}"
    )
    boot = np.load(PUBLISHED / "properties_rating_bootstrap.npz", allow_pickle=True)
    names = [str(n) for n in boot["competitors"]]
    ranks = (-boot["bootstrap_ratings"]).argsort(axis=1).argsort(axis=1) + 1
    leader_ranks = ranks[:, names.index(top.competitor)]
    values, counts = np.unique(leader_ranks, return_counts=True)
    print(
        "leader rank over bootstrap rounds:",
        dict(zip(values.tolist(), counts.tolist())),
    )
    print("top 10 configurations:", board.head(10).cfg.value_counts().to_dict())
    print("last 5:", board.tail(5)[["competitor", "rank"]].values.tolist())
    plm = board[board.embedding_model != "noplm"]
    medians = plm.groupby("cfg").rating.median()
    spans = plm.groupby("cfg").rating.agg(lambda s: s.max() - s.min())
    print(
        f"configuration medians {medians.round(1).to_dict()}, "
        f"span {medians.max() - medians.min():.1f}"
    )
    print("PLM span within a configuration:", spans.round(1).to_dict())


def sensitivity(out_dir: Path) -> None:
    for scope in SCOPES:
        official = pd.read_csv(out_dir / "official" / f"{scope}.csv").set_index(
            "competitor"
        )
        published = pd.read_csv(PUBLISHED / f"{scope}.csv").set_index("competitor")
        diff = (official.rating - published.rating.reindex(official.index)).abs().max()
        half = ((official.ci_hi - official.ci_lo) / 2).median()
        print(
            f"[{scope}] official rerun vs published: max |diff| {diff:.2e}; "
            f"range {official.rating.max() - official.rating.min():.1f}, "
            f"median half-width {half:.2f}"
        )
        for name in [f"z{z}" for z in Z_VARIANTS] + ["equal_tasks"]:
            variant = pd.read_csv(out_dir / name / f"{scope}.csv").set_index(
                "competitor"
            )
            variant = variant.reindex(official.index)
            tau = stats.kendalltau(variant.rating, official.rating).statistic
            half = ((variant.ci_hi - variant.ci_lo) / 2).median()
            print(
                f"   {name:12s} tau {tau:.4f} range "
                f"{variant.rating.max() - variant.rating.min():.1f} median half-width "
                f"{half:.2f} top {variant.rating.idxmax()}"
            )
    equal = leaderboard(out_dir / "equal_tasks" / "properties.csv")
    rank = equal.set_index("competitor").loc["prost5/mlp/no_attention", "rank"]
    print(
        "equal weights top 5:",
        equal.head(5).competitor.tolist(),
        f"| ProstT5 MLP rank {rank}",
    )


def release_dates(board: pd.DataFrame, scope: str) -> None:
    plm = board[board.embedding_model != "noplm"]
    best = plm.loc[plm.groupby("embedding_model").rating.idxmax()].set_index(
        "embedding_model"
    )
    best = best.sort_values("rating", ascending=False)
    release = best.index.map(question_figures.RELEASE)
    rho = stats.spearmanr([d.toordinal() for d in release], best.rating)
    recent, older = best[release >= RECENT], best[release < RECENT]
    below = (recent.ci_hi < older.ci_lo.min()).sum()
    ranks = sorted(best.index.get_loc(e) + 1 for e in recent.index)
    print(
        f"[{scope}] rho {rho.statistic:.3f} (p {rho.pvalue:.4f}); recent PLMs at ranks "
        f"{ranks}, {below} with intervals below every older PLM"
    )


def attributes(board: pd.DataFrame) -> None:
    plm = board[board.embedding_model != "noplm"]
    attr = pd.read_csv(question_figures.ATTRIBUTES, index_col="emb").copy()
    attr["best"] = plm.groupby("embedding_model").rating.max()
    attr["mean"] = plm.groupby("embedding_model").rating.mean()
    attr["gap"] = attr.best - attr["mean"]
    for column in ("params", "dim"):
        for target in ("best", "mean"):
            rho = stats.spearmanr(attr[column], attr[target])
            print(
                f"Spearman({column}, {target}) rho {rho.statistic:.3f} "
                f"(p {rho.pvalue:.4f})"
            )
    print(
        attr.sort_values("best", ascending=False)[["dim", "best", "mean", "gap"]]
        .round(1)
        .to_string()
    )
    plm = plm.assign(dim=plm.embedding_model.map(attr.dim))
    narrow = plm[plm.dim <= 1024].groupby("cfg").rating.mean()
    wide = plm[plm.dim == 2560].groupby("cfg").rating.mean()
    print(
        pd.DataFrame({"narrow": narrow, "wide": wide, "wide - narrow": wide - narrow})
        .round(1)
        .to_string()
    )


def real_case_top(board: pd.DataFrame) -> None:
    top = board.head(10)
    print(
        "top 10:",
        top[["competitor", "rating", "ci_lo", "ci_hi"]].round(1).values.tolist(),
    )
    print(f"XGBoost in the top 10: {(top.model_family == 'xgboost').sum()}")
    print("no-PLM ranks:", board[board.embedding_model == "noplm"]["rank"].tolist())


def main() -> None:
    args = parse_args()
    if not args.skip_variants:
        run_variants(args.out_dir)
    properties = leaderboard(PUBLISHED / "properties.csv")
    real_case = leaderboard(PUBLISHED / "real_case.csv")

    print("\nProperty leader:")
    leader(properties)
    print("\nSensitivity to the margin and the task weights:")
    sensitivity(args.out_dir)

    z = json.loads(CARD.read_text())["rating"]["tie_margins"]["z"]
    for scope in SCOPES:
        started = time.time()
        competitors, cells = load_cells(scope)
        shares = {
            zz: f"{100 * tie_share(cells, zz):.3f}%" for zz in sorted({*Z_VARIANTS, z})
        }
        print(
            f"\n[{scope}] weighted tie share by z: {shares} "
            f"({time.time() - started:.0f} s)"
        )
        if scope != "properties":
            continue
        wins = win_shares(competitors, cells, z)
        print(
            "ProstT5 MLP win shares:",
            wins.loc["prost5/mlp/no_attention"].round(3).to_dict(),
        )
        configs = wins.groupby(wins.index.str.split("/", n=1).str[1]).mean()
        print(configs.round(2).to_string())

    xgb = properties[
        (properties.model_family == "xgboost") & (properties.embedding_model != "noplm")
    ]
    mlp = properties[
        (properties.cfg == "mlp/no_attention") & (properties.embedding_model != "noplm")
    ]
    print(
        f"\nmedian rating: XGBoost {xgb.rating.median():.1f}, "
        f"MLP mean pool {mlp.rating.median():.1f}"
    )
    print("\nRelease date against the best baseline of each PLM:")
    release_dates(properties, "properties")
    release_dates(real_case, "real_case")
    print("\nPLM attributes (properties):")
    attributes(properties)
    print("\nReal-case leaderboard:")
    real_case_top(real_case)


if __name__ == "__main__":
    main()
