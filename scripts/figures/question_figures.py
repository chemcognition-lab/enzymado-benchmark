"""Render the appendix figures on release date, model attributes and progress.

Example: python scripts/figures/question_figures.py
Test run: python scripts/figures/question_figures.py --out /tmp/q
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from enzymado import paths_and_constants, vis

_P = paths_and_constants.Paths
RELEASE = pd.to_datetime(
    pd.Series(
        {
            "prot5_xl_uniref50": "2020-07-13",
            "prot5_xl_half_uniref50": "2020-07-13",
            "esm2_650M": "2022-07-21",
            "esm2_3B": "2022-07-21",
            "ankh_base": "2023-01-18",
            "ankh_large": "2023-01-18",
            "prost5": "2023-07-25",
            "esm3_open": "2024-06-25",
            "esmc_300m": "2024-12-04",
            "esmc_600m": "2024-12-04",
            "ankh3_large": "2025-05-26",
            "ankh3_xl": "2025-05-26",
            "profluent_e1_300m": "2025-11-12",
            "profluent_e1_600m": "2025-11-12",
        }
    )
)
GROUPS = {
    "T5 span denoising": [
        "prot5_xl_half_uniref50",
        "prot5_xl_uniref50",
        "ankh_base",
        "ankh_large",
    ],
    "T5 denoising + completion": ["ankh3_xl", "ankh3_large"],
    "Structure in training": ["prost5", "esm3_open"],
    "Masked LM, encoder only": ["esm2_650M", "esm2_3B", "esmc_600m", "esmc_300m"],
    "Masked LM with homologs": ["profluent_e1_300m", "profluent_e1_600m"],
}
NOPLM = {"No PLM, sequence descriptors": ["noplm"]}
NOPLM_DIM = 36  # ProtParam descriptors of the pooled vector
HEADS = {
    "mlp/no_attention": "MLP",
    "xgboost/no_attention": "XGBoost",
    "linear/no_attention": "linear",
    "mlp/attention": "MLP + attn",
    "linear/attention": "linear + attn",
}
M_LEFT = 132
ATTRIBUTES = _P.scripts_dir / "figures" / "plm_attributes.csv"


def _plm_ratings(leaderboard: str) -> pd.DataFrame:
    lb = vis.ratings(leaderboard)
    return lb[lb.embedding_model.isin(RELEASE.index)]


def _row_y(i: int, offset: float = 0) -> float:
    return vis.Y_PLOT + offset + i * vis.ROW_H + vis.ROW_H / 2


def over_time(out: Path) -> None:
    order = sorted(RELEASE.index, key=lambda e: (RELEASE[e], e))
    best = {
        s: _plm_ratings(s).groupby("embedding_model").rating.max()
        for s in ("properties", "real_case")
    }
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + len(order) * vis.ROW_H + vis.M_BOT)
    c.title("Rating of each embedding by release date")
    lo, hi = vis.padded(list(best["properties"]) + list(best["real_case"]))
    x_of = c.axis(M_LEFT, c.w - M_LEFT - 96, lo, hi, "Bradley–Terry rating", "{:.0f}")
    for i, emb in enumerate(order):
        c.text(M_LEFT - 12, _row_y(i), vis.short(emb), "row")
        c.dumbbell(
            x_of(best["properties"][emb]),
            x_of(best["real_case"][emb]),
            _row_y(i),
            vis.TEAL,
            vis.BROWN,
        )
        c.text(c.w - 4, _row_y(i), RELEASE[emb].strftime("%Y-%m"), "value")
    c.text(c.w - 4, vis.Y_TICKS, "release", "tick", va="baseline")
    c.legend([(vis.TEAL, "property leaderboard"), (vis.BROWN, "real-case leaderboard")])
    c.note("best baseline of each embedding · rows by first public release")
    c.save(out, "q2_over_time")


# Display names for the points labelled on the release-date scatter.
LABEL = {
    "prot5_xl_half_uniref50": "ProtT5",
    "prost5": "ProstT5",
    "ankh_large": "Ankh large",
    "esmc_300m": "ESM C 300M",
    "esmc_600m": "ESM C 600M",
    "ankh3_large": "Ankh3 large",
    "ankh3_xl": "Ankh3 XL",
    "profluent_e1_300m": "E1 300M",
    "profluent_e1_600m": "E1 600M",
}
# Which points carry a label on each board, and on which side of the dot.
LABEL_SIDE = {
    "properties": {
        "prost5": "r",
        "prot5_xl_half_uniref50": "r",
        "esmc_600m": "l",
        "esmc_300m": "l",
        "ankh3_large": "r",
        "ankh3_xl": "r",
        "profluent_e1_300m": "l",
        "profluent_e1_600m": "l",
    },
    "real_case": {
        "ankh_large": "r",
        "prot5_xl_half_uniref50": "r",
        "esmc_300m": "l",
        "ankh3_xl": "r",
        "profluent_e1_600m": "l",
    },
}
# PLMs released on the same day are nudged apart so their intervals do not overlap.
DODGE = {
    "prot5_xl_uniref50": -1,
    "prot5_xl_half_uniref50": 1,
    "esm2_650M": -1,
    "esm2_3B": 1,
    "ankh_base": -1,
    "ankh_large": 1,
    "esmc_300m": -1,
    "esmc_600m": 1,
    "ankh3_large": -1,
    "ankh3_xl": 1,
    "profluent_e1_300m": -1,
    "profluent_e1_600m": 1,
}


def over_time_scatter(out: Path) -> None:
    """Rating of each PLM with the MLP head and mean pooling by release date."""
    boards = {
        "properties": "Property leaderboard",
        "real_case": "Real-case leaderboard",
    }
    panel_h, top, left, gap = 300, 84, 64, 76
    c = vis.Canvas(vis.CANVAS_W, top + panel_h + 58)
    c.title("Rating of each PLM by release date")
    panel_w = (c.w - left - gap - 8) / 2
    t0, t1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-07-01")
    for k, (board, name) in enumerate(boards.items()):
        lb = _plm_ratings(board).query(
            "model_family == 'mlp' and pool_type == 'no_attention'"
        )
        d = lb.set_index("embedding_model")[["rating", "ci_lo", "ci_hi"]]
        d["release"] = RELEASE[d.index]
        x0 = left + k * (panel_w + gap)
        lo, hi = vis.padded(list(d.ci_lo) + list(d.ci_hi), 0.06)

        def x_of(t: pd.Timestamp, x0: float = x0) -> float:
            return x0 + (t - t0) / (t1 - t0) * panel_w

        def y_of(v: float, lo: float = lo, hi: float = hi) -> float:
            return top + panel_h - (v - lo) / (hi - lo) * panel_h

        rho = stats.spearmanr(d.release.map(pd.Timestamp.toordinal), d.rating)
        c.text(
            x0, vis.Y_AXIS, name, "label", ha="left", va="baseline", fontweight="bold"
        )
        c.text(
            x0,
            vis.Y_TICKS + 2,
            f"Spearman $\\rho$ = {rho.statistic:+.2f}, p = {rho.pvalue:.2f}",
            "tick",
            ha="left",
            va="baseline",
        )
        for v in np.arange(np.ceil(lo / 50) * 50, hi, 50):
            c.line(
                [x0, x0 + panel_w],
                [y_of(v)] * 2,
                color=vis.RULE_2,
                linewidth=1,
                zorder=1,
            )
            c.text(x0 - 6, y_of(v), f"{v:.0f}", "tick")
        for year in range(2020, 2027):
            xt = x_of(pd.Timestamp(f"{year}-01-01"))
            c.line(
                [xt] * 2,
                [top, top + panel_h],
                color=vis.RULE_2,
                linewidth=0.6,
                alpha=0.6,
                zorder=1,
            )
            c.text(
                xt, top + panel_h + 12, str(year), "tick", ha="center", va="baseline"
            )
        colour = vis.TEAL if k == 0 else vis.BROWN
        for emb, row in d.iterrows():
            x = x_of(row.release) + 4.5 * DODGE.get(emb, 0)
            c.line(
                [x, x],
                [y_of(row.ci_lo), y_of(row.ci_hi)],
                color=vis.INK,
                linewidth=1.0,
                alpha=0.35,
                zorder=2,
            )
            c.dot(x, y_of(row.rating), colour, r=3.8)
            side = LABEL_SIDE[board].get(emb)
            if side:
                c.text(
                    x + (8 if side == "r" else -8),
                    y_of(row.rating),
                    LABEL[emb],
                    "row",
                    ha="left" if side == "r" else "right",
                )
        c.text(
            x0 + panel_w / 2,
            top + panel_h + 26,
            "first public release",
            "axis",
            ha="center",
            va="baseline",
        )
        if k == 0:
            c.text(
                x0 - 50,
                top + panel_h / 2,
                "Bradley–Terry rating",
                "axis",
                ha="center",
                va="center",
                rotation=90,
            )
    c.note("MLP head · mean pooling · bars: 95% bootstrap interval · separate fits")
    c.save(out, "q2_over_time_scatter")


def attributes(out: Path, attr: pd.DataFrame) -> None:
    lb = vis.ratings("properties").groupby("embedding_model").rating
    best, mean = lb.max(), lb.mean()
    groups = GROUPS | NOPLM
    c_par, c_dim, left = 190, 238, 278
    n_rows = sum(len(m) for m in groups.values()) + len(groups)
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + n_rows * vis.ROW_H + vis.M_BOT)
    c.title("Comparing the performance of PLM representations")
    x_of = c.axis(
        left,
        c.w - left - 56,
        *vis.padded(list(best) + list(mean)),
        "Bradley–Terry rating",
        "{:.0f}",
    )
    for x, s in ((c_par, "params"), (c_dim, "dim"), (c.w - 4, "gap")):
        c.text(x, vis.Y_TICKS, s, "tick", va="baseline")
    i = 0
    for group, members in groups.items():
        c.text(16, _row_y(i), group, "label", ha="left")
        for emb in members:
            i += 1
            if emb in attr.index:
                p = attr.params[emb]
                size = f"{p / 1000:.2f}B" if p >= 1000 else f"{p:.0f}M"
                dim = attr.dim[emb]
            else:
                size, dim = "n/a", NOPLM_DIM
            c.text(28, _row_y(i), vis.short(emb), "row", ha="left")
            c.text(c_par, _row_y(i), size, "tick")
            c.text(c_dim, _row_y(i), str(dim), "tick")
            c.dumbbell(x_of(mean[emb]), x_of(best[emb]), _row_y(i), vis.BROWN, vis.TEAL)
            c.text(c.w - 4, _row_y(i), f"{best[emb] - mean[emb]:.0f}", "value")
        i += 1
    c.legend(
        [(vis.TEAL, "best of the 5 baselines"), (vis.BROWN, "mean of the 5 baselines")]
    )
    c.note("property leaderboard · gap = best minus mean")
    c.save(out, "q3_representation_attributes")


def width_by_head(out: Path, attr: pd.DataFrame) -> None:
    lb = _plm_ratings("properties").assign(
        dim=lambda d: d.embedding_model.map(attr.dim)
    )
    lb["head"] = lb.model_family + "/" + lb.pool_type
    narrow = lb[lb.dim <= 1024].groupby("head").rating.mean()
    wide = lb[lb.dim == 2560].groupby("head").rating.mean()
    n_narrow, n_wide = (attr.dim <= 1024).sum(), (attr.dim == 2560).sum()
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + len(HEADS) * vis.ROW_H + 10 + vis.M_BOT)
    c.title("Rating by embedding width and head")
    x_of = c.axis(
        M_LEFT,
        c.w - M_LEFT - 56,
        *vis.padded(list(narrow) + list(wide)),
        "Bradley–Terry rating",
        "{:.0f}",
    )
    for i, (head, name) in enumerate(HEADS.items()):
        y = _row_y(i, 10 if i >= 3 else 0)
        c.text(M_LEFT - 12, y, name, "row")
        c.dumbbell(x_of(narrow[head]), x_of(wide[head]), y, vis.TEAL, vis.BROWN)
        c.text(c.w - 4, y, f"{wide[head] - narrow[head]:+.0f}", "value")
    c.legend(
        [
            (vis.TEAL, f"width 768–1024 ({n_narrow} models)"),
            (vis.BROWN, f"width 2560 ({n_wide} models)"),
        ]
    )
    c.note("property leaderboard · right = wide minus narrow")
    c.save(out, "q3_width_by_head")


def progress(out: Path) -> None:
    df = vis.primary_scores().query("scope == 'properties'")
    first = df.emb.map(RELEASE) <= RELEASE.min()
    best = {
        "2020": df[first].groupby(["split", "task"]).value.max(),
        "today": df.groupby(["split", "task"]).value.max(),
    }
    panel_gap, delta_w = 50, 62
    panel_w = (vis.CANVAS_W - M_LEFT - panel_gap - 2 * delta_w) // 2
    c = vis.Canvas(
        vis.CANVAS_W, vis.Y_PLOT + len(vis.TASK_LABEL) * vis.ROW_H + vis.M_BOT
    )
    c.title("Best score per task in 2020 and today")
    for i, label in enumerate(vis.TASK_LABEL.values()):
        c.text(M_LEFT - 12, _row_y(i), label, "label")
    for k, split in enumerate(("random", "similarity")):
        left = M_LEFT + k * (panel_w + panel_gap + delta_w)
        a, b = best["2020"][split], best["today"][split]
        x_of = c.axis(
            left, panel_w, vis.padded(list(a) + list(b))[0], 1.0, f"{split} split"
        )
        for i, task in enumerate(vis.TASK_LABEL):
            c.dumbbell(x_of(a[task]), x_of(b[task]), _row_y(i), vis.BROWN, vis.TEAL)
            c.text(
                left + panel_w + delta_w - 4,
                _row_y(i),
                f"{b[task] - a[task]:+.2f}",
                "value",
            )
    c.legend([(vis.BROWN, "2020 (ProtT5 only)"), (vis.TEAL, "today (best of 14)")])
    c.note("best baseline · axes end at 1.00 = perfect")
    c.save(out, "q4_done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_P.project_root / "docs" / "figures" / "questions"
    )
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    vis.setup()
    attr = pd.read_csv(ATTRIBUTES, index_col="emb")
    over_time(out)
    over_time_scatter(out)
    attributes(out, attr)
    width_by_head(out, attr)
    progress(out)


if __name__ == "__main__":
    main()
