"""Render the appendix figures on release date, model attributes and progress.

Example: python scripts/figures/question_figures.py
Test run: python scripts/figures/question_figures.py --out /tmp/q
"""

import argparse
from pathlib import Path

import pandas as pd

from enzymado.paths_and_constants import Paths
import enzymado.vis as vis

_P = Paths()
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
HEADS = {
    "mlp/no_attention": "MLP",
    "xgboost/no_attention": "XGBoost",
    "linear/no_attention": "linear",
    "mlp/attention": "MLP + attn",
    "linear/attention": "linear + attn",
}
M_LEFT = 132
ATTR = pd.read_csv(Path(__file__).with_name("plm_attributes.csv"), index_col="emb")


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


def attributes(out: Path) -> None:
    lb = _plm_ratings("properties").groupby("embedding_model").rating
    best, mean = lb.max(), lb.mean()
    c_par, c_dim, left = 190, 238, 278
    n_rows = sum(len(m) for m in GROUPS.values()) + len(GROUPS)
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + n_rows * vis.ROW_H + vis.M_BOT)
    c.title("Rating by training approach, size and width")
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
    for group, members in GROUPS.items():
        c.text(16, _row_y(i), group, "label", ha="left")
        for emb in members:
            i += 1
            p = ATTR.params[emb]
            c.text(28, _row_y(i), vis.short(emb), "row", ha="left")
            c.text(
                c_par,
                _row_y(i),
                f"{p / 1000:.2f}B" if p >= 1000 else f"{p:.0f}M",
                "tick",
            )
            c.text(c_dim, _row_y(i), str(ATTR.dim[emb]), "tick")
            c.dumbbell(x_of(mean[emb]), x_of(best[emb]), _row_y(i), vis.BROWN, vis.TEAL)
            c.text(c.w - 4, _row_y(i), f"{best[emb] - mean[emb]:.0f}", "value")
        i += 1
    c.legend(
        [(vis.TEAL, "best of the 5 baselines"), (vis.BROWN, "mean of the 5 baselines")]
    )
    c.note("property leaderboard · gap = best minus mean")
    c.save(out, "q3_representation_attributes")


def width_by_head(out: Path) -> None:
    lb = _plm_ratings("properties").assign(
        dim=lambda d: d.embedding_model.map(ATTR.dim)
    )
    lb["head"] = lb.model_family + "/" + lb.pool_type
    narrow = lb[lb.dim <= 1024].groupby("head").rating.mean()
    wide = lb[lb.dim == 2560].groupby("head").rating.mean()
    n_narrow, n_wide = (ATTR.dim <= 1024).sum(), (ATTR.dim == 2560).sum()
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
    over_time(out)
    attributes(out)
    width_by_head(out)
    progress(out)


if __name__ == "__main__":
    main()
