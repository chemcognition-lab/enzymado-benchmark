"""Render the main-text figures: leaderboard ends, task heatmap and split gap.

Example: python scripts/figures/paper_figures.py
Test run: python scripts/figures/paper_figures.py --out /tmp/figs
"""

import argparse
import itertools
from pathlib import Path

from matplotlib import patches
import numpy as np
import pandas as pd

from enzymado import paths_and_constants, vis

_P = paths_and_constants.Paths
BRACKET_TAG = {"kendall_tau": "Kendall $\\tau$", "auprc": "AUPRC"}  # fonts lack τ
HUE = {"kendall_tau": "brown", "auprc": "green"}


def _bracket(c: vis.Canvas, x: float, y0: float, y1: float, label: str = "") -> None:
    """A bracket from y0 to y1 at x, with its label, if any, to the right."""
    style = dict(color=vis.INK_3, linewidth=1.1, zorder=2)
    c.line([x, x], [y0, y1], **style)
    for edge in (y0, y1):
        c.line([x, x + 5], [edge, edge], **style)
    if label:
        c.text(x + 10, (y0 + y1) / 2, label, "legend", ha="left")


def leaderboard(out: Path, n_each: int = 5) -> None:
    lb = vis.ratings("properties").sort_values("rating", ascending=False)
    lb["rank"] = range(1, len(lb) + 1)
    shown = list(lb.head(n_each).itertuples()) + list(lb.tail(n_each).itertuples())
    lo, hi = vis.padded([r.ci_lo for r in shown] + [r.ci_hi for r in shown], 0.06)
    tag_w, foot = 84, 8  # tag_w holds the top and bottom brackets on the right
    m_left, plot_w = 262, vis.CANVAS_W - 262 - 132 - tag_w
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + (2 * n_each + 1) * vis.ROW_H + foot)
    c.title(f"Property leaderboard: top and bottom {n_each} of {len(lb)} models")
    c.text(
        c.w - 4,
        vis.Y_TITLE,
        "bar = 95% bootstrap interval · dashed = 1500, the mean".upper(),
        "note",
        va="baseline",
    )
    x_rating, x_se = c.w - tag_w - 52, c.w - tag_w - 4
    x_of = c.axis(
        m_left, plot_w, lo, hi, "Bradley–Terry rating", fmt="{:.0f}", foot=foot
    )
    c.line(
        [x_of(1500)] * 2,
        [vis.Y_PLOT, c.h - foot],
        color=vis.INK_3,
        linewidth=1,
        dashes=(3, 3),
        zorder=1,
    )
    c.text(x_rating, vis.Y_TICKS, "rating", "tick", va="baseline")
    c.text(x_se, vis.Y_TICKS, "SE", "tick", va="baseline")
    for slot, r in enumerate(shown[:n_each] + [None] + shown[n_each:]):
        y = vis.Y_PLOT + slot * vis.ROW_H + vis.ROW_H / 2
        if r is None:
            c.line(
                [m_left, m_left + plot_w],
                [y, y],
                color=vis.RULE,
                linewidth=0.8,
                dashes=(2, 3),
                zorder=1,
            )
            c.text(m_left - 12, y, f"{len(lb) - 2 * n_each} omitted", "tick")
            continue
        head = r.model_family + (" + attn" if r.pool_type == "attention" else "")
        c.text(22, y, str(r.rank), "tick")
        c.text(m_left - 12, y, f"{vis.short(r.embedding_model)} / {head}", "row")
        bar = dict(color=vis.INK, linewidth=1.3, alpha=0.30, zorder=2)
        c.line([x_of(r.ci_lo), x_of(r.ci_hi)], [y, y], solid_capstyle="butt", **bar)
        for edge in (r.ci_lo, r.ci_hi):
            c.line([x_of(edge)] * 2, [y - 4, y + 4], **bar)
        c.dot(x_of(r.rating), y, vis.TEAL if slot < n_each else vis.BROWN)
        c.text(x_rating, y, f"{r.rating:.1f}", "value")
        c.text(x_se, y, f"±{r.bootstrap_se:.1f}", "tick")
    for first, label in ((0, f"top {n_each}"), (n_each + 1, f"bottom {n_each}")):
        y0 = vis.Y_PLOT + first * vis.ROW_H + 2
        _bracket(c, c.w - tag_w + 14, y0, y0 + n_each * vis.ROW_H - 4, label)
    c.save(out, "property_leaderboard")


def _ramp_key(c: vis.Canvas, x: float, y0: float, y1: float, metric: str) -> None:
    """A metric's name over its colour ramp, high on top, filling y0 to y1."""
    c.text(x, y0, BRACKET_TAG[metric], "legend", ha="left", va="top")
    n, top, swatch_w = 9, y0 + 24, 18
    step = (y1 - top) / n
    for k in range(n):
        c.ax.add_patch(
            patches.FancyBboxPatch(
                (x, top + k * step),
                swatch_w,
                step - 3,
                boxstyle="round,pad=0,rounding_size=1.5",
                facecolor=vis.SEQ_CMAP[HUE[metric]](1 - k / (n - 1)),
                edgecolor=vis.RULE,
                linewidth=0.4,
                zorder=2,
            )
        )
    c.text(x + swatch_w + 8, top + (step - 3) / 2, "high", "tick", ha="left")
    c.text(x + swatch_w + 8, y1 - (step - 3) / 2 - 3, "low", "tick", ha="left")


def heatmap(out: Path, scores: pd.DataFrame, split: str = "similarity") -> None:
    df = scores.query(
        "scope == 'properties' and head == 'mlp' "
        "and pool == 'no_attention' and split == @split"
    )
    grid = df.pivot(index="task", columns="emb", values="value").loc[
        list(vis.TASK_LABEL)
    ]
    plms = [e for e in grid.columns if e != "noplm"]
    grid = grid[plms + ["noplm"]]  # the reference without a PLM closes each row
    metric = df.groupby("task").metric.first()
    cell_w, cell_h, gap, head_w, top_h, ref_gap = 62, 26, 4, 128, 88, 12
    xs = [
        head_w + j * (cell_w + gap) + ref_gap * (emb == "noplm")
        for j, emb in enumerate(grid.columns)
    ]
    right = xs[-1] + cell_w + gap
    c = vis.Canvas(right + 132, top_h + len(grid) * (cell_h + gap) + 4)
    c.title("Results matrix with primary metrics by task and embedding")
    c.text(
        c.w - 4,
        vis.Y_TITLE,
        f"mlp · no attention · {split} split · outlined = best".upper(),
        "note",
        va="baseline",
    )
    # Names alternate between two rows, so each can be wider than its column; a
    # tick ties the upper row to its cells. The longest name sits on the upper row.
    for j, (x, emb) in enumerate(zip(xs, grid.columns)):
        xc, upper = x + cell_w / 2, j % 2 == 0
        y = top_h - (30 if upper else 7)
        c.text(xc, y, vis.short(emb), "tick", ha="center", va="bottom")
        if upper:
            c.line([xc, xc], [y + 3, top_h - 3], color=vis.INK_3, linewidth=0.8)
    for i, (task, row) in enumerate(grid.iterrows()):
        y = top_h + i * (cell_h + gap)
        c.text(head_w - 10, y + cell_h / 2, vis.TASK_LABEL[task], "label")
        # The colour scale spans the PLMs only, so a weak no-PLM cell does not
        # wash them out; no-PLM takes the end of the scale it falls past.
        lo, hi = row[plms].min(), row[plms].max()
        for x, v in zip(xs, row):
            colour = vis.SEQ_CMAP[HUE[metric[task]]](
                min(max((v - lo) / (hi - lo), 0), 1)
            )
            c.ax.add_patch(
                patches.FancyBboxPatch(
                    (x, y),
                    cell_w,
                    cell_h,
                    boxstyle="round,pad=0,rounding_size=4",
                    facecolor=colour,
                    edgecolor="none",
                    zorder=2,
                )
            )
            if v == row.max():
                c.ax.add_patch(
                    patches.FancyBboxPatch(
                        (x + 1, y + 1),
                        cell_w - 2,
                        cell_h - 2,
                        boxstyle="round,pad=0,rounding_size=3",
                        facecolor="none",
                        edgecolor=vis.INK,
                        linewidth=1.3,
                        zorder=4,
                    )
                )
            dark = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2] <= 0.5
            c.text(
                x + cell_w / 2,
                y + cell_h / 2,
                f"{v:.2f}",
                "value",
                ha="center",
                zorder=3,
                color="#ffffff" if dark else vis.INK,
            )
    gutter = right + 10
    groups = itertools.groupby(enumerate(grid.index), key=lambda it: metric[it[1]])
    for m, members in groups:
        rows = [i for i, _ in members]
        y0 = top_h + rows[0] * (cell_h + gap) + 2
        y1 = top_h + rows[-1] * (cell_h + gap) + cell_h - 2
        _bracket(c, gutter, y0, y1)
        _ramp_key(c, gutter + 14, y0, y1, m)
    c.save(out, "results_matrix")


def split_gap_box(out: Path, scores: pd.DataFrame) -> None:
    """Score lost from the random to the similarity split, one box per task."""
    df = scores.query(
        "scope == 'properties' and head == 'mlp' and pool == 'no_attention'"
    )
    p = df.pivot_table(
        index=["task_type", "task", "emb"], columns="split", values="value"
    )
    drop = p.random - p.similarity
    groups = []
    for task_type, name in [
        ("regression", "Regression, Kendall's $\\tau$"),
        ("classification", "Classification, AUPRC"),
    ]:
        by_task = drop.loc[task_type].groupby("task")
        order = by_task.median().sort_values(ascending=False).index
        groups.append((name, [(t, by_task.get_group(t).to_numpy()) for t in order]))

    # Wide enough for the group headings, which sit in the label column.
    m_left, header_h, row_h = 172, 18, 21
    n_rows = sum(len(rows) for _, rows in groups)
    height = vis.Y_PLOT + n_rows * row_h + len(groups) * header_h + vis.M_BOT
    c = vis.Canvas(vis.CANVAS_W, height)
    c.title("Score lost from the random to the similarity split")

    class Tick:
        """Tick labels with a plain 0 instead of -0.0."""

        @staticmethod
        def format(v: float) -> str:
            v = round(v, 6) + 0.0
            return "0" if v == 0 else f"{v:+.1f}"

    x_of = c.axis(
        m_left,
        c.w - m_left - 96,
        -0.10,
        0.50,
        "Loss in performance, random minus similarity split (higher = larger loss)",
        Tick,
        n=7,
    )
    c.line(
        [x_of(0)] * 2,
        [vis.Y_PLOT, height - vis.M_BOT],
        color=vis.INK_3,
        linewidth=1.0,
        zorder=1.5,
    )
    c.text(c.w - 4, vis.Y_TICKS, "median", "tick", va="baseline")
    rng = np.random.default_rng(0)
    y = vis.Y_PLOT
    for name, rows in groups:
        c.text(4, y + header_h / 2 + 2, name, "label", ha="left", fontweight="bold")
        y += header_h
        for task, v in rows:
            yc = y + row_h / 2
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            fence = 1.5 * (q3 - q1)
            lo, hi = v[v >= q1 - fence].min(), v[v <= q3 + fence].max()
            c.line(
                [x_of(lo), x_of(hi)],
                [yc, yc],
                color=vis.INK,
                linewidth=1.0,
                alpha=0.45,
                zorder=2,
            )
            c.ax.add_patch(
                patches.FancyBboxPatch(
                    (x_of(q1), yc - 7),
                    x_of(q3) - x_of(q1),
                    14,
                    boxstyle="round,pad=0,rounding_size=2",
                    facecolor=vis.RAMP[6],
                    edgecolor="none",
                    alpha=0.55,
                    zorder=2,
                )
            )
            c.line(
                [x_of(med)] * 2,
                [yc - 7, yc + 7],
                color=vis.TEAL,
                linewidth=2.2,
                zorder=3,
                solid_capstyle="butt",
            )
            jitter = rng.uniform(-4.5, 4.5, len(v))
            c.ax.scatter(
                [x_of(a) for a in v],
                yc + jitter,
                s=6,
                color=vis.INK,
                alpha=0.55,
                zorder=4,
                linewidths=0,
            )
            c.text(m_left - 12, yc, vis.TASK_LABEL[task], "row")
            c.text(c.w - 4, yc, f"{med:+.3f}", "value")
            y += row_h
    c.note("MLP head · mean pooling · one dot per PLM · box: IQR · whiskers: 1.5 IQR")
    c.save(out, "split_gap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_P.project_root / "docs" / "figures"
    )
    out = parser.parse_args().out / "results"
    out.mkdir(parents=True, exist_ok=True)
    vis.setup()
    scores = vis.primary_scores(plm_only=False)
    leaderboard(out)
    heatmap(out, scores)
    split_gap_box(out, scores[scores.emb != "noplm"])


if __name__ == "__main__":
    main()
