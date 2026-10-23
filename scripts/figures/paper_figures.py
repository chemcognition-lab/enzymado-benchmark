"""Render the main-text figures: leaderboard ends, task heatmap and split gap.

Example: python scripts/figures/paper_figures.py
Test run: python scripts/figures/paper_figures.py --out /tmp/figs
"""

import argparse
import itertools
from pathlib import Path

from matplotlib import patches

from enzymado.paths_and_constants import Paths
import enzymado.vis as vis

_P = Paths()
METRIC_TAG = {"kendall_tau": "Kendall tau", "auprc": "AUPRC"}
HUE = {"kendall_tau": "brown", "auprc": "green"}


def leaderboard(out: Path, n_each: int = 5) -> None:
    lb = vis.ratings("properties").sort_values("rating", ascending=False)
    lb["rank"] = range(1, len(lb) + 1)
    shown = list(lb.head(n_each).itertuples()) + list(lb.tail(n_each).itertuples())
    lo, hi = vis.padded([r.ci_lo for r in shown] + [r.ci_hi for r in shown], 0.06)
    m_left, plot_w = 262, vis.CANVAS_W - 262 - 132
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + (2 * n_each + 1) * vis.ROW_H + vis.M_BOT)
    c.title(f"Property leaderboard: top and bottom {n_each} of {len(lb)} models")
    x_of = c.axis(m_left, plot_w, lo, hi, "Bradley–Terry rating", fmt="{:.0f}")
    c.line(
        [x_of(1500)] * 2,
        [vis.Y_PLOT, c.h - vis.M_BOT],
        color=vis.INK_3,
        linewidth=1,
        dashes=(3, 3),
        zorder=1,
    )
    c.text(c.w - 52, vis.Y_TICKS, "rating", "tick", va="baseline")
    c.text(c.w - 4, vis.Y_TICKS, "SE", "tick", va="baseline")
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
        c.text(c.w - 52, y, f"{r.rating:.1f}", "value")
        c.text(c.w - 4, y, f"±{r.bootstrap_se:.1f}", "tick")
    c.legend([(vis.TEAL, f"top {n_each}"), (vis.BROWN, f"bottom {n_each}")])
    c.note("bar = 95% bootstrap interval · dashed = 1500, the mean")
    c.save(out, "leaderboard_properties")


def _heatmap_legend(c: vis.Canvas, metrics: list[str], split: str) -> None:
    y, x = c.h - 12, 0
    for metric in metrics:
        tag = METRIC_TAG[metric].upper()
        c.text(x, y, tag, "note", ha="left", color=vis.INK_2)
        x += 9 * len(tag) + 12
        c.text(x, y, "LOW", "note", ha="left")
        for k in range(9):
            c.ax.add_patch(
                patches.FancyBboxPatch(
                    (x + 34 + k * 13, y - 4),
                    11,
                    8,
                    boxstyle="round,pad=0,rounding_size=1.5",
                    facecolor=vis.SEQ_CMAP[HUE[metric]](k / 8),
                    edgecolor=vis.RULE,
                    linewidth=0.4,
                    zorder=2,
                )
            )
        c.text(x + 34 + 9 * 13 + 6, y, "HIGH", "note", ha="left")
        x += 34 + 9 * 13 + 6 + 36 + 34
    c.text(
        c.w, y, f"mlp · mean pooling · {split} split · outlined = best".upper(), "note"
    )


def heatmap(out: Path, scores, split: str = "similarity") -> None:
    df = scores.query(
        "scope == 'properties' and head == 'mlp' "
        "and pool == 'no_attention' and split == @split"
    )
    grid = df.pivot(index="task", columns="emb", values="value").loc[
        list(vis.TASK_LABEL)
    ]
    metric = df.groupby("task").metric.first()
    cell_w, cell_h, gap, head_w, top_h = 62, 32, 4, 96, 136
    c = vis.Canvas(
        head_w + grid.shape[1] * (cell_w + gap) + 132,
        top_h + len(grid) * (cell_h + gap) + 30,
    )
    c.title("Primary metric by task and embedding")
    for j, emb in enumerate(grid.columns):
        c.text(
            head_w + j * (cell_w + gap) + cell_w / 2,
            top_h - 8,
            vis.short(emb),
            "note",
            ha="left",
            va="bottom",
            rotation=45,
            rotation_mode="anchor",
            fontsize=5.6,
        )
    for i, (task, row) in enumerate(grid.iterrows()):
        y = top_h + i * (cell_h + gap)
        c.text(head_w - 10, y + cell_h / 2, vis.TASK_LABEL[task], "label")
        for j, v in enumerate(row):
            x = head_w + j * (cell_w + gap)
            colour = vis.SEQ_CMAP[HUE[metric[task]]](
                (v - row.min()) / (row.max() - row.min())
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
                fontsize=5.8,
                zorder=3,
                color="#ffffff" if dark else vis.INK,
            )
    gutter = head_w + grid.shape[1] * (cell_w + gap) + 10
    groups = itertools.groupby(enumerate(grid.index), key=lambda it: metric[it[1]])
    for m, members in groups:
        rows = [i for i, _ in members]
        y0 = top_h + rows[0] * (cell_h + gap) + 2
        y1 = top_h + rows[-1] * (cell_h + gap) + cell_h - 2
        style = dict(color=vis.INK_3, linewidth=1.1, zorder=2)
        c.line([gutter, gutter], [y0, y1], **style)
        for edge in (y0, y1):
            c.line([gutter, gutter + 5], [edge, edge], **style)
        c.text(
            gutter + 10, (y0 + y1) / 2, METRIC_TAG[m], "legend", ha="left", fontsize=6.4
        )
    _heatmap_legend(c, list(dict.fromkeys(metric[list(vis.TASK_LABEL)])), split)
    c.save(out, "task_heatmap_two_hues")


def split_gap(out: Path, scores) -> None:
    df = scores.query(
        "scope == 'properties' and head == 'mlp' and pool == 'no_attention'"
    )
    mean = df.groupby(["task_type", "emb", "split"]).value.mean().unstack()
    models = mean.loc["regression"].random.sort_values(ascending=False).index
    m_left, panel_w, panel_gap, delta_w = 132, 232, 58, 46
    c = vis.Canvas(vis.CANVAS_W, vis.Y_PLOT + len(models) * vis.ROW_H + vis.M_BOT)
    c.title("Split type: random against similarity")
    c.legend([(vis.TEAL, "random split"), (vis.BROWN, "similarity split")])
    c.note("mlp head · mean pooling · mean over the tasks of each metric")
    for i, emb in enumerate(models):
        c.text(
            m_left - 12,
            vis.Y_PLOT + i * vis.ROW_H + vis.ROW_H / 2,
            vis.short(emb),
            "row",
        )
    for k, (task_type, name) in enumerate(
        [("regression", "Kendall tau"), ("classification", "AUPRC")]
    ):
        left = m_left + k * (panel_w + panel_gap + delta_w)
        pairs = mean.loc[task_type].loc[models]
        x_of = c.axis(
            left,
            panel_w,
            *vis.padded(list(pairs.random) + list(pairs.similarity)),
            name,
        )
        for i, (a, b) in enumerate(zip(pairs.random, pairs.similarity)):
            y = vis.Y_PLOT + i * vis.ROW_H + vis.ROW_H / 2
            c.dumbbell(x_of(a), x_of(b), y, vis.TEAL, vis.BROWN)
            c.text(left + panel_w + delta_w - 4, y, f"{b - a:+.2f}", "value")
    c.save(out, "split_gap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_P.project_root / "docs" / "figures"
    )
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    vis.setup()
    scores = vis.primary_scores()
    leaderboard(out)
    heatmap(out, scores)
    split_gap(out, scores)


if __name__ == "__main__":
    main()
