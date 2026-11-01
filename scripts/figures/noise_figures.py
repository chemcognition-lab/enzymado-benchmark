"""Render the appendix figure on the noise model behind the tie margin.

Reads the outputs of scripts/paper/shared_sigma.py and scripts/paper/noise_normality.py
from results/paper, so run those first.

Example: python scripts/figures/noise_figures.py
Test run: python scripts/figures/noise_figures.py --out /tmp/noise
"""

import argparse
from pathlib import Path

from matplotlib import patches
import numpy as np
import pandas as pd

from enzymado import paths_and_constants, vis

_P = paths_and_constants.Paths
DATA = _P.results_dir / "paper"
GROUPS = {"14 PLMs": vis.TEAL, "4 PLMs": vis.BROWN}
HEADING = 22  # room for the question above each panel
PAD = 14  # between the tick values and the first rule
PLOT_H = 250
Y0 = vis.Y_PLOT + HEADING + PAD
BIN = 0.05
DECADES = (1, 1e-1, 1e-2, 1e-3, 1e-4)


def _heading(c: vis.Canvas, left: float, s: str) -> None:
    c.text(left, vis.Y_AXIS, s, "label", ha="left", va="baseline", fontweight="bold")


def _rule(c: vis.Canvas, left: float, width: float, y: float, label: str) -> None:
    c.line([left, left + width], [y, y], color=vis.RULE_2, linewidth=1, zorder=1)
    c.text(left - 6, y, label, "tick")


def shared_sigma_panel(c: vis.Canvas, left: float, width: float) -> None:
    """Histogram of the observed spread per cell over the one-sigma simulation."""
    cells = pd.read_csv(DATA / "shared_sigma_cells.csv").copy()
    null = pd.read_csv(DATA / "shared_sigma_null.csv").copy()
    edges = np.round(np.arange(0.2, 1.4 + BIN / 2, BIN), 2)
    counts = np.histogram(cells.observed, edges)[0]
    expected = len(cells) * BIN * null.density
    top = 5 * np.ceil(max(counts.max(), expected.max()) / 5)
    _heading(c, left, "Is the noise the same for every PLM?")
    x_of = c.axis(
        left,
        width,
        edges[0],
        edges[-1],
        "difference in noise between the 14 PLMs",
        "{:.1f}",
        7,
        top=HEADING,
    )

    def y_of(v: float) -> float:
        return Y0 + PLOT_H - v / top * PLOT_H

    for v in np.arange(0, top + 1, 5):
        _rule(c, left, width, y_of(v), f"{v:.0f}")
    c.text(14, Y0 + PLOT_H / 2, "configurations", "axis", ha="center", rotation=90)
    for lo, n in zip(edges[:-1], counts):
        c.ax.add_patch(
            patches.Rectangle(
                (x_of(lo) + 1, y_of(n)),
                x_of(lo + BIN) - x_of(lo) - 2,
                y_of(0) - y_of(n),
                facecolor=vis.RAMP[6],
                edgecolor="none",
                zorder=2,
            )
        )
    shown = (null.lo >= edges[0]) & (null.hi <= edges[-1])
    c.line(
        list(x_of((null.lo[shown] + null.hi[shown]) / 2)),
        list(y_of(expected[shown])),
        color=vis.TEAL,
        linewidth=1.3,
        dashes=(3, 2),
        zorder=3,
    )


def normality_panel(c: vis.Canvas, left: float, width: float) -> None:
    """Share of seed pairs beyond each gap, observed and under normal noise."""
    curve = pd.read_csv(DATA / "noise_normality_curve.csv").query("t <= 5").copy()
    _heading(c, left, "Is the noise normal?")
    x_of = c.axis(
        left,
        width,
        0,
        5,
        "seed pairs farther apart than x noise units",
        "{:.0f}",
        6,
        top=HEADING,
    )

    def y_of(v: float) -> float:
        return Y0 + -np.log10(v) / 4 * PLOT_H

    for v in DECADES:
        _rule(c, left, width, y_of(v), f"{v * 100:g}%")
    c.line(
        [x_of(1.96)] * 2,
        [Y0 - PAD, Y0 + PLOT_H],
        color=vis.INK_3,
        linewidth=1,
        dashes=(3, 3),
        zorder=2,
    )
    c.text(x_of(1.96) + 5, y_of(10**-0.5), "noise threshold", "tick", ha="left")
    for group, colour in GROUPS.items():
        d = curve[curve.group == group]
        for column, style in (("observed", {}), ("normal", dict(dashes=(3, 2)))):
            kept = d[d[column] >= DECADES[-1]]
            c.line(
                list(x_of(kept.t)),
                list(y_of(kept[column])),
                color=colour,
                linewidth=1.3,
                zorder=3,
                **style,
            )


def noise_model(out: Path) -> None:
    c = vis.Canvas(vis.CANVAS_W, Y0 + PLOT_H + vis.M_BOT)
    c.title("How much scores move when only the seed changes")
    shared_sigma_panel(c, 50, 254)
    normality_panel(c, 404, 334)
    c.legend(
        [
            (vis.TEAL, "14 PLMs · mean pool"),
            (vis.BROWN, "4 PLMs · attention, EC, residue"),
        ]
    )
    c.note("bars, solid = real runs · dashed = simulated")
    c.save(out, "noise_model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_P.project_root / "docs" / "figures"
    )
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    vis.setup()
    noise_model(out)


if __name__ == "__main__":
    main()
