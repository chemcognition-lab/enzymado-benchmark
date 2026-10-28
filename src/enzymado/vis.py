"""Shared style, data loading and drawing primitives for the paper figures."""

import collections.abc
import json
from pathlib import Path

import matplotlib
from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
FONT_DIR = _P.scripts_dir / "figures" / "fonts"
MONO, SANS = "IBM Plex Mono", "Archivo"

RAMP = [
    "#7a4408",
    "#b0722a",
    "#d6a866",
    "#eed9ad",
    "#ece9e2",
    "#b2ded7",
    "#74c4b9",
    "#2e9a90",
    "#0a6259",
]  # site theme.css --ramp-0..8
TEAL, BROWN = RAMP[8], RAMP[0]
SEQ_CMAP = {
    "green": mcolors.LinearSegmentedColormap.from_list(
        "green", ["#ffffff", RAMP[5], RAMP[6], RAMP[7], RAMP[8]]
    ),
    "brown": mcolors.LinearSegmentedColormap.from_list(
        "brown", ["#ffffff", RAMP[3], RAMP[2], RAMP[1], RAMP[0]]
    ),
}
INK, INK_2, INK_3 = "#131e1d", "#3f4e4d", "#5a6665"
RULE, RULE_2, SURFACE, PAGE = "#b7c0bf", "#c9cfcf", "#dfe2e7", "#ffffff"

TEXT_WIDTH = 5.5  # inches
CANVAS_W = 746  # site pixels
PANEL_RADIUS = 7.2184558 / 213.43906  # card corner, as in the diagram SVGs
PANEL_MARGIN = 2.49 / 605.02
CARD_PAD = 0.1 / TEXT_WIDTH
Y_TITLE, Y_AXIS, Y_TICKS, Y_PLOT = 18, 44, 60, 68
ROW_H, M_BOT = 21, 36

STYLE = {
    "title": dict(fontsize=8.2, color=INK, fontfamily=SANS, fontweight="bold"),
    "axis": dict(fontsize=6.6, color=INK_3, fontfamily=MONO),
    "tick": dict(fontsize=6.2, color=INK_3, fontfamily=MONO),
    "row": dict(fontsize=6.6, color=INK_2, fontfamily=MONO),
    "label": dict(fontsize=6.6, color=INK, fontfamily=SANS),
    "value": dict(fontsize=6.2, color=INK, fontfamily=MONO),
    "legend": dict(fontsize=6.2, color=INK_2, fontfamily=MONO),
    "note": dict(fontsize=5.2, color=INK_3, fontfamily=MONO),
}

SHORT_MODEL = {
    "profluent_e1_300m": "e1_300m",
    "profluent_e1_600m": "e1_600m",
    "prot5_xl_half_uniref50": "prot5_xl_half",
    "prot5_xl_uniref50": "prot5_xl",
    "noplm": "no-PLM",
}
TASK_LABEL = {
    "kcat": "$k_\\mathrm{cat}$",
    "km": "$K_\\mathrm{M}$",
    "ph_opt": "pH$_\\mathrm{opt}$",
    "dtm": "$\\Delta T_\\mathrm{m}$",
    "ddg": "$\\Delta\\Delta G$",
    "binding": "Binding",
    "ec_number": "EC number",
    "active_sites": "Active sites",
    "binding_site": "Binding site",
    "solubility": "Solubility",
}
PRIMARY = {"regression": "kendall_tau", "classification": "auprc"}
Colour = str | tuple[float, ...]


def setup() -> None:
    """Register the site's fonts and set them as the matplotlib defaults."""
    matplotlib.use("Agg")
    for path in sorted(FONT_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": SANS,
            "text.color": INK,
            "figure.dpi": 100,
            "mathtext.fontset": "custom",
            "mathtext.rm": SANS,  # math in the same face
            "mathtext.it": f"{SANS}:italic",
            "mathtext.bf": f"{SANS}:bold",
            "mathtext.default": "regular",
        }
    )


def short(model: str) -> str:
    return SHORT_MODEL.get(model, model)


def primary_scores(*, plm_only: bool = True) -> pd.DataFrame:
    """One row per published run with its primary metric on the test split."""
    root = _P.results_dir / "predictions"
    rows = []
    for f in sorted(root.glob("*/*/*/*/*/*/metrics.json")):
        scope, pool, split, head, task, emb = f.parent.relative_to(root).parts
        test = json.loads(f.read_text())["test"]
        metric = PRIMARY[test["task_type"]]
        rows.append(
            dict(
                scope=scope,
                pool=pool,
                split=split,
                head=head,
                task=task,
                emb=emb,
                task_type=test["task_type"],
                metric=metric,
                value=test[metric],
            )
        )
    df = pd.DataFrame(rows)
    return df[df.emb != "noplm"] if plm_only else df


def ratings(leaderboard: str) -> pd.DataFrame:
    path = (
        _P.results_dir / "ratings" / "seed_42_two_leaderboards" / f"{leaderboard}.csv"
    )
    return pd.read_csv(path).copy()


def padded(values: list[float], frac: float = 0.08) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo - (hi - lo) * frac, hi + (hi - lo) * frac


class Canvas:
    """A figure drawn in site pixels on a rounded grey card, scaled to TEXT_WIDTH."""

    def __init__(self, width: float, height: float) -> None:
        self.w, self.h = width, height
        self.scale = TEXT_WIDTH / width
        self.fig = plt.figure(figsize=(TEXT_WIDTH, height * self.scale), facecolor=PAGE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        pad = CARD_PAD * width
        card_w, card_h = width + 2 * pad, height + 2 * pad
        self.ax.add_patch(
            patches.FancyBboxPatch(
                (-pad, -pad),
                card_w,
                card_h,
                clip_on=False,
                zorder=0,
                boxstyle=f"round,pad=0,rounding_size={PANEL_RADIUS * card_w:f}",
                facecolor=SURFACE,
                edgecolor="none",
            )
        )
        self.ax.axis("off")

    def text(
        self,
        x: float,
        y: float,
        s: str,
        style: str,
        ha: str = "right",
        va: str = "center",
        **kw,
    ) -> plt.Text:
        return self.ax.text(x, y, s, ha=ha, va=va, **STYLE[style] | kw)

    def title(self, s: str) -> None:
        self.text(0, Y_TITLE, s, "title", ha="left", va="baseline")

    def note(self, s: str) -> None:
        self.text(self.w - 4, self.h - 12, s.upper(), "note")

    def line(self, xs: list[float], ys: list[float], **kw) -> None:
        self.ax.plot(xs, ys, **kw)

    def dot(
        self, x: float, y: float, colour: Colour, shape: str = "o", r: float = 4.6
    ) -> None:
        self.ax.plot(
            [x],
            [y],
            shape,
            markersize=r * 2 * self.scale * 72,
            color=colour,
            markeredgecolor=(0, 0, 0, 0.22),
            markeredgewidth=0.5,
            zorder=3,
        )

    def dumbbell(self, x0: float, x1: float, y: float, c0: Colour, c1: Colour) -> None:
        """A grey bar between two values with a dot at each end."""
        self.line(
            [x0, x1],
            [y, y],
            color=INK,
            linewidth=1.3,
            alpha=0.30,
            zorder=2,
            solid_capstyle="round",
        )
        self.dot(x0, y, c0)
        self.dot(x1, y, c1)

    def axis(
        self,
        left: float,
        width: float,
        lo: float,
        hi: float,
        name: str,
        fmt: str = "{:.2f}",
        n: int = 4,
        top: float = 0,
    ) -> collections.abc.Callable[[float], float]:
        """Vertical rules with tick values and the axis name; returns value -> x.

        top moves the name, ticks and rules down, leaving room for a panel heading.
        """

        def x_of(v: float) -> float:
            return left + (v - lo) / (hi - lo) * width

        bottom = self.h - M_BOT
        for v in np.linspace(lo, hi, n):
            self.line(
                [x_of(v)] * 2,
                [Y_PLOT + top, bottom],
                color=RULE_2,
                linewidth=1,
                zorder=1,
            )
            self.text(
                x_of(v),
                Y_TICKS + top,
                fmt.format(v),
                "tick",
                ha="center",
                va="baseline",
            )
        self.text(left, Y_AXIS + top, name, "axis", ha="left", va="baseline")
        return x_of

    def legend(
        self, items: list[tuple[Colour, str]], x: float = 0, gap: float = 22
    ) -> None:
        """One row of (colour, label) dots at the foot of the card."""
        y = self.h - 12
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        for colour, label in items:
            self.dot(x + 4.6, y, colour)
            box = self.text(x + 14, y, label, "legend", ha="left").get_window_extent(
                renderer
            )
            x = inverse.transform((box.x1, 0))[0] + gap

    def save(self, out_dir: Path, name: str) -> None:
        for suffix in ("pdf", "svg", "png"):
            self.fig.savefig(
                out_dir / f"{name}.{suffix}",
                dpi=600,
                bbox_inches="tight",
                facecolor=PAGE,
                pad_inches=PANEL_MARGIN * TEXT_WIDTH,
            )
        plt.close(self.fig)
        print(f"  {name}")
