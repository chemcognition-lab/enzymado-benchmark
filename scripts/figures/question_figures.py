"""Render the results figures on release date, representation type and head.

Example: python scripts/figures/question_figures.py
Test run: python scripts/figures/question_figures.py --out /tmp/q
"""

import argparse
from collections.abc import Callable
import itertools
from pathlib import Path
import textwrap

from matplotlib import patches
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
    "Masked LM + homologs": ["profluent_e1_300m", "profluent_e1_600m"],
}
NOPLM = {"Sequence descriptors": ["noplm"]}
# Name and marker of each predictor head: shape by head, colour by pooling.
HEADS = {
    "mlp/no_attention": ("MLP", vis.TEAL, "o"),
    "xgboost/no_attention": ("XGBoost", vis.TEAL, "s"),
    "linear/no_attention": ("linear", vis.TEAL, "^"),
    "mlp/attention": ("MLP + attn", vis.BROWN, "o"),
    "linear/attention": ("linear + attn", vis.BROWN, "^"),
}
# Also read by scripts/paper/rating_numbers.py.
ATTRIBUTES = _P.scripts_dir / "figures" / "plm_attributes.csv"


def _plm_ratings(leaderboard: str) -> pd.DataFrame:
    lb = vis.ratings(leaderboard)
    return lb[lb.embedding_model.isin(RELEASE.index)]


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
BOARDS = {"properties": "Property leaderboard", "real_case": "Case studies leaderboard"}
BOARD_NAME = {"properties": "property", "real_case": "case"}  # file name prefix
# Marker of each PLM on the two-board scatter, as (shape, hollow): one shape per
# family, hollow for the smaller (or half-precision) model of a pair.
MARKER = {
    "prot5_xl_uniref50": ("o", False),
    "prot5_xl_half_uniref50": ("o", True),
    "esm2_650M": ("s", True),
    "esm2_3B": ("s", False),
    "ankh_base": ("^", True),
    "ankh_large": ("^", False),
    "prost5": ("D", False),
    "esm3_open": ("v", False),
    "esmc_300m": ("X", True),
    "esmc_600m": ("X", False),
    "ankh3_large": ("P", True),
    "ankh3_xl": ("P", False),
    "profluent_e1_300m": ("*", True),
    "profluent_e1_600m": ("*", False),
}
# Radius per marker shape so they read as the same size.
SHAPE_R = {"^": 4.8, "v": 4.8, "s": 3.8, "D": 3.6, "o": 4.2, "X": 4.4, "P": 4.8, "*": 6}


def _best_ratings(board: str) -> pd.DataFrame:
    """Rating and interval of the best predictor head of each PLM, as in the paper."""
    lb = _plm_ratings(board)
    best = lb.loc[lb.groupby("embedding_model").rating.idxmax()]
    d = best.set_index("embedding_model")[["rating", "ci_lo", "ci_hi"]]
    d["release"] = RELEASE[d.index]
    return d


def _overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap area of each rect in a with each rect in b, rects as x0, y0, x1, y1."""
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[:, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[:, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def _hits(segs: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Whether each segment passes through each rect (Liang-Barsky); NaN rows miss."""
    x0, y0 = segs[:, :1], segs[:, 1:2]
    dx, dy = segs[:, 2:3] - x0, segs[:, 3:4] - y0
    lo, hi = np.zeros((len(segs), len(rects))), np.ones((len(segs), len(rects)))
    miss = np.isnan(lo + x0)
    for p, q in (
        (-dx, x0 - rects[:, 0]),
        (dx, rects[:, 2] - x0),
        (-dy, y0 - rects[:, 1]),
        (dy, rects[:, 3] - y0),
    ):
        p = np.broadcast_to(p, lo.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = q / p
        miss |= (p == 0) & (q < 0)
        lo = np.where(p < 0, np.maximum(lo, t), lo)
        hi = np.where(p > 0, np.minimum(hi, t), hi)
    return ~miss & (lo <= hi)


def _crossings(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether each segment in a crosses each segment in b; NaN rows never do."""

    def turn(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    a0, a1 = (a[:, None, 0], a[:, None, 1]), (a[:, None, 2], a[:, None, 3])
    b0, b1 = (b[:, 0], b[:, 1]), (b[:, 2], b[:, 3])
    return (turn(a0, a1, b0) * turn(a0, a1, b1) < 0) & (
        turn(b0, b1, a0) * turn(b0, b1, a1) < 0
    )


# Label offsets from the dot tried by _place_labels, in site pixels.
LABEL_DX = (8, -8, 20, -20, 36, -36, 52, -52)
LABEL_DY = (0, *(k * d for d in range(11, 89, 11) for k in (-1, 1)))
# Costs of a label layout. A label covering anything, or a leader through a label,
# is all but forbidden; a leader may cross a bar or another leader at a price.
COVER, THROUGH_DOT, THROUGH_BAR, CROSS = 1000, 300, 150, 300


def _place_labels(
    c: vis.Canvas,
    points: list[tuple[str, float, float, int]],
    dots: list[tuple],
    bars: list[tuple],
    bounds: tuple,
    restarts: int = 200,
) -> None:
    """Put each label next to its dot where it covers no dot, bar or other label.

    Each label may sit left or right of its dot at growing offsets; one moved off
    the spot beside the dot gets a thin leader to its near corner. A layout is
    scored as a whole (see COVER and the costs after it), plus the leader length
    and a small price for the side away from the dot's own (left for the left dot
    of a dodged pair). Coordinate descent from seeded random starts keeps the
    cheapest layout.
    """
    renderer = c.fig.canvas.get_renderer()
    inverse = c.ax.transData.inverted()
    style = dict(
        zorder=4,
        bbox=dict(facecolor=vis.SURFACE, edgecolor="none", pad=0.6),
    )
    owner, rects, segs, cands, extra = [], [], [], [], []
    for i, (name, x, y, side) in enumerate(points):
        box = c.text(x, y, name, "row", ha="left", **style).get_window_extent(renderer)
        c.ax.texts[-1].remove()
        (x0, y1), (x1, y0) = inverse.transform([(box.x0, box.y0), (box.x1, box.y1)])
        w, h = x1 - x0, y1 - y0
        for dy in LABEL_DY:
            for dx in LABEL_DX:
                left = x + dx if dx > 0 else x + dx - w
                rects.append(
                    (left - 1, y + dy - h / 2 - 1, left + w + 1, y + dy + h / 2 + 1)
                )
                end = (x + dx - np.sign(dx) * 1.6, y + dy - np.sign(dy) * h / 2)
                moved = dy or abs(dx) > 8
                segs.append((x, y, *end) if moved else (np.nan,) * 4)
                owner.append(i)
                cands.append((name, x + dx, y + dy, "left" if dx > 0 else "right"))
                away = (dx < 0) != (side < 0)
                extra.append(0.8 * np.hypot(abs(dx) - 8, dy) + 5 * away)
    owner, rects, segs = np.array(owner), np.array(rects), np.array(segs)
    dots, bars = np.array(dots), np.array(bars)
    inside = (
        (rects[:, 0] >= bounds[0])
        & (rects[:, 1] >= bounds[1])
        & (rects[:, 2] <= bounds[2])
        & (rects[:, 3] <= bounds[3])
    )
    # A leader starts on its own dot and bar: the rects that hold the dot itself.
    xy = np.array([p[1:3] for p in points])[owner]
    own = {
        k: (r[:, 0] <= xy[:, :1])
        & (xy[:, :1] <= r[:, 2])
        & (r[:, 1] <= xy[:, 1:])
        & (xy[:, 1:] <= r[:, 3])
        for k, r in (("dots", dots), ("bars", bars))
    }
    cover = _overlap(rects, np.vstack([dots, bars]))
    static = (
        np.array(extra)
        + 1e6 * ~inside
        + ((cover > 0) * (COVER + cover)).sum(1)
        + THROUGH_DOT * (_hits(segs, dots) & ~own["dots"]).sum(1)
        + THROUGH_BAR * (_hits(segs, bars) & ~own["bars"]).sum(1)
    )
    cover = _overlap(rects, rects)
    through = _hits(segs, rects)
    pair = (
        (cover > 0) * (COVER + cover)
        + COVER * (through | through.T)
        + CROSS * _crossings(segs, segs)
    )
    pair[owner[:, None] == owner] = 0
    options = [np.flatnonzero(owner == i) for i in range(len(points))]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        pick = np.array([rng.choice(o) for o in options])
        changed = True
        while changed:
            changed = False
            for i in rng.permutation(len(points)):
                rest = np.delete(pick, i)
                cost = static[options[i]] + pair[options[i]][:, rest].sum(1)
                k = options[i][np.argmin(cost)]
                changed |= k != pick[i]
                pick[i] = k
        total = static[pick].sum() + pair[np.ix_(pick, pick)].sum() / 2
        if best is None or total < best[0]:
            best = (total, pick)
    for k in best[1]:
        name, tx, ty, ha = cands[k]
        c.text(tx, ty, name, "row", ha=ha, **style)
        if not np.isnan(segs[k, 0]):
            c.line(
                segs[k, ::2], segs[k, 1::2], color=vis.INK_3, linewidth=0.6, zorder=2.5
            )


IDEAL_DASH = (0, (4, 3))


def _ideal_key(c: vis.Canvas) -> None:
    """Legend entry for the dashed reference line, at the foot of the card."""
    y = c.h - 12
    c.line([0, 22], [y, y], color=vis.INK_3, linewidth=1.2, linestyle=IDEAL_DASH)
    c.text(30, y, "ideal: each newer PLM rates higher", "legend", ha="left")


def _head_key(
    c: vis.Canvas, top: float, panel_h: float, key_w: float, row_h: float = 18
) -> None:
    """Key of the predictor heads down the right edge, centred on the panel."""
    x_key = c.w - key_w
    y = top + (panel_h - len(HEADS) * row_h) / 2 + row_h / 2
    for name, colour, shape in HEADS.values():
        c.dot(x_key + 6, y, colour, shape, r=SHAPE_R[shape])
        c.text(x_key + 16, y, name, "legend", ha="left")
        y += row_h


def _release_panel(
    c: vis.Canvas,
    board: str,
    x0: float,
    top: float,
    panel_w: float,
    panel_h: float,
    shown: dict[str, str],
    colour: vis.Colour,
    *,
    heading: bool = True,
    ideal: bool = True,
    marker: dict[str, tuple[str, bool]] | None = None,
) -> None:
    """One board: rating by release date, against the path of a field that improves.

    shown names the PLMs labelled beside their dot; marker draws each PLM with its
    own (shape, hollow) instead, for a figure that names them in a legend.
    """
    d = _best_ratings(board)
    # Labels need room past the newest release, for the 2025 PLMs.
    t0 = pd.Timestamp("2019-09-01")
    t1 = pd.Timestamp("2027-04-01" if shown else "2026-07-01")
    lo, hi = vis.padded(list(d.ci_lo) + list(d.ci_hi), 0.06)

    def x_of(t: pd.Timestamp) -> float:
        return x0 + (t - t0) / (t1 - t0) * panel_w

    def y_of(v: float) -> float:
        return top + panel_h - (v - lo) / (hi - lo) * panel_h

    rho = stats.spearmanr(d.release.map(pd.Timestamp.toordinal), d.rating)
    # The heading and the correlation share one line, the correlation after the name.
    x_rho = x0
    if heading:
        name = c.text(
            x0,
            top - 12,
            BOARDS[board],
            "label",
            ha="left",
            va="baseline",
            fontweight="bold",
        )
        box = name.get_window_extent(c.fig.canvas.get_renderer())
        x_rho = c.ax.transData.inverted().transform((box.x1, 0))[0] + 10
    c.text(
        x_rho,
        top - 12,
        f"Spearman $\\rho$ = {rho.statistic:+.2f}",
        "tick",
        ha="left",
        va="baseline",
    )
    for v in np.arange(np.ceil(lo / 50) * 50, hi, 50):
        c.line(
            [x0, x0 + panel_w], [y_of(v)] * 2, color=vis.RULE_2, linewidth=1, zorder=1
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
        c.text(xt, top + panel_h + 14, str(year), "tick", ha="center", va="baseline")
    c.text(
        x0 + panel_w / 2,
        top + panel_h + 30,
        "release date",
        "axis",
        ha="center",
        va="baseline",
    )
    # Reference, not a fit: from the oldest release at the bottom of the axis to the
    # newest at the top, the path of a field where every newer PLM rates higher.
    if ideal:
        xa, ya = x_of(d.release.min()), y_of(lo)
        xb, yb = x_of(d.release.max()), y_of(hi)
        c.line([xa, xb], [ya, yb], color=vis.INK_3, linewidth=1.2, linestyle=IDEAL_DASH)
    dots, bars, points = [], [], []
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
        if marker:
            shape, hollow = marker[emb]
            c.dot(x, y_of(row.rating), colour, shape, SHAPE_R[shape], hollow=hollow)
        else:
            c.dot(x, y_of(row.rating), colour, r=3.8)
        dots.append((x - 5, y_of(row.rating) - 5, x + 5, y_of(row.rating) + 5))
        bars.append((x - 1.5, y_of(row.ci_hi), x + 1.5, y_of(row.ci_lo)))
        if emb in shown:
            points.append((shown[emb], x, y_of(row.rating), DODGE.get(emb, 0)))
    if points:
        bounds = (x0, top, x0 + panel_w, top + panel_h)
        _place_labels(c, points, dots, bars, bounds)


# PLMs in release order: down the marker key, and left to right within each box.
PLM_ORDER = sorted(MARKER, key=lambda e: (RELEASE[e], DODGE.get(e, 0)))


def _plm_key(
    c: vis.Canvas,
    top: float,
    panel_h: float,
    key_w: float,
    *,
    noplm: bool = False,
    row_h: float = 16,
) -> None:
    """Key of the PLM markers down the right edge, centred on the panels."""
    x_key = c.w - key_w
    y = top + (panel_h - (len(PLM_ORDER) + noplm) * row_h) / 2 + row_h / 2
    for emb in PLM_ORDER:
        shape, hollow = MARKER[emb]
        c.dot(x_key + 6, y, vis.INK_3, shape, SHAPE_R[shape], hollow=hollow)
        c.text(x_key + 16, y, vis.PLM_LABEL[emb], "legend", ha="left")
        y += row_h
    if noplm:
        _dashes(c, x_key, x_key + 12, y)
        c.text(x_key + 16, y, "no PLM", "legend", ha="left")


def _dashes(c: vis.Canvas, x0: float, x1: float, y: float) -> None:
    """A dashed rule of whole dashes from x0 to x1, so no stub of a dash ends it.

    A matplotlib dash pattern rarely fits the length a whole number of times.
    """
    n = max(2, round((x1 - x0) / 9))
    dash = (x1 - x0) / (n + 0.6 * (n - 1))
    xs = [
        x
        for i in range(n)
        for x in (x0 + 1.6 * i * dash, x0 + 1.6 * i * dash + dash, np.nan)
    ]
    c.line(xs, [y] * len(xs), color=vis.INK, linewidth=1.2, solid_capstyle="butt")


def _box(
    c: vis.Canvas, x: float, v: pd.Series, y_of: Callable[[float], float], hw: float
) -> None:
    """A box of quartiles with the median in ink and whiskers to the extremes."""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    c.line([x, x], [y_of(v.min()), y_of(v.max())], color=vis.INK_3, linewidth=1)
    c.ax.add_patch(
        patches.Rectangle(
            (x - hw, y_of(q3)),
            2 * hw,
            y_of(q1) - y_of(q3),
            facecolor=vis.SURFACE,
            edgecolor=vis.INK_3,
            linewidth=1,
            zorder=2,
        )
    )
    c.line([x - hw, x + hw], [y_of(med)] * 2, color=vis.INK, linewidth=1.4)


def over_time_scatter(out: Path) -> None:
    """Rating of the best predictor head of each PLM by release date.

    Each PLM has its own marker, named in a key down the right edge.
    """
    panel_h, top, left, gap, key_w = 206, 52, 64, 56, 116
    c = vis.Canvas(vis.CANVAS_W, top + panel_h + 36)
    c.title("Rating of each PLM by release date")
    panel_w = (c.w - left - 2 * gap - key_w) / 2
    for k, board in enumerate(BOARDS):
        colour = vis.TEAL if k == 0 else vis.BROWN
        x0 = left + k * (panel_w + gap)
        _release_panel(
            c, board, x0, top, panel_w, panel_h, {}, colour, ideal=False, marker=MARKER
        )
    c.text(
        left - 50,
        top + panel_h / 2,
        "Bradley–Terry rating",
        "axis",
        ha="center",
        va="center",
        rotation=90,
    )
    _plm_key(c, top, panel_h, key_w, row_h=14)
    c.text(
        c.w - 4,
        vis.Y_TITLE,
        "best predictor head of each PLM · bars: 95% interval".upper(),
        "note",
        va="baseline",
    )
    c.save(out, "release_rating")


def over_time_board(out: Path, board: str) -> None:
    """One board at full width, every PLM labelled."""
    panel_h, top, left = 420, 56, 64
    c = vis.Canvas(vis.CANVAS_W, top + panel_h + 66)
    c.title(f"Rating of each PLM by release date: {BOARDS[board].lower()}")
    colour = vis.TEAL if board == "properties" else vis.BROWN
    _release_panel(
        c,
        board,
        left,
        top,
        c.w - left - 8,
        panel_h,
        vis.PLM_LABEL,
        colour,
        heading=False,
    )
    c.text(
        left - 50,
        top + panel_h / 2,
        "Bradley–Terry rating",
        "axis",
        ha="center",
        va="center",
        rotation=90,
    )
    _ideal_key(c)
    c.note("best predictor head of each PLM · bars: 95% interval")
    c.save(out, f"{BOARD_NAME[board]}_release_rating")


def attributes(out: Path) -> None:
    """Rating of the 5 predictor heads on each representation, grouped by kind."""
    lb = vis.ratings("properties")
    rating = lb.set_index(
        ["embedding_model", lb.model_family + "/" + lb.pool_type]
    ).rating
    groups = GROUPS | NOPLM
    top, panel_h, left, gap, names_h, key_w = 36, 200, 64, 12, 54, 108
    c = vis.Canvas(vis.CANVAS_W, top + panel_h + names_h + 40)
    right = c.w - key_w - 10
    c.title("Comparing the performance of PLM representations")
    c.text(
        c.w - 4,
        vis.Y_TITLE,
        "property leaderboard · one point per predictor head".upper(),
        "note",
        va="baseline",
    )
    n = sum(len(m) for m in groups.values())
    slot = (right - left - gap * (len(groups) - 1)) / n
    lo, hi = vis.padded(list(rating), 0.06)

    def y_of(v: float) -> float:
        return top + panel_h - (v - lo) / (hi - lo) * panel_h

    for v in np.arange(np.ceil(lo / 50) * 50, hi, 50):
        c.line([left, right], [y_of(v)] * 2, color=vis.RULE_2, linewidth=1, zorder=1)
        c.text(left - 6, y_of(v), f"{v:.0f}", "tick")
    c.text(
        left - 50,
        top + panel_h / 2,
        "Bradley–Terry rating",
        "axis",
        ha="center",
        va="center",
        rotation=90,
    )
    renderer = c.fig.canvas.get_renderer()
    inverse = c.ax.transData.inverted()
    y_axis, y_bracket = top + panel_h + 4, top + panel_h + names_h
    # Far and near row of the names. The longest name goes on the far row, where
    # no tick crosses it.
    name_rows = itertools.cycle((34, 16))
    x0 = left
    for group, members in groups.items():
        for k, emb in enumerate(members):
            x = x0 + (k + 0.5) * slot
            v = rating[emb]
            _box(c, x, v, y_of, 13)
            for j, (head, (_, colour, shape)) in enumerate(HEADS.items()):
                c.dot(
                    x + (j - 2) * 5,
                    y_of(v[head]),
                    colour,
                    shape,
                    r=0.8 * SHAPE_R[shape],
                )
            # Names alternate between two rows, so each can be wider than its slot;
            # a tick ties the box to its name.
            y_name = y_axis + next(name_rows)
            c.line([x, x], [y_axis, y_name - 8], color=vis.INK_3, linewidth=0.8)
            c.text(x, y_name, vis.short(emb), "tick", ha="center", color=vis.INK_2)
        x1 = x0 + len(members) * slot
        c.line([x0 + 3, x1 - 3], [y_axis] * 2, color=vis.INK_3, linewidth=0.8)
        # A bracket open to the top spans the representations of one kind.
        c.line(
            [x0 + 3, x0 + 3, x1 - 3, x1 - 3],
            [y_bracket - 5, y_bracket, y_bracket, y_bracket - 5],
            color=vis.INK_3,
            linewidth=0.8,
        )
        # Wrap the kind onto more lines until it fits under its bracket.
        for width in (len(group), 16, 12, 9):
            t = c.text(
                (x0 + x1) / 2,
                y_bracket + 6,
                textwrap.fill(group, width, break_long_words=False),
                "label",
                ha="center",
                va="top",
            )
            box = t.get_window_extent(renderer)
            (bx0, _), (bx1, _) = inverse.transform([(box.x0, 0), (box.x1, 0)])
            if bx1 - bx0 <= x1 - x0 + gap - 4 or width == 9:
                break
            t.remove()
        x0 = x1 + gap
    _head_key(c, top, panel_h, key_w)
    c.save(out, "plm_attributes")


def _head_panel(
    c: vis.Canvas,
    board: str,
    x0: float,
    top: float,
    panel_w: float,
    panel_h: float,
    colour: vis.Colour,
) -> None:
    """One board: a box per predictor head, a point per PLM, a dash for no PLM.

    Each PLM keeps its marker and its place in every box, so it can be followed
    from head to head.
    """
    lb = vis.ratings(board)
    rating = lb.set_index([lb.model_family + "/" + lb.pool_type, "embedding_model"])
    rating = rating.rating
    lo, hi = vis.padded(list(rating), 0.06)
    slot = panel_w / len(HEADS)
    hw = 0.42 * slot
    step = 2 * (hw - 4) / (len(PLM_ORDER) - 1)

    def y_of(v: float) -> float:
        return top + panel_h - (v - lo) / (hi - lo) * panel_h

    for v in np.arange(np.ceil(lo / 100) * 100, hi, 100):
        c.line(
            [x0, x0 + panel_w], [y_of(v)] * 2, color=vis.RULE_2, linewidth=1, zorder=1
        )
        c.text(x0 - 6, y_of(v), f"{v:.0f}", "tick")
    # The board's name runs up the left edge; the note names the rating axis.
    c.text(
        x0 - 46,
        top + panel_h / 2,
        BOARDS[board],
        "label",
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
    )
    for j, (head, (name, _, _)) in enumerate(HEADS.items()):
        x = x0 + (j + 0.5) * slot
        _box(c, x, rating[head].drop("noplm"), y_of, hw)
        _dashes(c, x - hw, x + hw, y_of(rating[head, "noplm"]))
        for i, emb in enumerate(PLM_ORDER):
            shape, hollow = MARKER[emb]
            c.dot(
                x - hw + 4 + i * step,
                y_of(rating[head, emb]),
                colour,
                shape,
                0.85 * SHAPE_R[shape],
                hollow=hollow,
            )
        c.text(x, top + panel_h + 14, name, "row", ha="center", va="baseline")


HEAD_NOTE = (
    "y: Bradley–Terry rating · one point per PLM · box: quartiles · whiskers: range"
)
MERGED_NOTE = (
    "y: Bradley–Terry rating · box: quartiles · whiskers: range · bars: 95% interval"
)


def _linear_gap(board: str) -> pd.DataFrame:
    """Rating of the best head of each PLM minus its linear probe, with 95% interval.

    The gap is taken inside every bootstrap replicate, so the two ratings stay paired.
    """
    z = np.load(
        _P.results_dir
        / "ratings"
        / "seed_42_two_leaderboards"
        / f"{board}_rating_bootstrap.npz",
        allow_pickle=True,
    )
    point = pd.Series(z["point_ratings"], index=z["competitors"])
    boot = pd.DataFrame(z["bootstrap_ratings"], columns=z["competitors"])
    rows = {}
    for emb in [*PLM_ORDER, "noplm"]:
        best = point[point.index.str.startswith(f"{emb}/")].idxmax()
        linear = f"{emb}/linear/no_attention"
        gap = boot[best] - boot[linear]
        rows[emb] = (point[best] - point[linear], *np.percentile(gap, [2.5, 97.5]))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["gap", "lo", "hi"])


def _gain_panel(
    c: vis.Canvas,
    g: pd.DataFrame,
    span: tuple[float, float],
    x0: float,
    top: float,
    panel_w: float,
    panel_h: float,
    colour: vis.Colour,
) -> None:
    """One board: the rating the best head of each PLM gains over its linear probe.

    span is the rating range, shared by both boards so their gains compare.
    """
    lo, hi = span
    x, hw = x0 + panel_w / 2, panel_w / 2 - 6
    step = 2 * (hw - 4) / (len(PLM_ORDER) - 1)

    def y_of(v: float) -> float:
        return top + panel_h - (v - lo) / (hi - lo) * panel_h

    for v in np.arange(0, hi, 100):
        rule = vis.INK_3 if v == 0 else vis.RULE_2  # 0 is the linear probe itself
        c.line([x0, x0 + panel_w], [y_of(v)] * 2, color=rule, linewidth=1, zorder=1)
        c.text(x0 - 6, y_of(v), f"{v:.0f}", "tick")
    _box(c, x, g.gap.drop("noplm"), y_of, hw)
    _dashes(c, x - hw, x + hw, y_of(g.gap["noplm"]))
    for i, emb in enumerate(PLM_ORDER):
        xi = x - hw + 4 + i * step
        c.line(
            [xi, xi],
            [y_of(g.lo[emb]), y_of(g.hi[emb])],
            color=colour,
            linewidth=1,
            alpha=0.5,
            zorder=2.5,
        )
        shape, hollow = MARKER[emb]
        c.dot(
            xi,
            y_of(g.gap[emb]),
            colour,
            shape,
            0.85 * SHAPE_R[shape],
            hollow=hollow,
        )
    c.text(
        x, top + panel_h + 14, "best \u2212 linear", "row", ha="center", va="baseline"
    )


def head_boxes(out: Path) -> None:
    """Rating of each predictor head across the 14 PLMs, and its gain over linear.

    One row per board: the five heads, then what the best head of each PLM adds to
    its linear probe, on an axis of its own.
    """
    panel_h, top, left, gap, key_w = 150, 48, 64, 30, 116
    gain_w, gain_pad = 120, 44
    c = vis.Canvas(vis.CANVAS_W, top + 2 * panel_h + gap + 26)
    c.title("Rating of each predictor head across the 14 PLMs")
    c.text(c.w - 4, vis.Y_TITLE, MERGED_NOTE.upper(), "note", va="baseline")
    panel_w = c.w - left - 14 - key_w - gain_w - gain_pad
    gains = {board: _linear_gap(board) for board in BOARDS}
    # 0, the linear probe itself, stays on the axis as the reference.
    span = vis.padded([0, *(v for g in gains.values() for v in (*g.lo, *g.hi))], 0.04)
    x_gain = left + panel_w + gain_pad
    style = dict(ha="left", va="baseline", fontweight="bold")
    c.text(left, top - 12, "a) Rating of each head", "label", **style)
    c.text(x_gain, top - 12, "b) Gain over linear", "label", **style)
    # A rule sets the gains, on their own axis, apart from the ratings.
    c.line([x_gain - 36] * 2, [top - 22, c.h - 6], color=vis.INK_3, linewidth=0.8)
    for k, board in enumerate(BOARDS):
        colour = vis.TEAL if k == 0 else vis.BROWN
        y = top + k * (panel_h + gap)
        _head_panel(c, board, left, y, panel_w, panel_h, colour)
        _gain_panel(c, gains[board], span, x_gain, y, gain_w, panel_h, colour)
    _plm_key(c, top, 2 * panel_h + gap, key_w, noplm=True)
    c.save(out, "head_rating")


def head_boxes_board(out: Path, board: str) -> None:
    """Rating of each predictor head across the 14 PLMs on one board."""
    panel_h, top, left, key_w = 340, 34, 64, 116
    c = vis.Canvas(vis.CANVAS_W, top + panel_h + 26)
    c.title("Rating of each predictor head across the 14 PLMs")
    c.text(c.w - 4, vis.Y_TITLE, HEAD_NOTE.upper(), "note", va="baseline")
    colour = vis.TEAL if board == "properties" else vis.BROWN
    _head_panel(c, board, left, top, c.w - left - 24 - key_w, panel_h, colour)
    _plm_key(c, top, panel_h, key_w, noplm=True)
    c.save(out, f"{BOARD_NAME[board]}_head_rating")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_P.project_root / "docs" / "figures"
    )
    out = parser.parse_args().out / "results"
    out.mkdir(parents=True, exist_ok=True)
    vis.setup()
    over_time_scatter(out)
    over_time_board(out, "properties")
    over_time_board(out, "real_case")
    attributes(out)
    head_boxes(out)
    head_boxes_board(out, "properties")
    head_boxes_board(out, "real_case")


if __name__ == "__main__":
    main()
