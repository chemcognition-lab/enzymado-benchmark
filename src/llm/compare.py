"""Deterministic evidence builder for the hypothesis engine (torch-free, no network).

The LLM reads all the markdown cards, but it is unreliable at (a) aggregating numbers
across ~20 scattered leaderboard tables and (b) resisting post-hoc cherry-picking. This
module removes (a) from the model's plate: it computes the exact head-to-head numbers
here, so the table injected into the prompt is *correct by construction*.

Everything is assembled from the same torch-free aggregations the cards use
(:mod:`llm.sources`); the authored model attributes come from ``content/model_*.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import sources as S

CONTENT_DIR = Path(__file__).resolve().parent / "content"

# results_card buckets the similarity split under "loco".
SPLIT_DISPLAY = {"random": "random", "loco": "similarity"}

# Authored attribute fields we surface as candidate *causes* of a performance gap.
DIFF_FIELDS = [
    "family", "developer", "parameters", "context_length",
    "architecture", "pretraining_data", "pretraining_objective",
]


# ---------------------------------------------------------------------------
# Model-key handling
# ---------------------------------------------------------------------------

def normalize_key(name: str) -> str:
    """Resolve a user-supplied model name to a canonical benchmark key.

    Accepts exact keys (``esm2_650M``), any casing (``esm2_650m``), and the
    ``model_`` card-id prefix (``model_esm2_650m``).
    """
    keys = S.load_models()
    if name in keys:
        return name
    lower = {k.lower(): k for k in keys}
    cand = name.lower().removeprefix("model_")
    if cand in lower:
        return lower[cand]
    raise KeyError(f"Unknown model {name!r}. Known keys: {sorted(keys)}")


def load_model_content(key: str) -> dict[str, Any]:
    """Authored ``content/model_<key>.json`` (may be empty if absent)."""
    path = CONTENT_DIR / f"model_{key.lower()}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _embedding_dim(key: str) -> int | None:
    return S.load_models().get(key, {}).get("dim")


# ---------------------------------------------------------------------------
# Per-model best scores (mirrors the leaderboards shown in the task cards)
# ---------------------------------------------------------------------------

def _best_per_task(records: list[dict], emb: str) -> dict[tuple[str, str], dict]:
    """``(split_family, task) -> best record`` for ``emb``, across all heads/modes.

    "Best" is sign-aware (higher-is-better vs lower-is-better per metric), matching
    ``sources.task_leaderboard`` — i.e. the number a reader sees in a task card.
    """
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        if r["emb"] != emb or r["primary_value"] is None:
            continue
        k = (r["split_family"], r["task"])
        cur = best.get(k)
        if cur is None or (
            r["primary_sign"] * r["primary_value"]
            > cur["primary_sign"] * cur["primary_value"]
        ):
            best[k] = r
    return best


# ---------------------------------------------------------------------------
# Pairwise head-to-head
# ---------------------------------------------------------------------------

def _standing_pair(standing: dict, x: str, y: str) -> list[dict]:
    """Competition score (1 = best) for x and y across the four evaluation tracks."""
    tracks = [
        ("no_attention", "random"), ("no_attention", "loco"),
        ("attention", "random"), ("attention", "loco"),
    ]
    rows = []
    for emode, sfam in tracks:
        xc = standing.get(x, {}).get((emode, sfam))
        yc = standing.get(y, {}).get((emode, sfam))
        if not xc and not yc:
            continue
        rows.append({
            "track": f"{emode} · {SPLIT_DISPLAY.get(sfam, sfam)}",
            "x_score": xc["score"] if xc else None,
            "y_score": yc["score"] if yc else None,
            "x_n": xc["n"] if xc else 0,
            "y_n": yc["n"] if yc else 0,
        })
    return rows


def head_to_head(records: list[dict], x: str, y: str) -> dict[str, Any]:
    """Exact per-task, per-split comparison of two models + overall mean-rank.

    Returns a structured packet: ``rows`` (one per shared split×task with the metric,
    both values, sign-aware delta, and winner), ``standing`` (mean-rank per track), and
    a ``tally`` of task wins. Positive ``delta`` means x is better.
    """
    x, y = normalize_key(x), normalize_key(y)
    bx, by = _best_per_task(records, x), _best_per_task(records, y)

    rows: list[dict] = []
    shared = sorted(set(bx) & set(by), key=lambda k: (0 if k[0] == "random" else 1, k[1]))
    for sfam, task in shared:
        rx, ry = bx[(sfam, task)], by[(sfam, task)]
        sign = rx["primary_sign"]
        delta = sign * (rx["primary_value"] - ry["primary_value"])
        winner = x if delta > 0 else (y if delta < 0 else "tie")
        rows.append({
            "task": S._display_task(task),
            "split": SPLIT_DISPLAY.get(sfam, sfam),
            "metric": rx["primary_metric"],
            "x_value": round(rx["primary_value"], 4),
            "y_value": round(ry["primary_value"], 4),
            "delta": round(delta, 4),
            "winner": winner,
            "x_head": rx["model_name"],
            "y_head": ry["model_name"],
        })

    tally = {x: sum(r["winner"] == x for r in rows),
             y: sum(r["winner"] == y for r in rows),
             "tie": sum(r["winner"] == "tie" for r in rows)}
    return {
        "x": x, "y": y, "rows": rows,
        "standing": _standing_pair(S.model_standing(records), x, y),
        "tally": tally,
        "dropped": _dropped_tasks({x: bx, y: by}),
    }


def _dropped_tasks(best_by_model: dict[str, dict]) -> list[dict]:
    """Split×tasks present for only one side (excluded from the comparison).

    Pure set logic over which runs exist — no metric is computed. Used to disclose
    that the head-to-head is over the *intersection* of the two models' runs.
    """
    labels = list(best_by_model)
    all_keys: set = set().union(*(set(b) for b in best_by_model.values())) if best_by_model else set()
    shared = set.intersection(*(set(b) for b in best_by_model.values())) if best_by_model else set()
    dropped = []
    for sfam, task in sorted(all_keys - shared, key=lambda k: (0 if k[0] == "random" else 1, k[1])):
        present = [lbl for lbl in labels if (sfam, task) in best_by_model[lbl]]
        dropped.append({
            "task": S._display_task(task),
            "split": SPLIT_DISPLAY.get(sfam, sfam),
            "present_only_for": ", ".join(present),
        })
    return dropped


def attribute_diff(x: str, y: str) -> dict[str, Any]:
    """Diff two models' authored attributes into shared vs differing candidate causes.

    Surfaces ``needs_verification`` and ``caveats`` so a hypothesis leaning on a soft
    attribute can be down-weighted.
    """
    x, y = normalize_key(x), normalize_key(y)
    cx, cy = load_model_content(x), load_model_content(y)

    differing, shared = [], []
    for field in DIFF_FIELDS:
        vx, vy = cx.get(field), cy.get(field)
        (shared if vx == vy and vx is not None else differing).append(
            {"field": field, "x": vx, "y": vy})
    dx, dy = _embedding_dim(x), _embedding_dim(y)
    (shared if dx == dy and dx is not None else differing).append(
        {"field": "embedding_dim", "x": dx, "y": dy})

    return {
        "x": x, "y": y,
        "differing": differing,
        "shared": shared,
        "flags": {
            x: {"needs_verification": bool(cx.get("needs_verification")),
                "caveats": cx.get("caveats")},
            y: {"needs_verification": bool(cy.get("needs_verification")),
                "caveats": cy.get("caveats")},
        },
    }


# ---------------------------------------------------------------------------
# Combination (setA vs setB) — coverage + diversity
# ---------------------------------------------------------------------------

def _set_best_per_task(records, members) -> dict[tuple[str, str], dict]:
    """``(split, task) -> best record across the whole set`` (which member wins)."""
    per_member = {m: _best_per_task(records, m) for m in members}
    out: dict[tuple[str, str], dict] = {}
    for m, best in per_member.items():
        for k, r in best.items():
            cur = out.get(k)
            if cur is None or (
                r["primary_sign"] * r["primary_value"]
                > cur["primary_sign"] * cur["primary_value"]
            ):
                out[k] = {**r, "member": m}
    return out


def _diversity(members) -> list[dict]:
    rows = []
    for m in members:
        c = load_model_content(m)
        rows.append({
            "model": m,
            "embedding_dim": _embedding_dim(m),
            "family": c.get("family"),
            "pretraining_objective": c.get("pretraining_objective"),
        })
    return rows


def combo_coverage(records, set_a, set_b) -> dict[str, Any]:
    """Compare two *sets* of models by task coverage + attribute diversity.

    For each shared split×task, the better set is whichever contains the winning model.
    Returns per-task winners, a per-set win tally, and each set's diversity profile.
    """
    a = [normalize_key(m) for m in set_a]
    b = [normalize_key(m) for m in set_b]
    ba, bb = _set_best_per_task(records, a), _set_best_per_task(records, b)

    rows: list[dict] = []
    shared = sorted(set(ba) & set(bb), key=lambda k: (0 if k[0] == "random" else 1, k[1]))
    for sfam, task in shared:
        ra, rb = ba[(sfam, task)], bb[(sfam, task)]
        sign = ra["primary_sign"]
        delta = sign * (ra["primary_value"] - rb["primary_value"])
        winner = "A" if delta > 0 else ("B" if delta < 0 else "tie")
        rows.append({
            "task": S._display_task(task),
            "split": SPLIT_DISPLAY.get(sfam, sfam),
            "metric": ra["primary_metric"],
            "a_value": round(ra["primary_value"], 4), "a_member": ra["member"],
            "b_value": round(rb["primary_value"], 4), "b_member": rb["member"],
            "delta": round(delta, 4),
            "winner": winner,
        })

    tally = {"A": sum(r["winner"] == "A" for r in rows),
             "B": sum(r["winner"] == "B" for r in rows),
             "tie": sum(r["winner"] == "tie" for r in rows)}
    return {
        "set_a": a, "set_b": b, "rows": rows, "tally": tally,
        "diversity": {"A": _diversity(a), "B": _diversity(b)},
        "dropped": _dropped_tasks({"A": ba, "B": bb}),
    }


# ---------------------------------------------------------------------------
# Markdown renderers — the exact text injected into the prompt
# ---------------------------------------------------------------------------

def _methodology_md(dropped: list[dict]) -> str:
    """Deterministic disclaimer about how the comparison is constructed."""
    lines = [
        "#### Methodology & caveats",
        "- **Best-config per task.** Each score is the model's *best-performing* head/"
        "pooling for that task, so a comparison may mix configurations rather than hold "
        "one fixed. Note this when a claim hinges on a single task.",
    ]
    if dropped:
        items = "; ".join(
            f"{d['task']} ({d['split']}, run only for {d['present_only_for']})" for d in dropped)
        lines.append(f"- **Excluded from the comparison** (a side has no run): {items}.")
    else:
        lines.append("- **Excluded tasks:** none — every shared split×task is compared.")
    return "\n".join(lines)


def render_head_to_head_md(h2h: dict) -> str:
    x, y = h2h["x"], h2h["y"]
    n_models = len(S.load_models())
    lines = [f"### Head-to-head: `{x}` (X) vs `{y}` (Y)", ""]
    lines.append(
        f"Overall competition score against the other {n_models} models — each task/head "
        "group scored `(n - rank) / (n - 1)`, so **1.0 = beat every competitor, 0.0 = "
        "beaten by all**, on the same scale whether the group had 2 entrants or 14. "
        "`groups` is how many task/head groups each side was averaged over; when those "
        "differ the two sides were scored over different tasks, so read the gap loosely:")
    lines.append("")
    lines.append("| Track | X score | X groups | Y score | Y groups |")
    lines.append("|---|---|---|---|---|")
    for s in h2h["standing"]:
        # A side with no runs on a track scores None — render it as an explicit dash so
        # the model reads "did not run" rather than treating a blank cell as a zero.
        xs = f"{s['x_score']:.3f}" if s["x_score"] is not None else "—"
        ys = f"{s['y_score']:.3f}" if s["y_score"] is not None else "—"
        lines.append(f"| {s['track']} | {xs} | {s['x_n']} | {ys} | {s['y_n']} |")
    lines += ["", "Per-task best score (sign-aware; Δ>0 means X is better):", ""]
    lines.append("| Task | Split | Metric | X | Y | Δ | Winner |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in h2h["rows"]:
        win = "X" if r["winner"] == x else ("Y" if r["winner"] == y else "tie")
        lines.append(
            f"| {r['task']} | {r['split']} | {r['metric']} | "
            f"{r['x_value']} | {r['y_value']} | {r['delta']:+} | {win} |")
    t = h2h["tally"]
    lines += ["", f"**Task wins:** X={t[x]}, Y={t[y]}, tie={t['tie']}.",
              "", _methodology_md(h2h.get("dropped", []))]
    return "\n".join(lines)


def render_attribute_diff_md(diff: dict) -> str:
    x, y = diff["x"], diff["y"]
    lines = [f"### Differentiating attributes: `{x}` (X) vs `{y}` (Y)", ""]
    lines.append("| Attribute | X | Y |")
    lines.append("|---|---|---|")
    for d in diff["differing"]:
        lines.append(f"| {d['field']} | {d['x']} | {d['y']} |")
    for m in (x, y):
        f = diff["flags"][m]
        if f["needs_verification"]:
            lines.append("")
            lines.append(f"> ⚠️ `{m}` attributes are flagged **needs_verification** — "
                         "cap confidence for claims resting on them.")
        if f["caveats"]:
            lines.append(f"> Caveat (`{m}`): {f['caveats']}")
    return "\n".join(lines)


def render_combo_md(combo: dict) -> str:
    a, b = combo["set_a"], combo["set_b"]
    lines = [f"### Combination coverage: A={a} vs B={b}", ""]
    lines.append("Per-task best-in-set score (Δ>0 means set A is better):")
    lines.append("")
    lines.append("| Task | Split | Metric | A (member) | B (member) | Δ | Winner |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in combo["rows"]:
        lines.append(
            f"| {r['task']} | {r['split']} | {r['metric']} | "
            f"{r['a_value']} ({r['a_member']}) | {r['b_value']} ({r['b_member']}) | "
            f"{r['delta']:+} | {r['winner']} |")
    t = combo["tally"]
    lines += ["", f"**Task wins:** A={t['A']}, B={t['B']}, tie={t['tie']}.", ""]
    for label in ("A", "B"):
        lines.append(f"**Set {label} diversity:**")
        for d in combo["diversity"][label]:
            lines.append(
                f"- `{d['model']}` — dim {d['embedding_dim']}, {d['family']}, "
                f"objective: {d['pretraining_objective']}")
        lines.append("")
    lines.append(_methodology_md(combo.get("dropped", [])))
    return "\n".join(lines).rstrip()