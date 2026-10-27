"""Exact comparisons of two representations, or two sets of them, for the assistant.

A language model reading the cards is unreliable at adding up numbers spread over
many tables and at resisting a cherry-picked task, so every cross-model number the
assistant quotes is computed here. Each task uses a representation's best
baseline, and a gap within the noise threshold of the two winning baselines
counts as a tie, the same rule the leaderboards use. Overall standing comes from
the published Bradley-Terry leaderboards.

Example:
    >>> from enzymado.llm import compare, sources
    >>> compare.head_to_head(sources.load_records(), "prost5", "esm2_650M")["tally"]
"""

from __future__ import annotations

import json
from pathlib import Path
import typing

from enzymado.llm import sources

CONTENT_DIR = Path(__file__).resolve().parent / "content"

# Authored fields offered as candidate causes of a performance gap.
CONTENT_FIELDS = (
    "family",
    "developer",
    "architecture",
    "pretraining_data",
    "pretraining_objective",
)


def normalize_key(name: str) -> str:
    """Resolve a user-supplied name to a representation key.

    Accepts the exact key (``esm2_650M``), any casing (``esm2_650m``) and the
    card-id prefix (``model_esm2_650m``).
    """
    keys = sources.load_models()
    if name in keys:
        return name
    lower = {k.lower(): k for k in keys}
    candidate = name.strip().lower().removeprefix("model_")
    if candidate in lower:
        return lower[candidate]
    raise KeyError(f"Unknown model {name!r}. Known keys: {sorted(keys)}")


def load_model_content(key: str) -> dict[str, typing.Any]:
    """Authored ``content/model_<key>.json``, empty if absent."""
    path = CONTENT_DIR / f"model_{key.lower()}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _outcome(a: dict, b: dict) -> tuple[float, float | None, str]:
    """Signed gap (positive when ``a`` is better), threshold and winner label."""
    delta = a["primary_sign"] * (a["primary_value"] - b["primary_value"])
    threshold = sources.noise_threshold(a, b)
    if threshold is not None and abs(delta) <= threshold:
        return delta, threshold, "tie"
    return delta, threshold, "a" if delta > 0 else "b"


def _dropped_tasks(best_by_side: dict[str, dict]) -> list[dict]:
    """Split and task pairs present for only one side, so left out of a comparison."""
    keys = [set(best) for best in best_by_side.values()]
    every = set().union(*keys) if keys else set()
    shared = set.intersection(*keys) if keys else set()
    dropped = []
    for split, task in sorted(
        every - shared, key=lambda k: (sources.split_order(k[0]), k[1])
    ):
        present = [side for side, best in best_by_side.items() if (split, task) in best]
        dropped.append(
            {"task": task, "split": split, "present_only_for": ", ".join(present)}
        )
    return dropped


def _shared_keys(*bests: dict) -> list[tuple[str, str]]:
    shared = set.intersection(*(set(b) for b in bests))
    return sorted(shared, key=lambda k: (sources.split_order(k[0]), k[1]))


def head_to_head(records: list[dict], x: str, y: str) -> dict[str, typing.Any]:
    """Per task and split, the best score of two representations and who wins.

    ``delta`` is positive when x is better; ``winner`` is x, y or ``tie`` when the
    gap is within the noise threshold. ``standing`` holds both sides' leaderboard
    entries and ``dropped`` the tasks only one side ran.
    """
    x, y = normalize_key(x), normalize_key(y)
    best_x = sources.best_per_task(records, x)
    best_y = sources.best_per_task(records, y)
    rows = []
    for split, task in _shared_keys(best_x, best_y):
        rx, ry = best_x[(split, task)], best_y[(split, task)]
        delta, threshold, side = _outcome(rx, ry)
        rows.append(
            {
                "task": task,
                "split": split,
                "metric": rx["primary_metric"],
                "x_value": round(rx["primary_value"], 4),
                "y_value": round(ry["primary_value"], 4),
                "delta": round(delta, 4),
                "threshold": None if threshold is None else round(threshold, 4),
                "winner": {"a": x, "b": y, "tie": "tie"}[side],
                "x_config": rx["config"],
                "y_config": ry["config"],
            }
        )
    tally = {
        x: sum(r["winner"] == x for r in rows),
        y: sum(r["winner"] == y for r in rows),
        "tie": sum(r["winner"] == "tie" for r in rows),
    }
    return {
        "x": x,
        "y": y,
        "rows": rows,
        "standing": {x: sources.model_standing(x), y: sources.model_standing(y)},
        "tally": tally,
        "dropped": _dropped_tasks({x: best_x, y: best_y}),
    }


def _facts_row(field: str, fx: typing.Any, fy: typing.Any) -> dict[str, typing.Any]:
    return {"field": field, "x": fx, "y": fy}


def attribute_diff(x: str, y: str) -> dict[str, typing.Any]:
    """Split two representations' attributes into shared and differing ones.

    Parameters, width, architecture and objective come from
    ``scripts/figures/plm_attributes.csv``; the prose fields from the authored
    cards, whose ``needs_verification`` flag and caveats are passed along so a
    hypothesis resting on a soft attribute can be down-weighted.
    """
    x, y = normalize_key(x), normalize_key(y)
    facts = sources.load_models()
    cx, cy = load_model_content(x), load_model_content(y)
    candidates = [
        _facts_row(
            "parameters",
            sources.format_params(facts[x]["params"]),
            sources.format_params(facts[y]["params"]),
        ),
        _facts_row("embedding_dim", facts[x]["dim"], facts[y]["dim"]),
        _facts_row("architecture_class", facts[x]["arch"], facts[y]["arch"]),
        _facts_row("objective", facts[x]["objective"], facts[y]["objective"]),
    ] + [_facts_row(f, cx.get(f), cy.get(f)) for f in CONTENT_FIELDS]
    shared, differing = [], []
    for row in candidates:
        same = row["x"] == row["y"] and row["x"] is not None
        (shared if same else differing).append(row)
    return {
        "x": x,
        "y": y,
        "differing": differing,
        "shared": shared,
        "flags": {
            key: {
                "needs_verification": bool(content.get("needs_verification")),
                "caveats": content.get("caveats"),
            }
            for key, content in ((x, cx), (y, cy))
        },
    }


def _set_best_per_task(records: list[dict], members: list[str]) -> dict:
    """``(split, task) -> best record across a set``, tagged with its member."""
    out: dict[tuple[str, str], dict] = {}
    for member in members:
        for key, record in sources.best_per_task(records, member).items():
            current = out.get(key)
            sign = record["primary_sign"]
            if current is None or (
                sign * record["primary_value"] > sign * current["primary_value"]
            ):
                out[key] = {**record, "member": member}
    return out


def _diversity(members: list[str]) -> list[dict]:
    facts = sources.load_models()
    return [
        {
            "model": m,
            "embedding_dim": facts[m]["dim"],
            "parameters": sources.format_params(facts[m]["params"]),
            "family": load_model_content(m).get("family"),
            "objective": facts[m]["objective"],
        }
        for m in members
    ]


def combo_coverage(
    records: list[dict], set_a: list[str], set_b: list[str]
) -> dict[str, typing.Any]:
    """Compare two sets of representations task by task.

    On each shared task and split the set holding the better best-in-set score
    wins, unless the gap is within the noise threshold. Returns the per-task rows,
    a win tally and each set's diversity profile.
    """
    a = [normalize_key(m) for m in set_a]
    b = [normalize_key(m) for m in set_b]
    best_a, best_b = _set_best_per_task(records, a), _set_best_per_task(records, b)
    rows = []
    for split, task in _shared_keys(best_a, best_b):
        ra, rb = best_a[(split, task)], best_b[(split, task)]
        delta, threshold, side = _outcome(ra, rb)
        rows.append(
            {
                "task": task,
                "split": split,
                "metric": ra["primary_metric"],
                "a_value": round(ra["primary_value"], 4),
                "a_member": ra["member"],
                "b_value": round(rb["primary_value"], 4),
                "b_member": rb["member"],
                "delta": round(delta, 4),
                "threshold": None if threshold is None else round(threshold, 4),
                "winner": {"a": "A", "b": "B", "tie": "tie"}[side],
            }
        )
    tally = {w: sum(r["winner"] == w for r in rows) for w in ("A", "B", "tie")}
    return {
        "set_a": a,
        "set_b": b,
        "rows": rows,
        "tally": tally,
        "diversity": {"A": _diversity(a), "B": _diversity(b)},
        "dropped": _dropped_tasks({"A": best_a, "B": best_b}),
    }


def _methodology_md(dropped: list[dict]) -> str:
    """Fixed note on how the comparison is built."""
    lines = [
        "#### Methodology & caveats",
        "- **Best baseline per task.** Each score is the representation's best "
        "head and pooling on that task, so a comparison can mix baselines. Say so "
        "when a claim rests on a single task.",
        f"- **Ties.** A gap no larger than the noise threshold "
        f"z * sqrt(sigma_x^2 + sigma_y^2) (z = {sources.tie_z():g}, sigma the "
        "training-noise SD of each winning baseline) is a tie, as in the "
        "leaderboards. Scalar XGBoost heads give identical predictions for every "
        "seed, so their sigma is 0 and any gap between two of them counts.",
    ]
    if dropped:
        items = "; ".join(
            f"{d['task']} ({d['split']}, run only for {d['present_only_for']})"
            for d in dropped
        )
        lines.append(f"- **Left out** (one side has no run): {items}.")
    else:
        lines.append(
            "- **Left out:** nothing; every shared task and split is compared."
        )
    return "\n".join(lines)


def _standing_lines(standing: dict, key: str, label: str) -> list[str]:
    lines = []
    for leaderboard, entry in standing[key].items():
        best = entry["best"]
        lines.append(
            f"| {label} | {leaderboard} | {best['rating']} "
            f"[{best['ci_lo']}, {best['ci_hi']}] | {best['config']} | "
            f"{best['rank']} of {entry['n_competitors']} | {entry['mean']} |"
        )
    return lines


def _value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def render_head_to_head_md(h2h: dict) -> str:
    x, y = h2h["x"], h2h["y"]
    lines = [
        f"### Head-to-head: `{x}` (X) vs `{y}` (Y)",
        "",
        "Leaderboard standing (Bradley-Terry rating with 95% bootstrap interval of "
        "each side's best baseline, its rank among all competitors, and the mean "
        "rating of its five baselines):",
        "",
        "| Side | Leaderboard | Best rating [95% CI] | Baseline | Rank | Mean |",
        "|---|---|---|---|---|---|",
        *_standing_lines(h2h["standing"], x, "X"),
        *_standing_lines(h2h["standing"], y, "Y"),
        "",
        "Best score per task (delta > 0 means X is better; a gap within the noise "
        "threshold is a tie):",
        "",
        "| Task | Split | Metric | X | Y | Delta | Threshold | Winner |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in h2h["rows"]:
        win = {x: "X", y: "Y"}.get(r["winner"], "tie")
        lines.append(
            f"| {r['task']} | {r['split']} | {r['metric']} | {r['x_value']} "
            f"({r['x_config']}) | {r['y_value']} ({r['y_config']}) | "
            f"{r['delta']:+} | {_value(r['threshold'])} | {win} |"
        )
    t = h2h["tally"]
    lines += [
        "",
        f"**Task wins:** X={t[x]}, Y={t[y]}, tie={t['tie']}.",
        "",
        _methodology_md(h2h.get("dropped", [])),
    ]
    return "\n".join(lines)


def render_attribute_diff_md(diff: dict) -> str:
    x, y = diff["x"], diff["y"]
    lines = [
        f"### Differentiating attributes: `{x}` (X) vs `{y}` (Y)",
        "",
        "| Attribute | X | Y |",
        "|---|---|---|",
    ]
    lines += [f"| {d['field']} | {d['x']} | {d['y']} |" for d in diff["differing"]]
    for key in (x, y):
        flags = diff["flags"][key]
        if flags["needs_verification"]:
            lines += [
                "",
                f"> `{key}` attributes are flagged **needs_verification**: cap the "
                "confidence of claims resting on them.",
            ]
        if flags["caveats"]:
            lines.append(f"> Caveat (`{key}`): {flags['caveats']}")
    return "\n".join(lines)


def render_combo_md(combo: dict) -> str:
    lines = [
        f"### Combination coverage: A={combo['set_a']} vs B={combo['set_b']}",
        "",
        "Best-in-set score per task (delta > 0 means set A is better; a gap within "
        "the noise threshold is a tie):",
        "",
        "| Task | Split | Metric | A (member) | B (member) | Delta | Threshold "
        "| Winner |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in combo["rows"]:
        lines.append(
            f"| {r['task']} | {r['split']} | {r['metric']} | {r['a_value']} "
            f"({r['a_member']}) | {r['b_value']} ({r['b_member']}) | "
            f"{r['delta']:+} | {_value(r['threshold'])} | {r['winner']} |"
        )
    t = combo["tally"]
    lines += ["", f"**Task wins:** A={t['A']}, B={t['B']}, tie={t['tie']}.", ""]
    for label in ("A", "B"):
        lines.append(f"**Set {label} diversity:**")
        lines += [
            f"- `{d['model']}`: width {d['embedding_dim']}, {d['parameters']} "
            f"parameters, {d['family']}, objective: {d['objective']}"
            for d in combo["diversity"][label]
        ]
        lines.append("")
    lines.append(_methodology_md(combo.get("dropped", [])))
    return "\n".join(lines).rstrip()
