"""Build an interactive per-task leaderboard from results_card.json.

results_card.json is the single artifact from the training cluster that is
checked into this repo: for every finished run (embedding pooling mode x
split family x model family x task x foundation model) it already carries
the resolved train/val/test metrics, so this module never needs to reach the
cluster or read the (gitignored) raw prediction/bootstrap files under
results/predictions and results/bootstrap.

Usage:
    python src/visualizations/leaderboard.py
    python src/visualizations/leaderboard.py --results-card path/to/results_card.json --out out.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from enzymado.evaluation.metrics_registry import higher_is_better_for, primary_metric_for  # noqa: E402

MODEL_FAMILIES = ("linear", "mlp", "xgboost")
TEMPLATE_PATH = Path(__file__).with_name("leaderboard_template.html")
CHARTS_JS_PATH = Path(__file__).with_name("leaderboard_charts.js")
APP_JS_PATH = Path(__file__).with_name("leaderboard_app.js")


@dataclass(frozen=True)
class LeaderboardRecord:
    task: str
    mode: str
    split: str
    family: str
    embedding: str
    value: float
    n: int


@dataclass(frozen=True)
class TaskMeta:
    scope: str
    task_type: str
    metric: str
    higher_is_better: bool


def load_results_card(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_records(card: dict[str, Any]) -> tuple[list[LeaderboardRecord], dict[str, TaskMeta]]:
    """Walk predictions[mode][split][family][task][embedding] and pull the primary test metric.

    Skips invalidated entries and entries missing test metrics/the primary metric,
    matching is_result_entry_complete's notion of "usable" without requiring the
    on-disk artifacts it also checks (those artifacts only exist on the cluster).
    """
    known_properties = set(card.get("properties", []))
    known_real_case = set(card.get("real_case", []))

    records: list[LeaderboardRecord] = []
    task_meta: dict[str, TaskMeta] = {}

    predictions = card.get("predictions", {})
    for mode, by_split in predictions.items():
        for split, by_family in by_split.items():
            for family, by_task in by_family.items():
                if family not in MODEL_FAMILIES:
                    continue
                for task, by_embedding in by_task.items():
                    for embedding, entry in by_embedding.items():
                        if not isinstance(entry, dict) or entry.get("invalidated"):
                            continue
                        test_metrics = (entry.get("metrics") or {}).get("test")
                        if not test_metrics:
                            continue
                        task_type = test_metrics.get("task_type")
                        metric_name = primary_metric_for(task_type) if task_type else None
                        if metric_name is None:
                            continue
                        value = test_metrics.get(metric_name)
                        if value is None or not _is_finite(value):
                            continue

                        if task not in task_meta:
                            higher_is_better = higher_is_better_for(task_type)[metric_name] > 0
                            scope = (
                                "properties"
                                if task in known_properties
                                else "real_case" if task in known_real_case else "other"
                            )
                            task_meta[task] = TaskMeta(
                                scope=scope,
                                task_type=task_type,
                                metric=metric_name,
                                higher_is_better=higher_is_better,
                            )

                        records.append(
                            LeaderboardRecord(
                                task=task,
                                mode=mode,
                                split=split,
                                family=family,
                                embedding=embedding,
                                value=float(value),
                                n=int(test_metrics.get("n", 0)),
                            )
                        )
    return records, task_meta


def _is_finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))  # noqa: PLR0133 (NaN check)


def build_payload(records: list[LeaderboardRecord], task_meta: dict[str, TaskMeta]) -> dict[str, Any]:
    ordered_tasks = sorted(task_meta, key=lambda t: (task_meta[t].scope != "properties", t))
    return {
        "model_families": list(MODEL_FAMILIES),
        "tasks": {task: asdict(meta) for task, meta in task_meta.items()},
        "task_order": ordered_tasks,
        "records": [asdict(r) for r in records],
    }


def render_html(payload: dict[str, Any]) -> str:
    """Inline the data payload and the two JS modules into the template.

    The template's <script src="leaderboard_charts.js"> / <script
    src="leaderboard_app.js"> tags are for local development (open the
    template directly and edit the JS files with a live server); the shipped
    report is a single self-contained HTML file with no sibling JS to keep
    track of, matching the "one artifact" design of results_card.json itself.
    """
    template = TEMPLATE_PATH.read_text()
    data_json = json.dumps(payload, separators=(",", ":"))
    html = template.replace("__LEADERBOARD_DATA__", data_json)
    html = html.replace(
        '<script src="leaderboard_charts.js"></script>',
        f"<script>\n{CHARTS_JS_PATH.read_text()}\n</script>",
    )
    html = html.replace(
        '<script src="leaderboard_app.js"></script>',
        f"<script>\n{APP_JS_PATH.read_text()}\n</script>",
    )
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-card", type=Path, default=REPO_ROOT / "results_card.json",
        help="Path to results_card.json (default: repo root).",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "figures" / "leaderboard" / "leaderboard.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    card = load_results_card(args.results_card)
    records, task_meta = extract_records(card)
    if not records:
        raise SystemExit(f"No usable leaderboard entries found in {args.results_card}.")

    payload = build_payload(records, task_meta)
    html = render_html(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {len(records)} records across {len(task_meta)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
