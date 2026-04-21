#!/usr/bin/env python3
"""Audit benchmark-card training coverage and recorded hyperparameters.

By default this script reads the local repository card plus the cluster scratch
card when it exists. It reports:

1. Which benchmark card was updated most recently.
2. Coverage for the non-attention model families (XGB, MLP, SGD) across the
   expected embedding models.
3. Parameter inconsistencies across foundation models for the same property and
   model type.
4. Conflicts where two benchmark cards record the same result key with
   different params.

Typical usage:

    python scripts/enzyme_utils/audit_training_registry.py

    python scripts/enzyme_utils/audit_training_registry.py \
        --card benchmark_card.json \
        --card /scratch/$USER/enzymado/cards/benchmark_card.json \
        --merge-strategy union
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FAMILIES = ("xgb", "mlp", "sgd")


@dataclass(frozen=True)
class LoadedCard:
    path: Path
    mtime: float
    data: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_card_paths() -> list[Path]:
    user = os.environ.get("USER", "user")
    candidates = [
        REPO_ROOT / "benchmark_card.json",
        Path(f"/scratch/{user}/enzymado/cards/benchmark_card.json"),
    ]
    seen: set[Path] = set()
    resolved: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            resolved.append(candidate)
            seen.add(candidate)
    return resolved


def load_cards(paths: list[Path]) -> list[LoadedCard]:
    loaded: list[LoadedCard] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        loaded.append(
            LoadedCard(
                path=resolved,
                mtime=resolved.stat().st_mtime,
                data=read_json(resolved),
            )
        )
    return loaded


def iso_mtime(path: Path) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(sep=" ", timespec="seconds")


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, canonicalize(inner)) for key, inner in value.items()))
    if isinstance(value, list):
        return tuple(canonicalize(inner) for inner in value)
    return value


def materialize(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {key: materialize(inner) for key, inner in value}
        return [materialize(inner) for inner in value]
    return value


def model_family(model_name: str | None) -> str | None:
    if not model_name:
        return None
    for family in MODEL_FAMILIES:
        if model_name.startswith(f"{family}_"):
            return family
    return None


def load_expected_embeddings(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        return sorted(payload.keys())
    raise TypeError(f"Expected a JSON object in embeddings manifest: {path}")


def collect_property_names(cards: list[LoadedCard]) -> list[str]:
    names: set[str] = set()
    for card in cards:
        names.update(card.data.get("properties", {}).keys())
    return sorted(names)


def collect_training_results(card: LoadedCard, property_name: str) -> dict[str, dict[str, Any]]:
    properties = card.data.get("properties", {})
    property_payload = properties.get(property_name, {})
    training_results = property_payload.get("training_results")
    if isinstance(training_results, dict):
        return training_results
    return {}


def newest_card(cards: list[LoadedCard]) -> LoadedCard:
    return max(cards, key=lambda card: (card.mtime, str(card.path)))


def merge_results(
    cards: list[LoadedCard],
    property_names: list[str],
    strategy: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in property_names}

    if strategy == "first":
        active_cards = cards[:1]
    elif strategy == "newest":
        active_cards = [newest_card(cards)]
    else:
        active_cards = sorted(cards, key=lambda card: (card.mtime, str(card.path)))

    for card in active_cards:
        for property_name in property_names:
            merged[property_name].update(collect_training_results(card, property_name))

    return merged


def detect_cross_card_conflicts(cards: list[LoadedCard]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []

    for card in cards:
        for property_name in card.data.get("properties", {}):
            for result_key, entry in collect_training_results(card, property_name).items():
                seen[(property_name, result_key)].append((card.path, entry))

    for (property_name, result_key), versions in sorted(seen.items()):
        if len(versions) < 2:
            continue
        params_by_card = {
            str(path): entry.get("params", {})
            for path, entry in versions
        }
        if len({canonicalize(params) for params in params_by_card.values()}) > 1:
            conflicts.append(
                {
                    "property": property_name,
                    "result_key": result_key,
                    "versions": params_by_card,
                }
            )

    return conflicts


def audit_coverage(
    merged_results: dict[str, dict[str, dict[str, Any]]],
    expected_embeddings: list[str],
) -> list[dict[str, Any]]:
    expected_set = set(expected_embeddings)
    coverage: list[dict[str, Any]] = []

    for property_name in sorted(merged_results):
        row: dict[str, Any] = {"property": property_name, "families": {}}
        results = merged_results[property_name]
        for family in MODEL_FAMILIES:
            present = sorted(
                {
                    entry.get("embedding_model")
                    for entry in results.values()
                    if model_family(entry.get("name")) == family and entry.get("embedding_model")
                }
            )
            missing = sorted(expected_set - set(present))
            row["families"][family] = {
                "present": present,
                "missing": missing,
                "count": len(present),
                "model_names": sorted(
                    {
                        entry.get("name")
                        for entry in results.values()
                        if model_family(entry.get("name")) == family
                    }
                ),
            }
        coverage.append(row)
    return coverage


def audit_param_inconsistencies(
    merged_results: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    for property_name in sorted(merged_results):
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for result_key, entry in merged_results[property_name].items():
            name = entry.get("name")
            if model_family(name) is None:
                continue
            grouped[str(name)].append((result_key, entry))

        for model_name, entries in sorted(grouped.items()):
            buckets: dict[Any, list[dict[str, str]]] = defaultdict(list)
            for result_key, entry in entries:
                params = entry.get("params", {})
                buckets[canonicalize(params)].append(
                    {
                        "result_key": result_key,
                        "embedding_model": str(entry.get("embedding_model")),
                    }
                )

            if len(buckets) <= 1:
                continue

            findings.append(
                {
                    "property": property_name,
                    "model_name": model_name,
                    "param_sets": [
                        {
                            "params": materialize(params_key),
                            "rows": sorted(rows, key=lambda row: row["embedding_model"]),
                        }
                        for params_key, rows in buckets.items()
                    ],
                }
            )

    return findings


def print_cards(cards: list[LoadedCard]) -> None:
    current_newest = newest_card(cards)
    print("Cards loaded:")
    for card in sorted(cards, key=lambda item: (item.mtime, str(item.path)), reverse=True):
        newest_marker = "  <- newest" if card == current_newest else ""
        print(f"- {card.path} | mtime={iso_mtime(card.path)}{newest_marker}")
    print()


def print_coverage(coverage: list[dict[str, Any]], expected_embeddings: list[str]) -> None:
    print(f"Coverage against {len(expected_embeddings)} expected embedding models:")
    for row in coverage:
        print(f"- {row['property']}")
        for family in MODEL_FAMILIES:
            family_row = row["families"][family]
            missing = family_row["missing"]
            missing_text = "[]" if not missing else json.dumps(missing)
            print(
                f"  {family}: {family_row['count']}/{len(expected_embeddings)} "
                f"missing={missing_text}"
            )
    print()


def print_inconsistencies(findings: list[dict[str, Any]]) -> None:
    if not findings:
        print("Hyperparameter inconsistencies across foundation models: none")
        print()
        return

    print("Hyperparameter inconsistencies across foundation models:")
    for finding in findings:
        print(f"- {finding['property']} / {finding['model_name']}")
        for param_set in finding["param_sets"]:
            embeddings = [row["embedding_model"] for row in param_set["rows"]]
            print(f"  embeddings={json.dumps(embeddings)}")
            print(f"  params={json.dumps(param_set['params'], sort_keys=True)}")
    print()


def print_conflicts(conflicts: list[dict[str, Any]]) -> None:
    if not conflicts:
        print("Cross-card conflicts for the same result key: none")
        print()
        return

    print("Cross-card conflicts for the same result key:")
    for conflict in conflicts:
        print(f"- {conflict['property']} / {conflict['result_key']}")
        for card_path, params in sorted(conflict["versions"].items()):
            print(f"  {card_path}")
            print(f"  params={json.dumps(params, sort_keys=True)}")
    print()


def has_findings(coverage: list[dict[str, Any]], inconsistencies: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> bool:
    if inconsistencies or conflicts:
        return True
    for row in coverage:
        for family in MODEL_FAMILIES:
            if row["families"][family]["missing"]:
                return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit training coverage and recorded hyperparameters across one or more benchmark cards."
    )
    parser.add_argument(
        "--card",
        action="append",
        default=[],
        help="Benchmark card to audit. Repeatable. Defaults to local repo card plus scratch card when present.",
    )
    parser.add_argument(
        "--embeddings-json",
        default=str(REPO_ROOT / "embeddings_models_output.json"),
        help="Embedding manifest used to define the expected foundation-model set.",
    )
    parser.add_argument(
        "--merge-strategy",
        choices=("union", "newest", "first"),
        default="union",
        help=(
            "How to build the effective registry view before auditing coverage and per-model param consistency. "
            "'union' keeps all entries and lets newer cards win on duplicate keys."
        ),
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit with status 1 when coverage gaps, param inconsistencies, or cross-card conflicts are found.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    card_paths = [Path(raw).expanduser() for raw in args.card] if args.card else default_card_paths()
    cards = load_cards(card_paths)
    if not cards:
        parser.error("No benchmark cards could be loaded.")

    expected_embeddings = load_expected_embeddings(Path(args.embeddings_json).expanduser().resolve())
    property_names = collect_property_names(cards)
    merged = merge_results(cards, property_names, args.merge_strategy)
    coverage = audit_coverage(merged, expected_embeddings)
    inconsistencies = audit_param_inconsistencies(merged)
    conflicts = detect_cross_card_conflicts(cards)

    print_cards(cards)
    print(f"Merge strategy for effective audit view: {args.merge_strategy}")
    print()
    print_coverage(coverage, expected_embeddings)
    print_inconsistencies(inconsistencies)
    print_conflicts(conflicts)

    findings_present = has_findings(coverage, inconsistencies, conflicts)
    if args.fail_on_findings and findings_present:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
