#!/usr/bin/env python
"""Refresh every dataset card and write the shared glossary into the benchmark card.

Three things happen here:

* the shared vocabulary goes into ``benchmark_card.json`` once, and each card
  points at it rather than carrying its own copy to drift out of sync;
* provenance, a description and a statement of what the label is get written
  from ``card_metadata`` for all eighteen datasets, including the real-case
  suites, which had no cards at all;
* the statistics that can be derived from the published CSVs are recomputed, so
  they stop disagreeing with the files they describe.

The MMseqs2 clustering block is left untouched: it needs ``mmseqs`` on PATH and
is heavy enough to belong in a batch job, not here. Its ``generated`` marker says
when it was last refreshed so a stale block is visible rather than assumed
current.

    python scripts/data_prep/enrich_dataset_cards.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import statistics
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import card_metadata  # noqa: E402
from enzymado.data_loading import data_registry  # noqa: E402

GLOSSARY_POINTER = "benchmark_card.json#/definitions"


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(values):
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def label_distribution(rows, column, split_column):
    """Summarise the label overall and per split, without assuming it is numeric."""
    def summarise(subset):
        raw = [r.get(column) for r in subset]
        values = numeric(raw)
        if values and len(values) >= len(raw) * 0.9:
            return {
                "kind": "numeric",
                "total_valid_labels": len(values),
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "median": round(statistics.median(values), 4),
            }
        counts: dict[str, int] = {}
        for value in raw:
            key = (value or "").strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        return {
            "kind": "categorical",
            "total_valid_labels": sum(counts.values()),
            "unique_classes": len(counts),
            "most_frequent": [{"value": k, "count": v} for k, v in top],
        }

    out = {"overall": summarise(rows)}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if (r.get(split_column) or "").strip() == split]
        if subset:
            out[split] = summarise(subset)
    return out


def sequence_summary(rows, column):
    lengths = [len(s) for s in
               ((r.get(column) or "").strip() for r in rows) if s]
    if not lengths:
        return None
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "median": statistics.median(ordered),
        "p95": ordered[int(len(ordered) * 0.95) - 1],
        "count_greater_than_1024": sum(1 for v in ordered if v > 1024),
    }


def recompute(config):
    """Statistics derivable from the published CSVs, for every split variant."""
    stats = {"generated": datetime.date.today().isoformat(),
             "split_sizes_by_method": {}, "label_distribution_by_method": {}}
    reference = None
    for variant, path in sorted(config.dataset_paths.items()):
        rows = read_rows(path)
        reference = reference or rows
        counts: dict[str, int] = {}
        for row in rows:
            key = (row.get(config.split_column) or "").strip() or "unassigned"
            counts[key] = counts.get(key, 0) + 1
        stats["split_sizes_by_method"][path.stem] = counts
        stats["label_distribution_by_method"][path.stem] = label_distribution(
            rows, config.target_column, config.split_column)

    fields = list(reference[0]) if reference else []
    stats["nans_dups"] = {
        "nan_counts_per_column": {
            name: sum(1 for r in reference if not (r.get(name) or "").strip())
            for name in fields
        },
        "duplicate_rows_full": len(reference) - len(
            {tuple(sorted(r.items())) for r in reference}),
    }
    stats["rows"] = len(reference)
    stats["distinct_proteins"] = len(
        {(r.get(config.sequence_column) or "").strip() for r in reference} - {""})
    stats["sequence_length_summary"] = sequence_summary(
        reference, config.sequence_column)
    # A column whose values are nearly all distinct carries identity, not category.
    stats["categorical_high_variance_columns"] = [
        name for name in fields
        if name != config.sequence_column
        and len({(r.get(name) or "").strip() for r in reference}) > len(reference) * 0.9
    ]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=pathlib.Path,
                        default=_REPO / "benchmark_card.json")
    args = parser.parse_args()

    card = json.loads(args.card.read_text(encoding="utf-8"))
    card["definitions"] = card_metadata.DEFINITIONS
    args.card.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"  glosario -> {args.card.name}#/definitions "
          f"({len(card_metadata.DEFINITIONS)} entradas)")

    written, missing = 0, []
    for name, config in sorted(data_registry.TASK_REGISTRY.items()):
        meta = card_metadata.SOURCES.get(name)
        if meta is None:
            missing.append(name)
            continue
        directory = sorted(config.dataset_paths.values())[0].parent
        path = directory / "dataset_card.json"
        existing = (json.loads(path.read_text(encoding="utf-8"))
                    if path.exists() else {})

        existing["property_name"] = name
        existing["task_scope"] = config.task_scope
        existing["definitions"] = GLOSSARY_POINTER
        existing["description"] = meta["description"]
        existing["origin_of_data"] = {
            "source_name": meta["source"],
            "doi": meta["doi"],
            "url": "https://doi.org/" + meta["doi"],
            **({"note": meta["provenance_note"]} if "provenance_note" in meta else {}),
        }
        existing["label"] = {
            "column": config.target_column,
            "label_kind": str(config.label_kind),
            "description": meta["label"],
            "see_also": GLOSSARY_POINTER,
        }
        statistics_block = existing.get("statistics", {})
        statistics_block.update(recompute(config))
        existing["statistics"] = statistics_block

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        written += 1
        print(f"  {name:<28} {config.task_scope:<11} {statistics_block['rows']:>7,} filas"
              f"  -> {path.relative_to(_REPO)}")

    print(f"\n  {written} cards escritos")
    if missing:
        print(f"  SIN metadata en card_metadata.py: {missing}")


if __name__ == "__main__":
    main()
