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
import re
import statistics
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import card_metadata  # noqa: E402
from enzymado.data_loading import data_registry  # noqa: E402

GLOSSARY_POINTER = "benchmark_card.json#/definitions"

# Kept as is between runs: computing it needs mmseqs, so it runs separately.
PRESERVED_STATISTICS = ("entire_protein_pool_cluster_with_mmseqs2",)


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


def _numeric_summary(subset, column):
    values = numeric(r.get(column) for r in subset)
    if not values:
        return {"kind": "numeric", "total_valid_labels": 0}
    return {
        "kind": "numeric",
        "total_valid_labels": len(values),
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "median": round(statistics.median(values), 4),
    }


def _binary_summary(subset, column):
    """Positive count and prevalence for a 0/1 label."""
    values = [(r.get(column) or "").strip() for r in subset]
    values = [v for v in values if v]
    positive = sum(1 for v in values if v in {"1", "1.0", "True", "true"})
    return {
        "kind": "binary",
        "total_valid_labels": len(values),
        "positive": positive,
        "negative": len(values) - positive,
        "positive_prevalence": round(positive / len(values), 6) if values else None,
    }


def _residue_summary(subset, config):
    """Annotated positions and their density along the sequence."""
    positions = 0
    residues = 0
    annotated_proteins = 0
    per_protein = []
    for row in subset:
        raw = (row.get(config.target_column) or "").strip()
        found = [p for p in re.split(r"[\[\],;\s]+", raw) if p.strip()]
        sequence = (row.get(config.sequence_column) or "").strip()
        residues += len(sequence)
        positions += len(found)
        if found:
            annotated_proteins += 1
            per_protein.append(len(found))
    return {
        "kind": "residue_positions",
        "total_valid_labels": len(subset),
        "proteins_with_at_least_one_annotation": annotated_proteins,
        "annotated_positions": positions,
        "residues_total": residues,
        "positive_prevalence": round(positions / residues, 6) if residues else None,
        "annotations_per_protein": {
            "mean": round(statistics.fmean(per_protein), 3) if per_protein else 0.0,
            "median": statistics.median(per_protein) if per_protein else 0,
            "max": max(per_protein) if per_protein else 0,
        },
    }


def _ec_summary(subset, config):
    """EC classes per level and width of the flat label vector."""
    separator = getattr(config, "ec_separator", ";") or ";"
    levels = [set() for _ in range(4)]
    per_protein = []
    for row in subset:
        raw = (row.get(config.target_column) or "").strip()
        codes = [c.strip() for c in raw.replace(separator, ",").split(",") if c.strip()]
        per_protein.append(len(codes))
        for code in codes:
            parts = code.split(".")
            for depth in range(min(4, len(parts))):
                if parts[depth] and parts[depth] != "-":
                    levels[depth].add(".".join(parts[: depth + 1]))
    return {
        "kind": "hierarchical_multilabel",
        "total_valid_labels": len(subset),
        "unique_classes_by_level": {
            f"level_{i + 1}": len(s) for i, s in enumerate(levels)
        },
        "flat_vector_width": sum(len(s) for s in levels),
        "ec_numbers_per_protein": {
            "mean": round(statistics.fmean(per_protein), 3) if per_protein else 0.0,
            "max": max(per_protein) if per_protein else 0,
        },
    }


def label_distribution(rows, config, split_column):
    """Summarise the label according to its kind (numeric, binary, residue or EC)."""
    kind = str(config.label_kind)
    if kind == "residue_binary_vector":
        summarise = lambda subset: _residue_summary(subset, config)
    elif kind == "multilabel_ec":
        summarise = lambda subset: _ec_summary(subset, config)
    elif kind == "binary":
        summarise = lambda subset: _binary_summary(subset, config.target_column)
    else:
        summarise = lambda subset: _numeric_summary(subset, config.target_column)

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
            rows, config, config.split_column)

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
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    stats["sequence_length_summary"]["contains_non_standard_aas"] = sum(
        1 for row in reference
        if set((row.get(config.sequence_column) or "").strip().upper()) - standard
    )
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

        # The dataset id is the file name without the split suffix, e.g. catpred_kcat.
        stem = sorted(config.dataset_paths.values())[0].stem
        existing["dataset_id"] = re.split(r"_(random|similarity)_", stem)[0]
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
        # Rebuild the block instead of merging, so fields we stop writing disappear.
        previous = existing.get("statistics", {})
        statistics_block = {key: previous[key] for key in PRESERVED_STATISTICS
                            if key in previous}
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
