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

Length statistics come per row, per distinct protein, and per distinct protein
among the sequences every PLM embedded (``used_in_runs``), which is what Table 1
of the paper reports. The legacy ``additional_important`` and
``amount_of_removed_data`` blocks are dropped.

The MMseqs2 clustering block is kept as is unless ``--clustering`` is given: it
needs ``mmseqs`` on PATH (``module load mmseqs2``) and is then recomputed on the
sequences of the published split file, with its diversity figure.

    python scripts/data_prep/enrich_dataset_cards.py
    python scripts/data_prep/enrich_dataset_cards.py --clustering
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

import pandas as pd
from safetensors import safe_open

from enzymado import paths_and_constants
from enzymado.data_loading import data_registry
from enzymado.dataset_card import clustering

_P = paths_and_constants.Paths
_REPO = _P.project_root
sys.path.insert(0, str(_P.scripts_dir / "data_prep"))

import card_metadata  # noqa: E402

GLOSSARY_POINTER = "benchmark_card.json#/definitions"
CLUSTER_KEY = "entire_protein_pool_cluster_with_mmseqs2"
LEGACY_BLOCKS = ("additional_important", "amount_of_removed_data")
SEQ_IDENTITY = 0.5
COMMON_IDS = _P.results_dir / "embedding_intersection" / "common_ids.txt"
# Sequence sets embedded after common_ids.txt was written, each with the same 14
# models: the 28 case studies added on 2026-10-30. Without them `used_in_runs`
# reads zero for every dataset built on those sequences.
POOLED_ROOT = _P.data_dir / "embeddings_mean_pooled"
EXTRA_POOLED_DIRS = (POOLED_ROOT / "real_case_new28_sequences",)
POOL = _P.data_processed_dir / "protein_pool.csv"
DIVERSITY_DIR = _P.results_figures_dir / "cluster_diversity"


def embedded_ids():
    """Ids every one of the 14 models holds, across the published and later stores."""
    ids = set(COMMON_IDS.read_text().split())
    models = sorted(
        p.stem for p in (POOLED_ROOT / "protein_sequences_unique").glob("*.safetensors")
    )
    for directory in EXTRA_POOLED_DIRS:
        held = []
        for model in models:
            with safe_open(
                str(directory / f"{model}.safetensors"), framework="pt", device="cpu"
            ) as handle:
                held.append(set(handle.keys()))
        ids |= set.intersection(*held)
    return ids


def used_sequences():
    """Sequences every PLM embedded, the only ones any run trains or tests on."""
    ids = embedded_ids()
    pool = pd.read_csv(POOL)
    kept = pool[pool.sequence_id.isin(ids)].protein_sequence
    return set(kept.str.strip().str.upper())


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

    def summarise(subset):
        match kind:
            case "residue_binary_vector":
                return _residue_summary(subset, config)
            case "multilabel_ec":
                return _ec_summary(subset, config)
            case "binary":
                return _binary_summary(subset, config.target_column)
            case _:
                return _numeric_summary(subset, config.target_column)

    out = {"overall": summarise(rows)}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if (r.get(split_column) or "").strip() == split]
        if subset:
            out[split] = summarise(subset)
    return out


def sequence_summary(rows, column):
    lengths = [len(s) for s in ((r.get(column) or "").strip() for r in rows) if s]
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


def recompute(config, used):
    """Statistics derivable from the published CSVs, for every split variant."""
    stats = {
        "generated": datetime.date.today().isoformat(),
        "split_sizes_by_method": {},
        "label_distribution_by_method": {},
    }
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
            rows, config, config.split_column
        )

    fields = list(reference[0]) if reference else []
    stats["nans_dups"] = {
        "nan_counts_per_column": {
            name: sum(1 for r in reference if not (r.get(name) or "").strip())
            for name in fields
        },
        "duplicate_rows_full": len(reference)
        - len({tuple(sorted(r.items())) for r in reference}),
    }
    stats["rows"] = len(reference)
    stats["distinct_proteins"] = len(
        {(r.get(config.sequence_column) or "").strip() for r in reference} - {""}
    )
    stats["sequence_length_summary"] = sequence_summary(
        reference, config.sequence_column
    )
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    stats["sequence_length_summary"]["contains_non_standard_aas"] = sum(
        1
        for row in reference
        if set((row.get(config.sequence_column) or "").strip().upper()) - standard
    )
    distinct = {(r.get(config.sequence_column) or "").strip() for r in reference} - {""}
    stats["distinct_protein_length_summary"] = sequence_summary(
        [{"s": s} for s in distinct], "s"
    )
    kept = [
        r
        for r in reference
        if (r.get(config.sequence_column) or "").strip().upper() in used
    ]
    kept_distinct = {(r.get(config.sequence_column) or "").strip() for r in kept}
    stats["used_in_runs"] = {
        "note": "rows whose sequence every PLM embedded (common_ids)",
        "rows": len(kept),
        "distinct_proteins": len(kept_distinct),
        "distinct_protein_length_summary": sequence_summary(
            [{"s": s} for s in kept_distinct], "s"
        ),
    }
    # A column whose values are nearly all distinct carries identity, not category.
    stats["categorical_high_variance_columns"] = [
        name
        for name in fields
        if name != config.sequence_column
        and len({(r.get(name) or "").strip() for r in reference}) > len(reference) * 0.9
    ]
    return stats


def cluster_block(config, name):
    """MMseqs2 diversity of the published split file, and its rank-abundance figure."""
    path = sorted(config.dataset_paths.items())[0][1]
    stats = clustering.run(pd.read_csv(path, low_memory=False), name, SEQ_IDENTITY)
    plot = clustering.plot_diversity(stats, name, DIVERSITY_DIR, SEQ_IDENTITY)
    block = {k: v for k, v in stats.items() if not k.startswith("_")}
    block["source_file"] = path.name
    block["generated"] = datetime.date.today().isoformat()
    block["cluster_diversity_plot"] = str(pathlib.Path(plot).relative_to(_REPO))
    block["dendrogram_plot"] = None
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card", type=pathlib.Path, default=_REPO / "benchmark_card.json"
    )
    parser.add_argument(
        "--clustering",
        action="store_true",
        help="Recompute the MMseqs2 block (needs mmseqs on PATH).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="NAME",
        help="Repeatable. Write only these cards, leaving every other "
        "card on disk untouched. Defaults to all of them.",
    )
    args = parser.parse_args()
    used = used_sequences()

    card = json.loads(args.card.read_text(encoding="utf-8"))
    card["definitions"] = card_metadata.DEFINITIONS
    args.card.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(
        f"  glosario -> {args.card.name}#/definitions "
        f"({len(card_metadata.DEFINITIONS)} entradas)"
    )

    written, missing = 0, []
    for name, config in sorted(data_registry.TASK_REGISTRY.items()):
        if args.dataset and name not in args.dataset:
            continue
        meta = card_metadata.SOURCES.get(name)
        if meta is None:
            missing.append(name)
            continue
        directory = sorted(config.dataset_paths.values())[0].parent
        path = directory / "dataset_card.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        # The dataset id is the file name without the split suffix, e.g. catpred_kcat.
        stem = sorted(config.dataset_paths.values())[0].stem
        existing["dataset_id"] = re.split(r"_(random|similarity)_", stem)[0]
        existing["property_name"] = name
        existing["task_scope"] = config.task_scope
        existing["definitions"] = GLOSSARY_POINTER
        existing["description"] = meta["description"]
        existing["origin_of_data"] = {
            "source_name": meta["source"],
            "doi": meta.get("doi"),
            # A dataset whose publication the repo does not record carries no
            # DOI; the note says the citation is pending rather than guessing one.
            **({"url": "https://doi.org/" + meta["doi"]} if meta.get("doi") else {}),
            **({"note": meta["provenance_note"]} if "provenance_note" in meta else {}),
        }
        existing["label"] = {
            "column": config.target_column,
            "label_kind": str(config.label_kind),
            "description": meta["label"],
            "see_also": GLOSSARY_POINTER,
        }
        for key in LEGACY_BLOCKS:
            existing.pop(key, None)
        # Rebuild the block instead of merging, so fields we stop writing disappear.
        previous = existing.get("statistics", {})
        statistics_block = {}
        if CLUSTER_KEY in previous:
            statistics_block[CLUSTER_KEY] = (
                cluster_block(config, directory.name)
                if args.clustering
                else previous[CLUSTER_KEY]
            )
        statistics_block.update(recompute(config, used))
        existing["statistics"] = statistics_block

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        written += 1
        rows_count = statistics_block["rows"]
        print(
            f"  {name:<28} {config.task_scope:<11} {rows_count:>7,} filas"
            f"  -> {path.relative_to(_REPO)}"
        )

    print(f"\n  {written} cards escritos")
    if missing:
        print(f"  SIN metadata en card_metadata.py: {missing}")


if __name__ == "__main__":
    main()
