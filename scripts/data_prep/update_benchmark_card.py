#!/usr/bin/env python
"""Recompute the benchmark card's summary statistics from the published data.

The card already carries the functional registry that the evaluation pipeline
reads (task types, prevalences, dataset paths). This adds a sibling ``statistics``
block holding the counts that describe the benchmark as a whole, derived rather
than typed, so the numbers in the paper and on the site cannot drift from the
files they claim to summarise.

Residue-level tasks are counted per protein, not per annotated position: one row
is one labelled protein, and counting its residues instead would make two site
datasets outweigh every other dataset combined.

    python scripts/data_prep/update_benchmark_card.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado.data_loading import data_registry  # noqa: E402
from enzymado.evaluation import metrics_registry  # noqa: E402
from enzymado.training import model_registry  # noqa: E402

# One entry per pooling and head combination that the benchmark actually runs.
# Attention pooling is learned jointly with the head, so it has no counterpart
# for gradient-boosted trees; that is why this is five and not six.
BASELINES = [
    ("no_attention", "linear"), ("no_attention", "mlp"),
    ("no_attention", "xgboost"), ("attention", "linear"), ("attention", "mlp"),
]


def dataset_rows(config):
    """Row count and distinct protein count for one task's published CSV."""
    path = sorted(config.dataset_paths.items())[0][1]
    rows, proteins = 0, set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            sequence = (row.get(config.sequence_column) or "").strip()
            if sequence:
                proteins.add(sequence)
    return rows, len(proteins)


def embedding_models(repo):
    """The manifest is machine-local, so fall back to its committed example."""
    for name in ("embeddings_models_output.json",
                 "embeddings_models_output.example.json"):
        path = repo / name
        if path.exists():
            return sorted(json.loads(path.read_text(encoding="utf-8")))
    return []


def build_statistics(repo):
    per_dataset, totals = {}, {"rows": 0, "proteins": 0}
    scope_rows: dict[str, int] = {}
    for name, config in sorted(data_registry.TASK_REGISTRY.items()):
        rows, proteins = dataset_rows(config)
        per_dataset[name] = {
            "task_scope": config.task_scope,
            "label_kind": str(config.label_kind),
            "batch_kind": str(config.batch_kind),
            "datapoints": rows,
            "distinct_proteins": proteins,
            "split_variants": sorted(config.dataset_paths),
        }
        totals["rows"] += rows
        totals["proteins"] += proteins
        scope_rows[config.task_scope] = scope_rows.get(config.task_scope, 0) + rows

    metrics = {
        kind: [spec.name for spec in metrics_registry.metric_specs(kind)]
        for kind in ("regression", "classification")
    }
    models = embedding_models(repo)
    scopes = sorted({c.task_scope for c in data_registry.TASK_REGISTRY.values()})

    return {
        "generated": datetime.date.today().isoformat(),
        "datasets": {
            "total": len(per_dataset),
            "by_scope": {
                scope: sum(1 for d in per_dataset.values()
                           if d["task_scope"] == scope)
                for scope in scopes
            },
            "per_dataset": per_dataset,
        },
        "datapoints": {
            "total": totals["rows"],
            "by_scope": scope_rows,
            "distinct_proteins_summed_over_datasets": totals["proteins"],
            "counting_rule": (
                "One datapoint is one row of the published CSV. For the "
                "residue-level tasks a row is one labelled protein, not one "
                "annotated residue."
            ),
        },
        "protein_language_models": {"total": len(models), "names": models},
        "baselines": {
            "total": len(BASELINES),
            "combinations": [
                {"pooling": pooling, "head": head} for pooling, head in BASELINES
            ],
            "note": (
                "Attention pooling is trained jointly with the head, so it pairs "
                "only with the two neural heads."
            ),
        },
        "evaluation_metrics": {
            "total": sum(len(v) for v in metrics.values()) + 1,
            "regression": metrics["regression"],
            "classification": metrics["classification"],
            "aggregate": ["bradley_terry_rating"],
        },
        "runs": {
            "total": sum(
                len(d["split_variants"]) * len(BASELINES) * len(models)
                for d in per_dataset.values()
            ),
            "formula": "datasets x split variants x baselines x embeddings",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=pathlib.Path,
                        default=_REPO / "benchmark_card.json")
    args = parser.parse_args()

    card = json.loads(args.card.read_text(encoding="utf-8"))
    card["statistics"] = build_statistics(_REPO)
    args.card.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    stats = card["statistics"]
    print(f"  datasets              {stats['datasets']['total']}"
          f"  {stats['datasets']['by_scope']}")
    print(f"  datapoints            {stats['datapoints']['total']:,}")
    print(f"  protein LMs           {stats['protein_language_models']['total']}")
    print(f"  baselines             {stats['baselines']['total']}")
    print(f"  evaluation metrics    {stats['evaluation_metrics']['total']}"
          f"  (incluye el rating)")
    print(f"  runs                  {stats['runs']['total']:,}")
    print(f"  -> {args.card}")


if __name__ == "__main__":
    main()
