"""Render the assistant's knowledge-base cards from the results and the authored prose.

Each card merges two layers:

* the generated layer: numbers, settings and rankings that :mod:`sources` reads
  from the published results, so a card is current whenever it is rebuilt;
* the authored layer: verified prose and citations in ``content/<card_id>.json``,
  which this script never overwrites.

There are cards for the 14 PLMs and the no-PLM reference, the two splits, the five
baselines, the 18 tasks and the two leaderboards. They are written to
``results/llm_cards`` by default, which is not tracked: rebuild them after the
results change.

Example:
    python -m enzymado.llm --output results/llm_cards
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import re
import typing

import jinja2

from enzymado.llm import sources

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
CONTENT_DIR = PACKAGE_DIR / "content"
DEFAULT_OUTPUT = sources.REPO_ROOT / "results" / "llm_cards"

LEVEL_BY_BATCH_KIND = {
    "protein_global": ("protein level", "protein-global"),
    "protein_ligand_global": ("protein and ligand", "protein-ligand"),
    "mutation_delta_global": ("mutation effect (mutant minus wild type)", "mutation"),
    "protein_residue": ("per residue", "residue"),
}

FEATURIZATION_BY_BATCH_KIND = {
    "protein_global": (
        "The protein embedding is fed to the head: the mean over residues for mean "
        "pooling, or the learned attention pooling for the attention baselines."
    ),
    "protein_ligand_global": (
        "The protein embedding is concatenated with the 200 normalized RDKit 2D "
        "descriptors of the substrate (descriptastorus RDKit2DNormalized)."
    ),
    "mutation_delta_global": (
        "The feature is the mutant embedding minus the wild-type embedding. With "
        "mean pooling the pooled vectors are subtracted; with attention pooling the "
        "per-residue difference is pooled."
    ),
    "protein_residue": (
        "The full per-residue embedding (length x width) is kept and each residue "
        "gets a binary label."
    ),
}

USAGE = (
    "Embeddings are computed once from the frozen model's final hidden layer (the "
    "encoder for encoder-decoder models) over the deduplicated pool of sequences, "
    "with special tokens stripped so that position i is residue i, and stored in "
    "bfloat16. Sequences are never truncated: the 620 sequences of 2,645 residues "
    "or more that ESM3 could not embed are left out of every run, so all models "
    "see the same 269,409 sequences. Mean-pooling baselines average the residues; "
    "attention baselines learn the pooling (see the baseline cards)."
)
NOPLM_USAGE = (
    "No language model is involved. Each protein is described by 36 Biopython "
    "ProtParam descriptors (amino acid composition, secondary structure fractions, "
    "molecular weight, isoelectric point, GRAVY, instability index, aromaticity, "
    "extinction coefficients, net charge at pH 7.4 and flexibility statistics), "
    "and by a 20-letter one-hot per residue for attention pooling and the residue "
    "tasks. It runs the same five baselines as every PLM, which shows what the "
    "PLMs add."
)


def load_content(card_id: str) -> dict[str, typing.Any]:
    path = CONTENT_DIR / f"{card_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


class _Context:
    """Everything the card builders share, loaded once."""

    def __init__(self) -> None:
        self.models = sources.load_models()
        self.registry = sources.load_task_registry()
        self.bench = sources.load_benchmark_card()
        self.records = sources.load_records()
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )


def build_model_cards(ctx: _Context) -> list[tuple[str, str]]:
    template = ctx.env.get_template("model_card.md.j2")
    out = []
    for emb, facts in ctx.models.items():
        card_id = f"model_{emb.lower()}"
        best = sources.best_by_task_for_model(ctx.records, emb)
        rendered = template.render(
            card_id=card_id,
            emb=emb,
            facts=facts,
            params=sources.format_params(facts["params"]),
            usage=NOPLM_USAGE if emb == sources.NOPLM else USAGE,
            standing=sources.model_standing(emb),
            best_by_split={s: best[s] for s in ("random", "similarity") if s in best},
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def _variant_note(tasks: set[str]) -> str:
    if "ec_number" in tasks:
        return "EC number variant (one output group per EC level)"
    if tasks & {"active_sites", "binding_site"}:
        return "per-residue variant"
    return "whole-protein variant"


def _param_variants(records: list[dict], pool: str, head: str) -> list[dict]:
    """Distinct (model name, parameters) settings used by one baseline."""
    seen: dict[tuple, dict] = {}
    for r in records:
        if r["pool"] != pool or r["head"] != head:
            continue
        key = (r["model_name"], json.dumps(r["params"], sort_keys=True))
        entry = seen.setdefault(
            key, {"model_name": r["model_name"], "params": r["params"], "tasks": set()}
        )
        entry["tasks"].add(r["task"])
    variants = [
        {
            "model_name": v["model_name"],
            "note": _variant_note(v["tasks"]),
            "example_task": sorted(v["tasks"])[0],
            "params_json": json.dumps(v["params"], indent=2),
        }
        for v in seen.values()
    ]
    return sorted(
        variants,
        key=lambda v: (not v["note"].startswith("whole"), str(v["model_name"])),
    )


def build_baseline_cards(ctx: _Context) -> list[tuple[str, str]]:
    template = ctx.env.get_template("baseline_card.md.j2")
    n_plms = sum(emb != sources.NOPLM for emb in ctx.models)
    out = []
    for base in sources.BASELINES:
        card_id = f"baseline_{base['id']}"
        tables = sources.baseline_task_table(ctx.records, base["pool"], base["head"])
        rendered = template.render(
            card_id=card_id,
            baseline_id=base["id"],
            title=base["title"],
            pool=base["pool"],
            head=base["head"],
            n_plms=n_plms,
            param_variants=_param_variants(ctx.records, base["pool"], base["head"]),
            tables={s: tables[s] for s in ("random", "similarity") if s in tables},
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def build_split_cards(ctx: _Context) -> list[tuple[str, str]]:
    template = ctx.env.get_template("split_card.md.j2")
    gap_rows = sources.split_gap_table(ctx.records)
    out = []
    for split in sources.SPLITS:
        card_id = f"split_{split['id']}"
        rendered = template.render(
            card_id=card_id,
            split_id=split["id"],
            title=split["title"],
            gap_rows=gap_rows,
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def _split_label(method: str) -> str:
    return next((s for s in ("random", "similarity") if s in method), method)


def _task_stats(raw: dict | None) -> dict[str, typing.Any] | None:
    """The dataset-card statistics a task card shows, as display strings.

    Counts are the rows and proteins used in the runs, without the sequences that
    could not be embedded, as in the paper's dataset table.
    """
    if not raw:
        return None
    stats = raw.get("statistics") or {}
    used = stats.get("used_in_runs") or {}
    out: dict[str, typing.Any] = {
        "datapoints": used.get("rows", stats.get("rows")),
        "proteins": used.get("distinct_proteins", stats.get("distinct_proteins")),
    }
    sizes = stats.get("split_sizes_by_method") or {}
    if sizes:
        out["split_sizes_str"] = "; ".join(
            f"{_split_label(m)}: train {v.get('train')} / val {v.get('val')} / "
            f"test {v.get('test')}"
            for m, v in sorted(sizes.items(), key=lambda kv: _split_label(kv[0]))
        )
    labels = stats.get("label_distribution_by_method") or {}
    overall = next(iter(labels.values()), {}).get("overall") or {}
    if overall.get("kind") == "binary":
        out["label_balance_str"] = (
            f"{overall.get('positive')} positive of "
            f"{overall.get('total_valid_labels')} "
            f"({overall.get('positive_prevalence', 0):.1%})"
        )
    length = used.get("distinct_protein_length_summary") or stats.get(
        "distinct_protein_length_summary"
    )
    if length:
        out["seq_len_str"] = (
            f"min {length.get('min')}, median {length.get('median')}, "
            f"max {length.get('max')} residues"
        )
    cluster = stats.get("entire_protein_pool_cluster_with_mmseqs2")
    if cluster:
        params = cluster.get("cluster_parameters") or {}
        out["clustering_str"] = (
            f"{cluster.get('number_of_clusters')} clusters from "
            f"{cluster.get('total_sequences')} sequences at "
            f"{params.get('min_sequence_identity')} identity "
            f"({cluster.get('singleton_clusters')} singletons)"
        )
    return out


def build_task_cards(ctx: _Context) -> list[tuple[str, str]]:
    template = ctx.env.get_template("task_card.md.j2")
    out = []
    for task, cfg in ctx.registry.items():
        card_id = f"task_{task}"
        entry = sources.benchmark_entry_for(ctx.bench, task) or {}
        dataset = (entry.get("datasets") or [{}])[0]
        task_type = entry.get("task_type") or (
            "regression"
            if str(cfg.label_kind) == "scalar_regression"
            else "classification"
        )
        level, level_slug = LEVEL_BY_BATCH_KIND[str(cfg.batch_kind)]
        rendered = template.render(
            card_id=card_id,
            task_name=task,
            task_scope=cfg.task_scope,
            task_type=task_type,
            level=level,
            level_slug=level_slug,
            label_kind=str(cfg.label_kind),
            target_column=cfg.target_column,
            label_units=dataset.get("label_units"),
            primary_metric=sources.PRIMARY_METRIC[task_type],
            splits_available=[str(k) for k in cfg.dataset_paths],
            ligand_column=cfg.ligand_column,
            wildtype_column=cfg.wildtype_sequence_column,
            numeric_features=list(cfg.numeric_feature_columns),
            description=entry.get("description"),
            featurization_default=FEATURIZATION_BY_BATCH_KIND[str(cfg.batch_kind)],
            stats=_task_stats(sources.dataset_stats_for(cfg)),
            leaderboards=sources.task_leaderboard(ctx.records, task),
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def _weights_text(rating: dict, leaderboard: str) -> str:
    tasks = rating["task_weights"].get(leaderboard)
    splits = rating["split_weights"][leaderboard]
    split_text = ", ".join(f"{s} {w:g}" for s, w in splits.items())
    if not tasks:
        return f"Every dataset weighs the same; split weights: {split_text}."
    task_text = ", ".join(f"{t} {w:g}" for t, w in tasks.items())
    return f"Task weights: {task_text}. Split weights: {split_text}."


def build_leaderboard_cards(ctx: _Context) -> list[tuple[str, str]]:
    template = ctx.env.get_template("leaderboard_card.md.j2")
    rating = ctx.bench["rating"]
    metric_weights = "; ".join(
        f"{kind}: " + ", ".join(f"{m} {w:g}" for m, w in weights.items())
        for kind, weights in rating["metric_weights"].items()
    )
    out = []
    for leaderboard in sources.LEADERBOARDS:
        ratings = sources.load_ratings(leaderboard)
        rows = [
            {
                "rank": i,
                "embedding_model": r.embedding_model,
                "config": sources.config_label(r.model_family, r.pool_type),
                "rating": round(float(r.rating), 1),
                "ci_lo": round(float(r.ci_lo), 1),
                "ci_hi": round(float(r.ci_hi), 1),
                "se": round(float(r.bootstrap_se), 1),
            }
            for i, r in enumerate(ratings.itertuples(), start=1)
        ]
        name = "Property" if leaderboard == "properties" else "Real-case"
        rendered = template.render(
            card_id=f"leaderboard_{leaderboard}",
            leaderboard=leaderboard,
            title=f"{name} leaderboard ({len(rows)} competitors)",
            rating_description=(
                f"Bradley-Terry ratings of every representation and baseline on the "
                f"{name.lower()} datasets, best first."
            ),
            metric_weights=metric_weights,
            z=f"{rating['tie_margins']['z']:g}",
            weights=_weights_text(rating, leaderboard),
            rows=rows,
        )
        out.append((f"leaderboard_{leaderboard}.md", rendered))
    return out


def _normalize_md(text: str) -> str:
    """Collapse runs of blank lines and keep a blank line before each heading."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([^\n])\n(#{1,6} )", r"\1\n\n\2", text)
    return text if text.endswith("\n") else text + "\n"


def build_all() -> list[tuple[str, str]]:
    ctx = _Context()
    cards = (
        build_model_cards(ctx)
        + build_split_cards(ctx)
        + build_baseline_cards(ctx)
        + build_task_cards(ctx)
        + build_leaderboard_cards(ctx)
    )
    return [(name, _normalize_md(text)) for name, text in cards]


def render_all(output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, text in build_all():
        (output_dir / filename).write_text(text)
        written.append(filename)
    return written


@functools.lru_cache(maxsize=1)
def card_ids() -> list[str]:
    """Every authored card id (without ``.md``); leaderboard cards have no content."""
    ids = [f"model_{k.lower()}" for k in sources.load_models()]
    ids += [f"split_{s['id']}" for s in sources.SPLITS]
    ids += [f"baseline_{b['id']}" for b in sources.BASELINES]
    ids += [f"task_{t}" for t in sources.load_task_registry()]
    return ids


def scaffold_content() -> list[str]:
    """Write empty content stubs for card ids without one; never overwrites."""
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for card_id in card_ids():
        path = CONTENT_DIR / f"{card_id}.json"
        if path.exists():
            continue
        stub = {"summary": "", "when_to_use": "", "caveats": "", "citations": []}
        path.write_text(json.dumps(stub, indent=2) + "\n")
        created.append(card_id)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Write empty content/*.json stubs for missing cards, then exit.",
    )
    args = parser.parse_args(argv)
    if args.scaffold:
        created = scaffold_content()
        print(f"Scaffolded {len(created)} content stub(s): {', '.join(created)}")
        return 0
    written = render_all(args.output)
    print(f"Wrote {len(written)} cards to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
