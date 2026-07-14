"""Render the RAG knowledge-base cards.

Two layers are merged per card:

* the **auto layer** — numbers/params/rankings computed by :mod:`sources` from the
  repo's live artifacts (regenerated whenever results change), and
* the **authored layer** — web-verified prose + citations stored in
  ``content/<card_id>.json`` (never overwritten by this script).

Run via the wrapper ``scripts/generate_cards.py`` (which puts ``src`` on the
path), or ``python -m llm.render`` with ``src`` importable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from . import sources as S

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
CONTENT_DIR = PACKAGE_DIR / "content"
DEFAULT_OUTPUT = S.REPO_ROOT / "markdown"

SPLIT_ID_BY_FAMILY = {"random": "random", "loco": "similarity"}

TRACK_ORDER = [
    ("no_attention", "random", "No-attention · Random"),
    ("no_attention", "loco", "No-attention · Similarity"),
    ("attention", "random", "Attention · Random"),
    ("attention", "loco", "Attention · Similarity"),
]

LEVEL_BY_BATCH_KIND = {
    "protein_global": ("protein-level (global)", "protein-global"),
    "protein_ligand_global": ("protein + ligand (global)", "protein-ligand"),
    "mutation_delta_global": ("mutation effect (Δ vs wild-type)", "mutation-delta"),
    "protein_residue": ("per-residue", "residue"),
}

FEATURIZATION_BY_BATCH_KIND = {
    "protein_global": (
        "The protein embedding (mean-pooled residues for no-attention heads, or "
        "attention-pooled for attention heads) is fed directly to the head."
    ),
    "protein_ligand_global": (
        "The protein embedding is concatenated with an 8-dim RDKit descriptor "
        "vector computed from the ligand SMILES (MolWt, MolLogP, TPSA, HBD, HBA, "
        "rotatable bonds, rings, heavy atoms)."
    ),
    "mutation_delta_global": (
        "The feature is the mutant-minus-wild-type embedding delta (mutant and "
        "wild-type sequences must be equal length)."
    ),
    "protein_residue": (
        "The full per-residue embedding tensor (length × dim) is retained; labels "
        "are a binary vector over residue positions."
    ),
}


# ---------------------------------------------------------------------------
# Content sidecar
# ---------------------------------------------------------------------------

def load_content(card_id: str) -> dict[str, Any]:
    path = CONTENT_DIR / f"{card_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Per-card-type context builders
# ---------------------------------------------------------------------------

def _standing_rows(standing: dict, emb: str) -> list[dict]:
    rows = []
    buckets = standing.get(emb, {})
    for emode, sfam, label in TRACK_ORDER:
        cell = buckets.get((emode, sfam))
        if cell:
            rows.append({"track": label, "mean_rank": cell["mean_rank"], "n": cell["n"]})
    return rows


def build_model_cards(ctx) -> list[tuple[str, str]]:
    template = ctx.env.get_template("model_card.md.j2")
    out = []
    for emb, facts in ctx.models.items():
        card_id = f"model_{emb.lower()}"
        best = S.best_by_task_for_model(ctx.records, emb)
        rendered = template.render(
            card_id=card_id,
            emb=emb,
            dim=facts["dim"],
            shard_path=facts["shard_path"],
            n_models=ctx.n_models,
            standing_rows=_standing_rows(ctx.standing, emb),
            best_random=best.get("random"),
            best_similarity=best.get("loco"),
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def _param_variants(raw_entries, embedding_mode, model_family) -> list[dict]:
    # Use the raw entries (not the metric-filtered records) so configured
    # regimes with incomplete runs (e.g. per-residue XGBoost) still appear.
    seen: dict[tuple, dict] = {}
    for rec in raw_entries:
        if rec.get("embedding_mode") != embedding_mode or rec.get("model_family") != model_family:
            continue
        params = rec.get("params") or {}
        model_name = rec.get("model_name")
        key = (model_name, json.dumps(params, sort_keys=True))
        entry = seen.setdefault(
            key, {"model_name": model_name, "params": params, "tasks": set()}
        )
        entry["tasks"].add(rec.get("task_name"))
    variants = []
    for entry in seen.values():
        tasks = entry["tasks"]
        if "ec_number" in tasks:
            note = "EC multi-label (multi-head) variant"
        elif tasks & {"active_sites", "binding_site"}:
            note = "per-residue variant"
        else:
            note = "standard (global) variant"
        variants.append({
            "model_name": entry["model_name"],
            "note": note,
            "example_task": S._display_task(sorted(tasks)[0]),
            "params_json": json.dumps(entry["params"], indent=2),
        })
    variants.sort(key=lambda v: (0 if v["note"].startswith("standard") else 1, v["model_name"]))
    return variants


def build_baseline_cards(ctx) -> list[tuple[str, str]]:
    template = ctx.env.get_template("baseline_card.md.j2")
    out = []
    for base in S.BASELINES:
        card_id = f"baseline_{base['id']}"
        table = S.baseline_task_table(ctx.records, base["embedding_mode"], base["model_family"])
        rendered = template.render(
            card_id=card_id,
            baseline_id=base["id"],
            title=base["title"],
            embedding_mode=base["embedding_mode"],
            model_family=base["model_family"],
            n_models=ctx.n_models,
            param_variants=_param_variants(ctx.raw_entries, base["embedding_mode"], base["model_family"]),
            table_random=table.get("random"),
            table_similarity=table.get("loco"),
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def build_split_cards(ctx) -> list[tuple[str, str]]:
    template = ctx.env.get_template("split_card.md.j2")
    gap_rows = S.split_gap_table(ctx.records)
    out = []
    for split in S.SPLITS:
        card_id = f"split_{split['id']}"
        rendered = template.render(
            card_id=card_id,
            split_id=split["id"],
            split_family=split["split_family"],
            title=split["title"],
            gap_rows=gap_rows,
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


def _task_stats(raw: dict | None) -> dict | None:
    if not raw:
        return None
    stats_block = raw.get("statistics", {}) or {}
    add = raw.get("additional_important", {}) or {}
    out: dict[str, Any] = {
        "datapoints": (raw.get("amount_of_removed_data") or {}).get("datapoints_after_cleaning"),
    }

    label_dist = stats_block.get("label_distribution", {}) or {}
    classes = (label_dist.get("overall") or {}).get("classes")
    if classes:
        out["label_balance"] = [
            {"label": k, "count": v.get("count"), "ratio": v.get("ratio")}
            for k, v in classes.items()
        ]

    methods: dict[str, dict] = {}
    for split_name in ("train", "validation", "test"):
        for method_key, block in (label_dist.get(split_name) or {}).items():
            slot = "val" if split_name == "validation" else split_name
            methods.setdefault(method_key, {})[slot] = block.get("total_valid_labels")
    split_sizes = []
    for method_key, sizes in methods.items():
        label = "random" if "random" in method_key else (
            "similarity" if "similarity" in method_key else method_key)
        split_sizes.append({"method": label, **sizes})
    split_sizes.sort(key=lambda s: 0 if s["method"] == "random" else 1)
    if split_sizes:
        out["split_sizes_str"] = "; ".join(
            f"{s['method']} → train {s.get('train')} / val {s.get('val')} / test {s.get('test')}"
            for s in split_sizes
        )

    if out.get("label_balance"):
        out["label_balance_str"] = ", ".join(
            f"`{c['label']}`={c['count']} ({c['ratio']})" for c in out["label_balance"]
        )

    seq_len = add.get("sequence_length_summary")
    if seq_len:
        out["seq_len_str"] = (
            f"min {seq_len.get('min')}, median {seq_len.get('median')}, "
            f"mean {seq_len.get('mean')}, max {seq_len.get('max')} "
            f"({seq_len.get('count_greater_than_1024')} > 1024 residues)"
        )

    cluster = stats_block.get("entire_protein_pool_cluster_with_mmseqs2")
    if cluster:
        params = cluster.get("cluster_parameters") or {}
        out["clustering_str"] = (
            f"{cluster.get('number_of_clusters')} clusters from "
            f"{cluster.get('total_sequences')} sequences at "
            f"{params.get('min_sequence_identity')} identity "
            f"({cluster.get('singleton_clusters')} singletons)"
        )
    return out


def build_task_cards(ctx) -> list[tuple[str, str]]:
    template = ctx.env.get_template("task_card.md.j2")
    out = []
    for task_name, cfg in ctx.registry.items():
        card_id = f"task_{task_name}"
        results_name = S.results_task_name(task_name)
        entry = S.benchmark_entry_for(ctx.bench, task_name) or {}
        dataset0 = (entry.get("datasets") or [{}])[0]
        task_type = entry.get("task_type") or (
            "classification" if str(cfg.label_kind) != "scalar_regression" else "regression")
        level, level_slug = LEVEL_BY_BATCH_KIND.get(str(cfg.batch_kind), (str(cfg.batch_kind), "other"))

        leaderboard_raw = S.task_leaderboard(ctx.records, results_name)
        leaderboards = {
            SPLIT_ID_BY_FAMILY.get(sfam, sfam): rows
            for sfam, rows in sorted(leaderboard_raw.items(),
                                     key=lambda kv: 0 if kv[0] == "random" else 1)
        }

        rendered = template.render(
            card_id=card_id,
            task_name=task_name,
            results_name=results_name,
            task_scope=cfg.task_scope,
            task_type=task_type,
            level=level,
            level_slug=level_slug,
            label_kind=str(cfg.label_kind),
            target_column=cfg.target_column,
            label_units=dataset0.get("label_units"),
            primary_metric=ctx.primary_metric_for(task_type),
            splits_available=[str(k) for k in cfg.dataset_paths],
            ligand_column=cfg.ligand_column,
            wildtype_column=cfg.wildtype_sequence_column,
            residue_positions_column=cfg.residue_positions_column,
            description=entry.get("description"),
            source_name=(dataset0.get("provenance") or {}).get("source_name"),
            featurization_default=FEATURIZATION_BY_BATCH_KIND.get(str(cfg.batch_kind), ""),
            stats=_task_stats(S.dataset_stats_for(cfg)),
            leaderboards=leaderboards,
            content=load_content(card_id),
        )
        out.append((f"{card_id}.md", rendered))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class _Context:
    def __init__(self):
        self.models = S.load_models()
        self.registry = S.load_task_registry()
        self.bench = S.load_benchmark_card()
        self.records = S.load_records()
        self.raw_entries = list(S.iter_raw_entries())
        self.standing = S.model_standing(self.records)
        self.primary_metric_for, _ = S._metric_conventions()
        self.n_models = len(self.models)
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )


def _normalize_md(text: str) -> str:
    """Tidy Jinja whitespace: collapse blank runs, ensure a blank line before headings."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([^\n])\n(#{1,6} )", r"\1\n\n\2", text)
    if not text.endswith("\n"):
        text += "\n"
    return text


def build_all() -> list[tuple[str, str]]:
    ctx = _Context()
    cards: list[tuple[str, str]] = []
    cards += build_model_cards(ctx)
    cards += build_split_cards(ctx)
    cards += build_baseline_cards(ctx)
    cards += build_task_cards(ctx)
    return [(name, _normalize_md(text)) for name, text in cards]


def render_all(output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, text in build_all():
        (output_dir / filename).write_text(text)
        written.append(filename)
    return written


def card_ids() -> list[str]:
    """All expected card ids (without .md) — useful for scaffolding/tests."""
    ids = [f"model_{k.lower()}" for k in S.load_models()]
    ids += [f"split_{s['id']}" for s in S.SPLITS]
    ids += [f"baseline_{b['id']}" for b in S.BASELINES]
    ids += [f"task_{t}" for t in S.load_task_registry()]
    return ids


def scaffold_content() -> list[str]:
    """Write empty content stubs for any card_id missing one. Non-destructive."""
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for cid in card_ids():
        path = CONTENT_DIR / f"{cid}.json"
        if path.exists():
            continue
        stub = {"summary": "", "when_to_use": "", "caveats": "", "citations": []}
        path.write_text(json.dumps(stub, indent=2) + "\n")
        created.append(cid)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate enzymado RAG knowledge-base cards.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--scaffold", action="store_true",
                        help="Write empty content/*.json stubs for any missing cards, then exit.")
    args = parser.parse_args(argv)

    if args.scaffold:
        created = scaffold_content()
        print(f"Scaffolded {len(created)} content stub(s): {', '.join(created) or '(none missing)'}")
        return 0

    written = render_all(args.output)
    print(f"Wrote {len(written)} cards to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
