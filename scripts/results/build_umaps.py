#!/usr/bin/env python
"""Compute the 2D UMAP of each dataset for the benchmark website.

One UMAP per dataset and embedding, with both splits stored as columns.
Only proteins present in all 14 embeddings are used.

Examples:
    python scripts/results/build_umaps.py --dataset kcat --embedding esmc_300m
    python scripts/results/build_umaps.py --dataset kcat
    python scripts/results/build_umaps.py --pool
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import safetensors

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado.data_loading import data_registry  # noqa: E402
from enzymado.data_loading.stores import load_sequence_to_id_index  # noqa: E402
from enzymado.data_loading.task_csv import normalize_sequence  # noqa: E402

POOLED_DIR = _REPO / "data" / "embeddings_mean_pooled" / "protein_sequences_unique"
SEQUENCE_INDEX = _REPO / "data" / "processed" / "protein_pool.csv"
COMMON_IDS = _REPO / "results" / "embedding_intersection" / "common_ids.txt"
OUT_DIR = _REPO / "results" / "figures" / "umaps"


def embedding_names() -> list[str]:
    return sorted(path.stem for path in POOLED_DIR.glob("*.safetensors"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display_label(rows: list[dict[str, str]], config) -> pd.Series:
    """Label used to colour the points: the raw value, or the site count for residue tasks."""
    if str(config.label_kind) == "residue_binary_vector":
        return pd.Series([
            len([p for p in re.split(r"[\[\],;\s]+", row.get(config.target_column) or "")
                 if p.strip()])
            for row in rows
        ], dtype="float64")
    column = config.target_column.removesuffix("_standardized")
    if column not in rows[0]:
        column = config.target_column
    values = pd.Series([(row.get(column) or "").strip() for row in rows])
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return numeric
    return values


def split_by_sequence(rows: list[dict[str, str]], config, name: str) -> dict[str, str]:
    """Map each sequence to its split."""
    mapping: dict[str, str] = {}
    for row in rows:
        seq = normalize_sequence(row.get(config.sequence_column, ""))
        split = (row.get(config.split_column) or "").strip()
        previous = mapping.setdefault(seq, split)
        if previous != split:
            raise ValueError(f"{name}: a sequence sits in both {previous} and {split}")
    return mapping


def load_vectors(embedding: str, ids: list[str]) -> np.ndarray:
    """Load the pooled embeddings for the given ids as float32."""
    path = POOLED_DIR / f"{embedding}.safetensors"
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return np.stack([handle.get_tensor(i).float().numpy() for i in ids]).astype(np.float32)


def write_umap(frame: pd.DataFrame, target: Path) -> None:
    """Save a compact parquet: the website Space has a 1 GB storage limit."""
    frame = frame.drop(columns=["sequence_id"], errors="ignore")
    for column in ("umap_1", "umap_2"):
        frame[column] = frame[column].round(3).astype("float32")
    frame["seq_len"] = frame["seq_len"].astype("int32")
    for column in ("split", "split_similarity"):
        if column in frame:
            frame[column] = frame[column].astype("category")
    staging = target.with_suffix(".parquet.tmp")
    frame.to_parquet(staging, index=False, compression="zstd", compression_level=19)
    staging.replace(target)


def project(frame: pd.DataFrame, embeddings: list[str], out_dir: Path, stem: str) -> None:
    """Fit one UMAP per embedding on the distinct sequences and write each layout."""
    import umap  # slow to import, so only when needed

    ids = sorted(frame["sequence_id"].unique())
    out_dir.mkdir(parents=True, exist_ok=True)
    for embedding in embeddings:
        start = time.time()
        vectors = load_vectors(embedding, ids)
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(vectors)
        where = dict(zip(ids, coords))
        out = frame.copy()
        out["umap_1"] = [float(where[i][0]) for i in out["sequence_id"]]
        out["umap_2"] = [float(where[i][1]) for i in out["sequence_id"]]
        target = out_dir / f"{stem}_{embedding}.parquet"
        write_umap(out, target)
        print(f"  {embedding:<24} {vectors.shape[1]:>5} dims  {time.time() - start:7.1f}s"
              f"  -> {target.relative_to(_REPO)}", flush=True)


def pool_frame(seq_to_id: dict[str, str], common: set[str]) -> pd.DataFrame:
    """One row per protein in the benchmark, with the datasets it belongs to and its EC class."""
    datasets: dict[str, set[str]] = {}
    uniprot: dict[str, str] = {}
    length: dict[str, int] = {}
    ec_top: dict[str, set[str]] = {}
    for name, config in data_registry.TASK_REGISTRY.items():
        columns = [config.sequence_column]
        if getattr(config, "wildtype_sequence_column", None):
            columns.append(config.wildtype_sequence_column)
        for row in read_rows(sorted(config.dataset_paths.values())[0]):
            for column in columns:
                seq = normalize_sequence(row.get(column, ""))
                sid = seq_to_id.get(seq)
                if sid not in common:
                    continue
                datasets.setdefault(sid, set()).add(name)
                length[sid] = len(seq)
                if row.get("uniprot_id") and column == config.sequence_column:
                    uniprot.setdefault(sid, row["uniprot_id"].strip())
                if str(config.label_kind) == "multilabel_ec":
                    codes = (row.get(config.target_column) or "").replace(config.ec_separator, ";")
                    ec_top.setdefault(sid, set()).update(
                        c.strip().split(".")[0] for c in codes.split(";") if c.strip())

    ids = sorted(datasets)
    top = [ec_top.get(i, set()) - {"-", ""} for i in ids]
    return pd.DataFrame({
        "sequence_id": ids,
        "uniprot_id": [uniprot.get(i) for i in ids],
        "seq_len": [length[i] for i in ids],
        "datasets": [";".join(sorted(datasets[i])) for i in ids],
        "n_datasets": [len(datasets[i]) for i in ids],
        "ec_class": [next(iter(t)) if len(t) == 1 else ("multiple" if t else None) for t in top],
    })


def build(name: str, embeddings: list[str], seq_to_id: dict[str, str],
          common: set[str]) -> None:
    config = data_registry.TASK_REGISTRY[name]
    paths = sorted(config.dataset_paths.values())
    random_path = next((p for p in paths if "random" in p.stem), paths[0])
    rows = read_rows(random_path)

    frame = pd.DataFrame({
        "sequence_id": [seq_to_id.get(normalize_sequence(r.get(config.sequence_column, "")))
                        for r in rows],
        "uniprot_id": [(r.get("uniprot_id") or "").strip() or None for r in rows],
        "seq_len": [len(normalize_sequence(r.get(config.sequence_column, ""))) for r in rows],
        "label": display_label(rows, config),
        "split": [(r.get(config.split_column) or "").strip() for r in rows],
    })
    similarity = [p for p in paths if "similarity" in p.stem]
    if similarity:
        by_seq = split_by_sequence(read_rows(similarity[0]), config, name)
        frame["split_similarity"] = [
            by_seq[normalize_sequence(r.get(config.sequence_column, ""))] for r in rows
        ]

    kept = frame["sequence_id"].isin(common)
    frame = frame[kept].reset_index(drop=True)
    card_dir = random_path.parent.name   # the explorer keys files by this, e.g. ddg_mut
    print(f"{name}: {len(rows):,} rows, {int(kept.sum()):,} in the common set, "
          f"{frame['sequence_id'].nunique():,} distinct sequences", flush=True)
    project(frame, embeddings, OUT_DIR, card_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", choices=sorted(data_registry.TASK_REGISTRY))
    target.add_argument("--pool", action="store_true",
                        help="One UMAP of every protein in the benchmark.")
    parser.add_argument("--embedding", action="append",
                        help="Repeatable. Defaults to every pooled store.")
    args = parser.parse_args()

    embeddings = args.embedding or embedding_names()
    unknown = set(embeddings) - set(embedding_names())
    if unknown:
        raise SystemExit(f"no pooled store for: {sorted(unknown)}")
    seq_to_id = load_sequence_to_id_index(SEQUENCE_INDEX)
    common = set(COMMON_IDS.read_text().split())
    if args.pool:
        frame = pool_frame(seq_to_id, common)
        print(f"pool: {len(frame):,} proteins", flush=True)
        # Own folder, so the website sync does not ship it before there is a view for it.
        project(frame, embeddings, OUT_DIR / "pool", "pool")
    else:
        build(args.dataset, embeddings, seq_to_id, common)


if __name__ == "__main__":
    main()
