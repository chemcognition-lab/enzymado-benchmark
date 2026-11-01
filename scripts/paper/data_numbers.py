#!/usr/bin/env python
"""Dataset numbers of the paper, from data/processed and the embedding intersection.

Prints the sequence index and the sequences left out of every run, the property
tables (rows, proteins, lengths, random split sizes) on the rows every PLM
embedded, the rows the exclusion removed, the UniProt fragments, class balance,
the EC label structure, and how related the sequences of each real-case dataset
are.

Example:
    python scripts/paper/data_numbers.py
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path
import sys
import time

import pandas as pd

from enzymado import paths_and_constants
from enzymado.data_loading import data_registry, labels

_P = paths_and_constants.Paths
sys.path.insert(0, str(_P.scripts_dir / "data_prep"))

import remove_uniprot_fragments  # noqa: E402

csv.field_size_limit(sys.maxsize)
PROPERTIES = (
    "kcat", "km", "ph_opt", "dtm", "ddg", "binding",
    "ec_number", "active_sites", "binding_site", "solubility",
)  # fmt: skip
FRAGMENT_TASKS = ("binding", "ec_number", "kcat", "km", "ph_opt", "solubility")
RESIDUE_TASKS = ("active_sites", "binding_site")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-ids",
        type=Path,
        default=_P.results_dir / "embedding_intersection" / "common_ids.txt",
    )
    return parser.parse_args()


def used_sequences(common_ids: Path) -> tuple[pd.DataFrame, set[str]]:
    """Return the sequence index flagged by use, and the set of used sequences."""
    pool = pd.read_csv(_P.data_processed_dir / "protein_pool.csv").copy()
    pool["seq"] = pool.protein_sequence.str.strip().str.upper()
    pool["used"] = pool.sequence_id.isin(set(common_ids.read_text().split()))
    return pool, set(pool.seq[pool.used])


def read_task(
    task: str, split: data_registry.SplitType, used: set[str]
) -> pd.DataFrame:
    df = pd.read_csv(
        data_registry.TASK_REGISTRY[task].dataset_paths[split], low_memory=False
    ).copy()
    df["seq"] = df.protein_sequence.str.strip().str.upper()
    df["used"] = df.seq.isin(used)
    return df


def table_row(task: str, df: pd.DataFrame) -> dict:
    lengths = pd.Series(df.seq.unique()).str.len()
    sizes = df.split.value_counts()
    return dict(
        task=task,
        rows=len(df),
        proteins=df.seq.nunique(),
        min=lengths.min(),
        median=lengths.median(),
        max=lengths.max(),
        train=sizes["train"],
        val=sizes["val"],
        test=sizes["test"],
    )


def print_balance(task: str, df: pd.DataFrame) -> None:
    column = data_registry.TASK_REGISTRY[task].target_column
    if task in RESIDUE_TASKS:
        positives = df[column].map(lambda s: len(json.loads(s))).sum()
        residues = df.seq.str.len().sum()
        print(
            f"{task}: {positives:,} of {residues:,} residues annotated "
            f"({positives / residues:.4%})"
        )
        return
    print(f"{task}: {df[column].mean():.4f} positive")


def print_ec_labels(random: pd.DataFrame, similarity: pd.DataFrame) -> None:
    """EC vocabulary as training fits it: on every train row, before the exclusion."""
    builder = labels.HierarchicalECLabelBuilder()
    for name, df in (("random", random), ("similarity", similarity)):
        for base, train in (
            ("all train rows", df[df.split == "train"]),
            ("used train rows", df[df.used & (df.split == "train")]),
        ):
            builder.fit([{"ec_number": s} for s in train.ec_number.fillna("")])
            sizes = [len(v) for v in builder.vocabulary_by_level.values()]
            print(f"EC vocabulary, {name}, {base}: {sizes}, flat {sum(sizes):,}")
    per_enzyme = random.ec_number.fillna("").map(
        lambda s: len(builder.parse_labels({"ec_number": s}))
    )
    for base, n in (("all", per_enzyme), ("used", per_enzyme[random.used])):
        print(f"EC numbers per enzyme, {base} rows: mean {n.mean():.3f}, max {n.max()}")


def print_fragments() -> None:
    removed = {}
    for task in FRAGMENT_TASKS:
        path = data_registry.TASK_REGISTRY[task].dataset_paths[
            data_registry.SplitType.RANDOM
        ]
        meta_path = next(path.parent.glob("*.splits.metadata.json"))
        meta = json.loads(meta_path.read_text())
        removed[task] = sum(
            meta["modes"]["random"]["uniprot_fragments_removed"].values()
        )
    fragments = pd.read_csv(_P.data_processed_dir / "uniprot_fragments.csv")
    accessions = {
        a
        for sequence_id, _ in remove_uniprot_fragments.read_pool()
        if (a := remove_uniprot_fragments.base_accession(sequence_id))
    }
    print(
        f"fragment rows removed (random protocol): {sum(removed.values()):,} {removed}"
    )
    print(
        f"fragment accessions: {len(fragments):,} of {len(accessions):,} in the index"
    )


def print_real_case_relatedness() -> None:
    """Similarity of each real-case sequence to the dataset's most common one."""
    for task, config in data_registry.TASK_REGISTRY.items():
        if config.task_scope != "real_case":
            continue
        df = pd.read_csv(config.dataset_paths[data_registry.SplitType.RANDOM])
        counts = df.protein_sequence.value_counts()
        ref = counts.index[0]
        ratios = pd.Series(
            [
                difflib.SequenceMatcher(None, ref, s, autojunk=False).ratio()
                for s in counts.index
            ]
        )
        lengths = counts.index.str.len()
        print(
            f"{task:28s} rows {len(df):5d} sequences {len(counts):5d} "
            f"length {lengths.min()}-{lengths.max()} "
            f"similarity to the most common: min {ratios.min():.2f}, "
            f"median {ratios.median():.2f}"
        )


def main() -> None:
    args = parse_args()
    started = time.time()
    pool, used = used_sequences(args.common_ids)
    left = pool[~pool.used]
    print(
        f"index {len(pool):,} sequences ({pool.seq.nunique():,} unique), "
        f"used {pool.used.sum():,}, left out {len(left):,} "
        f"({left.seq.str.len().min():,} to {left.seq.str.len().max():,} residues)"
    )

    rows, removed = [], {}
    for task in PROPERTIES:
        random = read_task(task, data_registry.SplitType.RANDOM, used)
        similarity = read_task(task, data_registry.SplitType.SIMILARITY, used)
        rows.append(table_row(task, random[random.used]))
        removed[task] = int((~random.used).sum())
        if task in ("binding", "solubility", *RESIDUE_TASKS):
            print_balance(task, random[random.used])
        if task == "ec_number":
            print_ec_labels(random, similarity)

    print("\nProperty table and random split sizes, on used rows:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("rows removed with the left-out sequences:", removed)
    print()
    print_fragments()
    print("\nReal-case datasets (difflib similarity):")
    print_real_case_relatedness()
    print(f"\nelapsed {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
