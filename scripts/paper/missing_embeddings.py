#!/usr/bin/env python
"""Sequences of the index that each PLM could not embed, and the LaTeX table of them.

Reads only safetensors headers: the mean-pooled store and every per-residue shard
of each PLM. A sequence counts as embedded when both stores hold it. Prints, per
PLM, the failures, the shortest failed and the longest embedded sequence, and
how many embedded sequences lie past the first failure; then how the failures
overlap and whether they are the sequences left out of every run.

Example:
    python scripts/paper/missing_embeddings.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

import pandas as pd
import safetensors

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
CAUSE = "GPU memory on long sequences"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_P.results_dir / "paper")
    return parser.parse_args()


def store_keys(path: Path) -> set[str]:
    with safetensors.safe_open(str(path), framework="numpy") as f:
        return set(f.keys())


def embedded(plm: str, pooled: dict, residue: dict) -> set[str]:
    pooled_keys = store_keys(_P.project_root / pooled[plm]["path"])
    shards = sorted(Path(residue[plm]["path"]).glob("*.safetensors"))
    residue_keys = set().union(*(store_keys(s) for s in shards))
    return pooled_keys & residue_keys


def summarize(plm: str, lengths: pd.Series, failed: set[str]) -> dict:
    """Failure counts of one PLM; the length bounds are NaN when nothing failed."""
    is_failed = lengths.index.isin(failed)
    first = lengths[is_failed].min()
    last = lengths[~is_failed].max()
    return dict(
        plm=plm,
        failed=int(is_failed.sum()),
        shortest_failed=first,
        longest_embedded=last,
        embedded_past_first_failure=int((~is_failed & (lengths > first)).sum()),
        failed_below_longest_embedded=int((is_failed & (lengths < last)).sum()),
    )


def latex_name(plm: str) -> str:
    return plm.replace("esm2_", "ESM-2_").replace("_", r"\_")


def latex_table(summary: pd.DataFrame) -> str:
    def number(v: float | None) -> str:
        return "n/a" if pd.isna(v) else f"{int(v):,}".replace(",", "{,}")

    lines = [
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Failed} & \textbf{Shortest failed} & "
        r"\textbf{Longest embedded} & \textbf{Probable cause} \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        cause = CAUSE if row.failed else "n/a"
        lines.append(
            f"{latex_name(row.plm)} & {row.failed} & {number(row.shortest_failed)} & "
            f"{number(row.longest_embedded)} & {cause} \\\\"
        )
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    args = parse_args()
    pooled = json.loads(
        (_P.project_root / "embeddings_models_output_mean_pooled.json").read_text()
    )
    residue = json.loads(
        (_P.project_root / "embeddings_models_output.json").read_text()
    )
    pool = pd.read_csv(_P.data_processed_dir / "protein_pool.csv")
    lengths = pool.set_index("sequence_id").protein_sequence.str.len()
    common = set(
        (_P.results_dir / "embedding_intersection" / "common_ids.txt")
        .read_text()
        .split()
    )

    failures, rows = {}, []
    for plm in pooled:
        started = time.time()
        failures[plm] = set(lengths.index) - embedded(plm, pooled, residue)
        rows.append(summarize(plm, lengths, failures[plm]))
        print(f"{plm}: {len(failures[plm])} failed ({time.time() - started:.0f} s)")
    summary = pd.DataFrame(rows).sort_values(["failed", "plm"], ascending=[False, True])
    print(summary.to_string(index=False))

    union = set().union(*failures.values())
    left_out = set(lengths.index) - common
    per_sequence = collections.Counter(s for f in failures.values() for s in f)
    print(
        f"\nfailed in at least one PLM: {len(union)}; "
        f"left out of every run: {len(left_out)}"
    )
    print(f"identical sets: {union == left_out}")
    print(
        "PLMs that fail each left-out sequence:",
        dict(sorted(collections.Counter(per_sequence.values()).items())),
    )
    largest = summary.plm.iloc[0]
    print(f"every failure is also a {largest} failure: {union <= failures[largest]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "missing_embeddings.csv", index=False)
    (args.out_dir / "missing_embeddings_table.tex").write_text(latex_table(summary))
    print(f"wrote {args.out_dir / 'missing_embeddings.csv'} and the LaTeX table")


if __name__ == "__main__":
    main()
