#!/usr/bin/env python3
from __future__ import annotations
"""Summarize overlap and shape consistency across generated embedding files."""

import argparse
import json
import sys
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_JSON = REPO_ROOT / "embeddings_models_output_mean_pooled.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "embedding_intersection"


def inspect_embedding_file(path: Path, expected_dim: int) -> tuple[set[str], int, int, list[tuple[str, tuple[int, ...]]]]:
    ids: set[str] = set()
    bad_shapes: list[tuple[str, tuple[int, ...]]] = []

    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            ids.add(key)
            shape = tuple(h5[key].shape)
            if shape != (expected_dim,) and len(bad_shapes) < 5:
                bad_shapes.append((key, shape))

    return ids, len(ids), len(bad_shapes), bad_shapes


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pooled embedding H5 files and build the common-ID intersection "
            "across all embedding models."
        )
    )
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=DEFAULT_EMBEDDINGS_JSON,
        help=(
            "Embedding metadata JSON with model names, H5 paths, and dimensions. "
            f"Default: {DEFAULT_EMBEDDINGS_JSON}"
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=279_644,
        help="Expected number of sequence IDs per complete H5 file. Default: 279644.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for common_ids.txt, excluded_ids.txt, and report.tsv. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Still write intersection outputs for the currently readable files even if some "
            "models are missing, locked, partial, or have bad shapes."
        ),
    )
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    with args.embeddings_json.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    statuses: dict[str, dict[str, object]] = {}
    common_ids: set[str] | None = None
    union_ids: set[str] = set()

    for model in sorted(meta):
        info = meta[model]
        path = Path(info["path"])
        expected_dim = int(info["dim"])

        if not path.exists():
            statuses[model] = {
                "status": "MISSING",
                "path": str(path),
                "n_ids": 0,
                "expected_dim": expected_dim,
                "bad_shape_count": 0,
                "bad_shape_examples": [],
            }
            continue

        try:
            ids, n_ids, bad_shape_count, bad_shapes = inspect_embedding_file(path, expected_dim)
        except BlockingIOError:
            statuses[model] = {
                "status": "LOCKED",
                "path": str(path),
                "n_ids": 0,
                "expected_dim": expected_dim,
                "bad_shape_count": 0,
                "bad_shape_examples": [],
            }
            continue

        count_ok = n_ids == args.expected_count
        shape_ok = bad_shape_count == 0
        if count_ok and shape_ok:
            status = "COMPLETE"
        elif not count_ok and shape_ok:
            status = "PARTIAL"
        else:
            status = "BAD_SHAPE"

        statuses[model] = {
            "status": status,
            "path": str(path),
            "n_ids": n_ids,
            "expected_dim": expected_dim,
            "bad_shape_count": bad_shape_count,
            "bad_shape_examples": bad_shapes,
        }

        union_ids.update(ids)
        common_ids = ids if common_ids is None else common_ids & ids

    report_path = args.output_dir / "report.tsv"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("model\tstatus\tn_ids\texpected_count\texpected_dim\tpath\tbad_shape_count\tbad_shape_examples\n")
        for model in sorted(statuses):
            row = statuses[model]
            handle.write(
                f"{model}\t{row['status']}\t{row['n_ids']}\t{args.expected_count}\t{row['expected_dim']}\t"
                f"{row['path']}\t{row['bad_shape_count']}\t{row['bad_shape_examples']}\n"
            )

    for model in sorted(statuses):
        row = statuses[model]
        print(
            f"{model:24} {row['status']:10} "
            f"n={row['n_ids']}/{args.expected_count} dim={row['expected_dim']} path={row['path']}"
        )
        if row["bad_shape_examples"]:
            print(f"  bad_shape_examples={row['bad_shape_examples']}")

    blocking_statuses = {"MISSING", "LOCKED", "PARTIAL", "BAD_SHAPE"}
    blockers = [model for model, row in statuses.items() if row["status"] in blocking_statuses]
    if blockers and not args.allow_incomplete:
        print("\nRefusing to write intersection because some models are not complete.")
        print("Blocking models:", ", ".join(sorted(blockers)))
        print(f"Report written to: {report_path}")
        return 1

    common_ids = common_ids or set()
    excluded_ids = union_ids - common_ids

    common_path = args.output_dir / "common_ids.txt"
    excluded_path = args.output_dir / "excluded_ids.txt"

    common_path.write_text("\n".join(sorted(common_ids)) + ("\n" if common_ids else ""), encoding="utf-8")
    excluded_path.write_text("\n".join(sorted(excluded_ids)) + ("\n" if excluded_ids else ""), encoding="utf-8")

    print()
    print(f"Union size: {len(union_ids)}")
    print(f"Common size: {len(common_ids)}")
    print(f"Excluded size: {len(excluded_ids)}")
    print(f"Wrote: {common_path}")
    print(f"Wrote: {excluded_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
