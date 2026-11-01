#!/usr/bin/env python3
"""Summarize overlap and shape consistency across mean-pooled embedding files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from enzymado import paths_and_constants
from enzymado.data_loading.stores import normalize_sequence

_P = paths_and_constants.Paths
REPO_ROOT = _P.project_root
DEFAULT_EMBEDDINGS_JSON = REPO_ROOT / "embeddings_models_output_mean_pooled.json"
DEFAULT_OUTPUT_DIR = _P.results_dir / "embedding_intersection"


def inspect_embedding_file(
    path: Path, expected_dim: int
) -> tuple[set[str], int, int, list[tuple[str, tuple[int, ...]]]]:
    from safetensors import safe_open

    bad_shapes: list[tuple[str, tuple[int, ...]]] = []

    with safe_open(str(path), framework="numpy", device="cpu") as handle:
        all_keys = list(handle.keys())
        ids = set(all_keys)
        # Check shape of the first key — all tensors in a mean-pooled
        # file share the same dim. Read the shape from the header rather than
        # materializing the tensor: the stores are bfloat16, which numpy cannot
        # represent, so get_tensor() raises "data type 'bfloat16' not understood"
        # for every model and the whole intersection fails.
        if all_keys:
            shape = tuple(handle.get_slice(all_keys[0]).get_shape())
            if shape != (expected_dim,):
                bad_shapes.append((all_keys[0], shape))

    return ids, len(ids), len(bad_shapes), bad_shapes


def _read_index(path: Path) -> dict[str, str]:
    """Read a protein_pool.csv index as ``{sequence_id: sequence}``."""
    csv.field_size_limit(sys.maxsize)
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "sequence_id" not in fields or "protein_sequence" not in fields:
            raise SystemExit(f"{path} needs sequence_id and protein_sequence columns.")
        for row in reader:
            sid = row.get("sequence_id")
            if sid:
                mapping[sid] = normalize_sequence(row.get("protein_sequence", ""))
    return mapping


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pooled embedding safetensors files and build the common-ID "
            "intersection "
            "across all embedding models."
        )
    )
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=DEFAULT_EMBEDDINGS_JSON,
        help=(
            "Embedding metadata JSON with model names, safetensors paths, and "
            "dimensions. "
            f"Default: {DEFAULT_EMBEDDINGS_JSON}"
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=279_644,
        help="Expected number of sequence IDs per complete file. Default: 279644.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory for common_ids.txt, excluded_ids.txt, and report.tsv. Default: "
            f"{DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--current-index",
        type=Path,
        default=None,
        help=(
            "Freshly built protein_pool.csv listing the sequences the "
            "current datasets still use. When given, allowlist ids whose sequence no "
            "dataset uses any more are dropped. Ids are NOT compared directly: a "
            "rebuilt index may elect a different representative id for a sequence "
            "that is still in use, and dropping that id would silently remove "
            "training rows."
        ),
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "protein_pool.csv",
        help=(
            "Index that training resolves sequences through, used to map each "
            "allowlist id back to its sequence. Must be the index the embedding "
            "stores were keyed by."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Still write intersection outputs for the currently readable files even if "
            "some "
            "models are missing, partial, or have bad shapes."
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
        if not path.is_absolute():
            path = REPO_ROOT / path
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
            ids, n_ids, bad_shape_count, bad_shapes = inspect_embedding_file(
                path, expected_dim
            )
        except Exception as exc:  # noqa: BLE001 - report each unreadable file.
            statuses[model] = {
                "status": "ERROR",
                "path": str(path),
                "n_ids": 0,
                "expected_dim": expected_dim,
                "bad_shape_count": 0,
                "bad_shape_examples": [str(exc)],
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
        handle.write(
            "model\tstatus\tn_ids\texpected_count\texpected_dim\t"
            "path\tbad_shape_count\tbad_shape_examples\n"
        )
        for model in sorted(statuses):
            row = statuses[model]
            handle.write(
                f"{model}\t{row['status']}\t{row['n_ids']}\t"
                f"{args.expected_count}\t{row['expected_dim']}\t"
                f"{row['path']}\t{row['bad_shape_count']}\t"
                f"{row['bad_shape_examples']}\n"
            )

    for model in sorted(statuses):
        row = statuses[model]
        print(
            f"{model:24} {row['status']:10} "
            f"n={row['n_ids']}/{args.expected_count} dim={row['expected_dim']} "
            f"path={row['path']}"
        )
        if row["bad_shape_examples"]:
            print(f"  bad_shape_examples={row['bad_shape_examples']}")

    blocking_statuses = {"MISSING", "PARTIAL", "BAD_SHAPE", "ERROR"}
    blockers = [
        model for model, row in statuses.items() if row["status"] in blocking_statuses
    ]
    if blockers and not args.allow_incomplete:
        print("\nRefusing to write intersection because some models are not complete.")
        print("Blocking models:", ", ".join(sorted(blockers)))
        print(f"Report written to: {report_path}")
        return 1

    common_ids = common_ids or set()
    if args.current_index is not None:
        reference = _read_index(args.reference_index)
        in_use = set(_read_index(args.current_index).values())
        before = len(common_ids)
        common_ids = {i for i in common_ids if reference.get(i) in in_use}
        print(
            f"\nDropped ids whose sequence no dataset uses any more: "
            f"{before} -> {len(common_ids)} ({before - len(common_ids)} removed)"
        )
        print(f"  reference index: {args.reference_index}")
        print(f"  current index  : {args.current_index}")
    excluded_ids = union_ids - common_ids

    common_path = args.output_dir / "common_ids.txt"
    excluded_path = args.output_dir / "excluded_ids.txt"

    common_path.write_text(
        "\n".join(sorted(common_ids)) + ("\n" if common_ids else ""), encoding="utf-8"
    )
    excluded_path.write_text(
        "\n".join(sorted(excluded_ids)) + ("\n" if excluded_ids else ""),
        encoding="utf-8",
    )

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
