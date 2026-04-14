#!/usr/bin/env python3
from __future__ import annotations
"""Validate embedding H5 files before they are fed into training jobs."""

import argparse
import sys
from pathlib import Path

import h5py


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an embedding H5 by checking dataset count and optional vector dimension."
    )
    parser.add_argument("--h5", required=True, type=Path, help="Path to the H5 file to validate.")
    parser.add_argument(
        "--expected-count",
        required=True,
        type=int,
        help="Expected number of top-level datasets in the H5 file.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="Optional expected shape for each embedding vector, e.g. 1024 for shape (1024,).",
    )
    parser.add_argument(
        "--sample-bad-shapes",
        type=int,
        default=5,
        help="Maximum number of bad-shape examples to print. Default: 5.",
    )
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"ERROR: missing H5 file: {args.h5}", file=sys.stderr)
        return 1

    bad_shapes: list[tuple[str, tuple[int, ...]]] = []

    try:
        with h5py.File(args.h5, "r") as h5:
            n = len(h5)
            if args.expected_dim is not None:
                for key in h5.keys():
                    shape = tuple(h5[key].shape)
                    if shape != (args.expected_dim,):
                        if len(bad_shapes) < args.sample_bad_shapes:
                            bad_shapes.append((key, shape))
            print(f"H5: {args.h5}")
            print(f"Dataset count: {n}/{args.expected_count}")
            if args.expected_dim is not None:
                print(f"Expected dim: ({args.expected_dim},)")
    except BlockingIOError as exc:
        print(f"ERROR: H5 is locked or still being written: {args.h5} ({exc})", file=sys.stderr)
        return 1

    if n != args.expected_count:
        print("ERROR: dataset count does not match expected count.", file=sys.stderr)
        return 1

    if bad_shapes:
        print("ERROR: found embeddings with unexpected shapes:", file=sys.stderr)
        for key, shape in bad_shapes:
            print(f"  {key}: {shape}", file=sys.stderr)
        return 1

    print("Validation OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
