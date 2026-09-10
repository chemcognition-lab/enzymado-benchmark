#!/usr/bin/env python3
"""Convert a FireProtDB dump into a plain CSV for downstream processing."""

import argparse
import csv
import gzip
import os
from typing import TextIO


def open_maybe_gzip(path: str) -> TextIO:
    """Open a plain-text or gzip-compressed input file."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(line: str) -> str | None:
    """Detect the supported delimiter without changing fallback behavior."""
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def split_line(line: str, delimiter: str | None) -> list[str]:
    """Split one input row using the detected delimiter."""
    if delimiter:
        return next(csv.reader([line], delimiter=delimiter))
    return line.split()


def create_parser() -> argparse.ArgumentParser:
    """Create the FireProtDB conversion command-line parser."""
    parser = argparse.ArgumentParser(
        description="Convert FireProtDB TSV/CSV to CSV for dataframe use."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/raw/fireprotdb/fireprotdb_ec.tsv",
        help="Input FireProtDB file.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/processed/fireprotdb/fireprotdb_ec.csv",
        help="Output CSV path.",
    )
    return parser


def convert_fireprotdb(in_path: str, out_path: str) -> None:
    """Convert FireProtDB rows to CSV while preserving values and row order."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with (
        open_maybe_gzip(in_path) as fh,
        open(out_path, "w", encoding="utf-8", newline="") as out_fh,
    ):
        writer = csv.writer(out_fh)

        header_line = fh.readline()
        if not header_line:
            raise RuntimeError("Input file is empty.")

        header_line = header_line.rstrip("\n")
        delimiter = detect_delimiter(header_line)
        header = split_line(header_line, delimiter)
        writer.writerow(header)

        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            row = split_line(line, delimiter)
            writer.writerow(row)


def main() -> None:
    """Run the FireProtDB conversion command-line interface."""
    args = create_parser().parse_args()
    convert_fireprotdb(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
