#!/usr/bin/env python3
"""Convert a FireProtDB dump into a plain CSV for downstream processing."""

import argparse
import csv
import gzip
import os


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(line):
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def split_line(line, delimiter):
    if delimiter:
        return next(csv.reader([line], delimiter=delimiter))
    return line.split()


def main():
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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    with open_maybe_gzip(args.in_path) as fh, open(
        args.out_path, "w", encoding="utf-8", newline=""
    ) as out_fh:
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


if __name__ == "__main__":
    main()
