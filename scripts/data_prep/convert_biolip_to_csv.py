#!/usr/bin/env python3
"""Convert the BioLiP text export into a CSV with a stable header row."""

import argparse
import csv
import gzip
import os
from typing import TextIO

DEFAULT_HEADER = (
    "pdb_id",
    "chain_id",
    "resolution",
    "binding_site_id",
    "ligand_id",
    "ligand_chain_id",
    "ligand_serial_number",
    "binding_site_residues",
    "nearby_residues",
    "ligand_contact_residues",
    "catalytic_site_residues",
    "ec_number",
    "go_terms",
    "binding_affinity_1",
    "binding_affinity_2",
    "binding_affinity_3",
    "binding_affinity_4",
    "uniprot_id",
    "pubmed_id",
    "binding_site_range",
    "protein_sequence",
)


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
    """Create the BioLiP conversion command-line parser."""
    parser = argparse.ArgumentParser(
        description="Convert BioLiP tabular file to CSV for dataframe use."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/raw/biolip/BioLiP_nr.txt.gz",
        help="Input BioLiP file (.gz).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/processed/biolip/BioLiP_nr.csv",
        help="Output CSV path.",
    )
    return parser


def convert_biolip(in_path: str, out_path: str) -> None:
    """Convert BioLiP rows to CSV while preserving their values and order."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with (
        open_maybe_gzip(in_path) as fh,
        open(out_path, "w", encoding="utf-8", newline="") as out_fh,
    ):
        writer = csv.writer(out_fh)

        header = None
        delimiter = None
        first_data_row = None

        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                header_line = line.lstrip("#").strip()
                if header_line:
                    delimiter = detect_delimiter(header_line)
                    header = split_line(header_line, delimiter)
                continue

            delimiter = detect_delimiter(line)
            first_data_row = split_line(line, delimiter)
            break

        if first_data_row is None:
            raise RuntimeError("No data rows found in BioLiP file.")

        if header is None:
            if len(first_data_row) != len(DEFAULT_HEADER):
                raise RuntimeError(
                    "BioLiP column count does not match expected schema."
                )
            header = DEFAULT_HEADER

        writer.writerow(header)
        writer.writerow(first_data_row)

        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            row = split_line(line, delimiter)
            writer.writerow(row)


def main() -> None:
    """Run the BioLiP conversion command-line interface."""
    args = create_parser().parse_args()
    convert_biolip(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
