#!/usr/bin/env python3
"""Convert the BioLiP text export into a CSV with a stable header row."""

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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    default_header = [
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
    ]

    with open_maybe_gzip(args.in_path) as fh, open(
        args.out_path, "w", encoding="utf-8", newline=""
    ) as out_fh:
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
            if len(first_data_row) != len(default_header):
                raise RuntimeError(
                    "BioLiP column count does not match expected schema."
                )
            header = default_header

        writer.writerow(header)
        writer.writerow(first_data_row)

        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            row = split_line(line, delimiter)
            writer.writerow(row)


if __name__ == "__main__":
    main()
