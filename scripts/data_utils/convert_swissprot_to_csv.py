#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import re


EC_PATTERN = re.compile(r"EC=([0-9]+(?:\.[0-9-]+){3})")


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def parse_header(header):
    accession = ""
    entry_name = ""
    description = ""
    parts = header.split("|", 2)
    rest = header
    if len(parts) == 3:
        _, accession, rest = parts
        rest_parts = rest.split(" ", 1)
        entry_name = rest_parts[0]
        rest = rest_parts[1] if len(rest_parts) > 1 else ""
    if rest:
        os_idx = rest.find(" OS=")
        description = rest[:os_idx] if os_idx != -1 else rest
        description = description.strip()
    ecs = EC_PATTERN.findall(header)
    ec_value = ";".join(sorted(set(ecs)))
    return accession, entry_name, description, ec_value


def main():
    parser = argparse.ArgumentParser(
        description="Convert Swiss-Prot FASTA to CSV for dataframe use."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/raw/swissprot/swissprot_ec.fasta.gz",
        help="Input FASTA(.gz) path.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/processed/swissprot/swissprot_ec.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    if args.in_path.endswith(".tsv"):
        with open_maybe_gzip(args.in_path) as fh, open(
            args.out_path, "w", encoding="utf-8", newline=""
        ) as out_fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if not reader.fieldnames:
                raise RuntimeError("Missing TSV header in Swiss-Prot export.")
            writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
        return

    with open_maybe_gzip(args.in_path) as fh, open(
        args.out_path, "w", encoding="utf-8", newline=""
    ) as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(["accession", "uniprot_id", "description", "ec", "sequence"])

        header = None
        seq_lines = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    accession, entry_name, desc, ec = parse_header(header)
                    sequence = "".join(seq_lines)
                    writer.writerow([accession, entry_name, desc, ec, sequence])
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            accession, entry_name, desc, ec = parse_header(header)
            sequence = "".join(seq_lines)
            writer.writerow([accession, entry_name, desc, ec, sequence])


if __name__ == "__main__":
    main()
