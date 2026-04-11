#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_fasta(input_csv: Path, output_fasta: Path) -> int:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    written = 0

    with input_csv.open("r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        with output_fasta.open("w", encoding="utf-8") as out_handle:
            for row in reader:
                sequence_id = (row.get("sequence_id") or "").strip()
                sequence = (row.get("protein_sequence") or "").strip()
                if not sequence_id or not sequence:
                    continue
                if sequence_id in seen_ids:
                    continue

                safe_id = sequence_id.replace("/", "_").replace(".", "_")
                out_handle.write(f">{safe_id}\n{sequence}\n")
                seen_ids.add(sequence_id)
                written += 1

    return written


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FASTA file from data/processed/protein_sequences_unique.csv "
            "using sequence_id as the FASTA header."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input CSV with sequence_id and protein_sequence columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output FASTA path.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    written = build_fasta(args.input, args.output)
    print(f"Wrote {written} sequences to {args.output}")


if __name__ == "__main__":
    main()
