#!/usr/bin/env python3
"""
Generate one FASTA file per dataset from benchmark_card.json.
FASTAs are deduplicated by protein_id and saved to data/fastas/<dataset_id>.fasta
(relative to the card's directory, i.e. the git root).
"""
import argparse
import json
import pandas as pd
from pathlib import Path


def make_fastas(card_path: Path, output_dir: Path):
    with open(card_path) as f:
        card = json.load(f)

    git_root = card_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for prop_name, prop in card["properties"].items():
        for dataset in prop["datasets"]:
            dataset_id  = dataset["dataset_id"]
            csv_path    = git_root / dataset["path"]
            id_col      = dataset["schema"]["protein_id_col"]
            seq_col     = dataset["schema"]["sequence_col"]

            if not csv_path.exists():
                print(f"[{prop_name}] {dataset_id}: SKIPPED — file not found: {dataset['path']}")
                continue

            try:
                df = pd.read_csv(csv_path, usecols=[id_col, seq_col])
            except ValueError as e:
                print(f"[{prop_name}] {dataset_id}: SKIPPED — column mismatch: {e}")
                continue

            df = df.dropna(subset=[id_col, seq_col])
            df = df.drop_duplicates(subset=[id_col])

            fasta_path = output_dir / f"{dataset_id}.fasta"
            with open(fasta_path, "w") as fh:
                for _, row in df.iterrows():
                    pid = str(row[id_col]).replace("/", "_").replace(".", "_")
                    seq = str(row[seq_col]).strip()
                    fh.write(f">{pid}\n{seq}\n")

            print(f"[{prop_name}] {dataset_id}: {len(df)} sequences → {fasta_path.relative_to(git_root)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate FASTA files for every dataset in benchmark_card.json. "
            "Example: python make_fasta_from_card.py --card benchmark_card.json"
        )
    )
    parser.add_argument("--card", required=True, type=str,
                        help="Path to benchmark_card.json")
    parser.add_argument("--output_dir", required=False, type=str, default=None,
                        help="Output directory for FASTAs. "
                             "Default: data/fastas/ relative to the card file.")
    args = parser.parse_args()

    card_path = Path(args.card).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else card_path.parent / "data" / "fastas"

    make_fastas(card_path, output_dir)


if __name__ == "__main__":
    main()
