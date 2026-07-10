#!/usr/bin/env bash
# Prepare the deduplicated FASTA and generate the embedding batch script.
# Run this helper from the repository root.

# 1) Rebuild the canonical unique-sequence index from processed datasets.
python ../data_prep/concat_unique_protein_sequences.py \
      --output ../data/processed/protein_sequences_unique.csv

# 2) Build one deduplicated FASTA from that index.
python ../data_prep/make_fasta_from_unique_sequences.py \
      --input ../data/processed/protein_sequences_unique.csv \
      --output ../data/fastas/protein_sequences_unique.fasta

# 3) Generate the embedding run script (one model run over the whole FASTA).
python ../embeddings/generate_embedding_sh.py \
      --fasta ../data/fastas/protein_sequences_unique.fasta \
      --output_sh ../scripts/run_embeddings.sh \
      --git_root ..
