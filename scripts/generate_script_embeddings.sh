#!/usr/bin/env bash 
# Prepare FASTA inputs and generate the embedding batch script from the card.
# Run this helper from the repository root.

# Generate FASTA files from the benchmark card.
python ../scripts/enzyme_utils/make_fasta_from_card.py \
      --card ../benchmark_card.json

# Generate the embedding run script and optionally update the card metadata.
python ../scripts/enzyme_utils/generate_embedding_sh_from_card.py \
      --card ../benchmark_card.json \
      --output_sh ../scripts/run_embeddings.sh \
      --update_card

