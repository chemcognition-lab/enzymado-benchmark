#!/usr/bin/env bash 
#Generate fasta from JSON
python ../scripts/enzyme_utils/make_fasta_from_card.py \
      --card ../benchmark_card.json

#Generate a sh file with all models to calculate embeddings from and update json with location of results
python ../scripts/enzyme_utils/generate_embedding_sh_from_card.py \
      --card ../benchmark_card.json \
      --output_sh ../scripts/run_embeddings.sh \
      --update_card


