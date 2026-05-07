#!/usr/bin/env bash
#SBATCH --job-name=tar_embeddings
#SBATCH --array=0-4%1
#SBATCH --mem=16G
#SBATCH --output=logs/tar_embeddings_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ralmadamonter@ucsd.edu

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }
mkdir -p logs

EMB="data/embeddings/protein_sequences_unique"

# 14 h5 files split into 5 groups of 3 (last group has 2)
case $SLURM_ARRAY_TASK_ID in
  0) FILES=("$EMB/prot5_xl_half_uniref50.h5" "$EMB/prot5_xl_uniref50.h5" "$EMB/prost5.h5")
     OUT="embeddings_part0_prot5.tar.zst" ;;
  1) FILES=("$EMB/esm2_650M.h5" "$EMB/esm2_3B.h5" "$EMB/esm3_open.h5")
     OUT="embeddings_part1_esm2_esm3.tar.zst" ;;
  2) FILES=("$EMB/esmc_300m.h5" "$EMB/esmc_600m.h5" "$EMB/ankh_base.h5")
     OUT="embeddings_part2_esmc_ankh.tar.zst" ;;
  3) FILES=("$EMB/ankh_large.h5" "$EMB/ankh3_large.h5" "$EMB/ankh3_xl.h5")
     OUT="embeddings_part3_ankh3.tar.zst" ;;
  4) FILES=("$EMB/profluent_e1_300m.h5" "$EMB/profluent_e1_600m.h5")
     OUT="embeddings_part4_profluent.tar.zst" ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID: creating $OUT"
echo "Files: ${FILES[*]}"

tar --use-compress-program="zstd -v" -cvf "$OUT" "${FILES[@]}"
echo "Done: $OUT"

# Hold the next array task so you can transfer the file before it starts
NEXT=$((SLURM_ARRAY_TASK_ID + 1))
if [[ $NEXT -le 4 ]]; then
  scontrol hold "${SLURM_ARRAY_JOB_ID}_${NEXT}"
  echo "Next task (${SLURM_ARRAY_JOB_ID}_${NEXT}) is ON HOLD."
  echo "Transfer $OUT, then release with:"
  echo "  scontrol release ${SLURM_ARRAY_JOB_ID}_${NEXT}"
fi
