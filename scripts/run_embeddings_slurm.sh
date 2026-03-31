#!/usr/bin/env bash
#SBATCH --job-name=enzymado_embed
#SBATCH --array=0-13%2
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=logs/embed_%A_%a.out

# Run from the git root:
#   sbatch scripts/run_embeddings_slurm.sh

# ── Load conda ───────────────────────────────────────────────────────
source /home/ralmadamonter/miniconda3/etc/profile.d/conda.sh

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }
SCRIPTS="scripts/enzyme_utils/embedding_scripts"

# ── Generate FASTA from protein_sequences_unique.csv if needed ───────
FASTA="$REPO/data/fastas/protein_sequences_unique_no_fluorescence.fasta"
CSV="$REPO/data/processed/protein_sequences_unique_no_fluorescence.csv"
OUTDIR="$REPO/data/embeddings/protein_sequences_unique"
if [[ ! -f "$FASTA" ]]; then
  echo "Generating FASTA from $CSV ..."
  mkdir -p "$REPO/data/fastas"
  TMPPY=$(mktemp "$TMPDIR/make_fasta_XXXXXX.py")
  cat > "$TMPPY" <<'EOF'
import sys, pandas as pd
from pathlib import Path
csv  = Path(sys.argv[1])
out  = Path(sys.argv[2])
df   = pd.read_csv(csv).dropna(subset=["sequence_id", "protein_sequence"])
df   = df.drop_duplicates(subset=["sequence_id"])
with open(out, "w") as fh:
    for _, row in df.iterrows():
        pid = str(row["sequence_id"]).replace("/", "_").replace(".", "_")
        fh.write(f">{pid}\n{str(row['protein_sequence']).strip()}\n")
print(f"Generated {len(df)} sequences -> {out}")
EOF
  conda run -n prot5_blackwell python "$TMPPY" "$CSV" "$FASTA"
  rm -f "$TMPPY"
fi

# ── Task index → command mapping ─────────────────────────────────────
# 13 tasks (IDs 0-12), one per model
# Model order:
#  0 prot5_xl_half_uniref50   1 prot5_xl_uniref50   2 prost5
#  3 esm2_650M                4 esm2_3B              5 esm3_open
#  6 esmc_300m                7 esmc_600m            8 ankh_base
#  9 ankh_large              10 ankh3_large         11 ankh3_xl
# 12 profluent_e1_300m       13 profluent_e1_600m

CMDS=(
  "conda run -n prot5_blackwell python $SCRIPTS/prot5_embedding_script.py -i $FASTA -o $OUTDIR/prot5_xl_half_uniref50.h5 --model Rostlab/prot_t5_xl_half_uniref50-enc"
  "conda run -n prot5_blackwell python $SCRIPTS/prot5_embedding_script.py -i $FASTA -o $OUTDIR/prot5_xl_uniref50.h5 --model Rostlab/prot_t5_xl_uniref50"
  "conda run -n prot5_blackwell python $SCRIPTS/prost5_embedding_script.py -i $FASTA -o $OUTDIR/prost5.h5 --model Rostlab/ProstT5"
  "conda run -n esm_blackwell python $SCRIPTS/esm2_embedding_script.py -i $FASTA -o $OUTDIR/esm2_650M.h5 --model facebook/esm2_t33_650M_UR50D"
  "conda run -n esm_blackwell python $SCRIPTS/esm2_embedding_script.py -i $FASTA -o $OUTDIR/esm2_3B.h5 --model facebook/esm2_t36_3B_UR50D"
  "conda run -n esm_blackwell python $SCRIPTS/esm3_embedding_script.py -i $FASTA -o $OUTDIR/esm3_open.h5 --model esm3-open"
  "conda run -n esm_blackwell python $SCRIPTS/esmc_embedding_script.py -i $FASTA -o $OUTDIR/esmc_300m.h5 --model esmc_300m"
  "conda run -n esm_blackwell python $SCRIPTS/esmc_embedding_script.py -i $FASTA -o $OUTDIR/esmc_600m.h5 --model esmc_600m"
  "conda run -n ankh python $SCRIPTS/ankh_embedding_script.py -i $FASTA -o $OUTDIR/ankh_base.h5 --model ankh_base"
  "conda run -n ankh python $SCRIPTS/ankh_embedding_script.py -i $FASTA -o $OUTDIR/ankh_large.h5 --model ankh_large"
  "conda run -n ankh python $SCRIPTS/ankh_embedding_script.py -i $FASTA -o $OUTDIR/ankh3_large.h5 --model ankh3_large"
  "conda run -n ankh python $SCRIPTS/ankh_embedding_script.py -i $FASTA -o $OUTDIR/ankh3_xl.h5 --model ankh3_xl"
  "conda run -n profluent-e1 python $SCRIPTS/profluent_embedding_script.py -i $FASTA -o $OUTDIR/profluent_e1_300m.h5 --model Profluent-Bio/E1-300m"
  "conda run -n profluent-e1 python $SCRIPTS/profluent_embedding_script.py -i $FASTA -o $OUTDIR/profluent_e1_600m.h5 --model Profluent-Bio/E1-600m"
)

# ── Sanity check ─────────────────────────────────────────────────────
N_CMDS=${#CMDS[@]}
if [[ $SLURM_ARRAY_TASK_ID -ge $N_CMDS ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (max=$((N_CMDS-1)))"
  exit 1
fi

# ── Resolve output path and create parent directory ──────────────────
CMD="${CMDS[$SLURM_ARRAY_TASK_ID]}"
OUTFILE=$(echo "$CMD" | grep -oP '(?<=-o )\S+')
mkdir -p "$(dirname "$OUTFILE")"
mkdir -p logs

echo "Task $SLURM_ARRAY_TASK_ID / $((N_CMDS-1))"
echo "CMD: $CMD"
echo "OUT: $OUTFILE"

# ── Skip if output already exists ────────────────────────────────────
# esm3/esmc/profluent write incrementally and handle resume internally,
# so only skip for batch-write scripts (prot5, prost5, esm2, ankh).
#if [[ -f "$OUTFILE" ]] && [[ "$CMD" != *"esm3"* ]] && [[ "$CMD" != *"esmc"* ]] && [[ "$CMD" != *"profluent"* ]]; then
#  echo "SKIP: output already exists at $OUTFILE"
#  exit 0
#fi

# ── Run ──────────────────────────────────────────────────────────────
eval "$CMD"
