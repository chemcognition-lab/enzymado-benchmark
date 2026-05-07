#!/usr/bin/env bash
#SBATCH --job-name=hf_upload
#SBATCH --mem=16G
#SBATCH --output=logs/hf_upload_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ralmadamonter@ucsd.edu
#SBATCH --nodelist=gpu4

# ── Load conda ───────────────────────────────────────────────────────
source /home/ralmadamonter/miniconda3/etc/profile.d/conda.sh

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }
mkdir -p logs

# ── HuggingFace authentication ───────────────────────────────────────
# Set your token before submitting:
#   export HF_TOKEN=hf_xxxx && sbatch scripts/upload_to_hf.sh
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN is not set. Run:"
  echo "  export HF_TOKEN=hf_xxxx && sbatch scripts/upload_to_hf.sh"
  exit 1
fi

echo "Uploading tar files to HuggingFace..."
for TAR in "$REPO"/embeddings_part*.tar.zst; do
  echo "Uploading $(basename $TAR)..."
  conda run -n prot5_blackwell \
    huggingface-cli upload Enzymado/Enzymado_embeddings \
    "$TAR" \
    "$(basename $TAR)" \
    --repo-type=dataset \
    --token="$HF_TOKEN"
done

echo "Done."
