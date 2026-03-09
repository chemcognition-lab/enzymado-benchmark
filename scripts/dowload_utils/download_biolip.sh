#!/usr/bin/env bash
set -euo pipefail

OUT_PATH="${1:-data/raw/biolip/BioLiP_nr.txt.gz}"
URL="${2:-https://aideepmed.com/BioLiP/download/BioLiP_nr.txt.gz}"

mkdir -p "$(dirname "${OUT_PATH}")"
curl -L --fail "${URL}" -o "${OUT_PATH}"

echo "Downloaded BioLiP to ${OUT_PATH}"
