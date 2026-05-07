#!/usr/bin/env bash
#SBATCH --job-name=h5_to_lmdb
#SBATCH --array=0-13
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/h5_to_lmdb_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ralmadamonter@ucsd.edu

# ── Load conda ───────────────────────────────────────────────────────
source /home/ralmadamonter/miniconda3/etc/profile.d/conda.sh

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }
mkdir -p logs

EMB="$REPO/data/embeddings/protein_sequences_unique"
LMDB_DIR="$REPO/data/embeddings_lmdb/protein_sequences_unique"
mkdir -p "$LMDB_DIR"

# One h5 file per task
H5_FILES=(
  "$EMB/prot5_xl_half_uniref50.h5"
  "$EMB/prot5_xl_uniref50.h5"
  "$EMB/prost5.h5"
  "$EMB/esm2_650M.h5"
  "$EMB/esm2_3B.h5"
  "$EMB/esm3_open.h5"
  "$EMB/esmc_300m.h5"
  "$EMB/esmc_600m.h5"
  "$EMB/ankh_base.h5"
  "$EMB/ankh_large.h5"
  "$EMB/ankh3_large.h5"
  "$EMB/ankh3_xl.h5"
  "$EMB/profluent_e1_300m.h5"
  "$EMB/profluent_e1_600m.h5"
)

H5="${H5_FILES[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(basename "$H5" .h5)
OUT="$LMDB_DIR/${MODEL}.lmdb"

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL"
echo "Input:  $H5"
echo "Output: $OUT"

if [[ ! -f "$H5" ]]; then
  echo "ERROR: $H5 not found, skipping."
  exit 1
fi

TMPPY=$(mktemp "$TMPDIR/h5_to_lmdb_XXXXXX.py")
cat > "$TMPPY" <<'EOF'
import sys
import h5py
import lmdb
import numpy as np
from pathlib import Path

h5_path  = Path(sys.argv[1])
out_path = Path(sys.argv[2])

if out_path.exists():
    print(f"Output already exists at {out_path}, skipping.")
    sys.exit(0)

# Estimate map_size from h5 file size * 1.2 headroom
h5_size = h5_path.stat().st_size
map_size = int(h5_size * 1.2)
print(f"H5 size: {h5_size / 1e9:.2f} GB — setting LMDB map_size to {map_size / 1e9:.2f} GB")

env = lmdb.open(str(out_path), map_size=map_size)

with h5py.File(str(h5_path), "r") as hf:
    n_total = len(hf.keys())
    print(f"Converting {n_total} sequences...")
    with env.begin(write=True) as txn:
        for i, key in enumerate(hf.keys(), 1):
            arr = hf[key][:]  # numpy array, float32
            # Store shape alongside bytes: prefix with shape info
            shape = np.array(arr.shape, dtype=np.int32)
            value = shape.tobytes() + arr.astype(np.float32).tobytes()
            txn.put(key.encode(), value)
            if i % 10000 == 0:
                print(f"  {i}/{n_total}")

env.close()
print(f"Done: {out_path}")
EOF

conda run -n prot5_blackwell python "$TMPPY" "$H5" "$OUT"
rm -f "$TMPPY"
