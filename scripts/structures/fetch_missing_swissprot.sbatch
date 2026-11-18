#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --job-name=fetch_sp_afdb
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/chemist/enzymado/data/afdb_api/logs/fetch_%j.log

# Download AlphaFold DB structures that the bulk downloads cannot provide and
# that are only reachable via the per-accession API: the 5,514 Swiss-Prot
# accessions not in the bulk AFDB tar, plus the 621 non-Swiss-Prot accessions
# whose F1 file returned 404 (they may still have per-chain models, as
# polyproteins do). 6,135 accessions in total, in accessions_api.txt.
# Resume works if the job times out: re-submit and rows already written
# with status ok/no_afdb/no_match are skipped; rows with status error are
# retried.
#
# A row is included when ANY accession of its ID is in the list. For an ID
# with several accessions, fetch_alphafold.py keeps the first accession whose
# model contains the sequence, so the manifest's accession column can be a
# different accession of the same ID than the one that triggered inclusion.
# The structure still contains the pool sequence exactly.
#
# Expected input rows: 6,947 pool rows for those 6,135 accessions (counted
# 2026-11-11, including multi-accession IDs and polyprotein chain IDs).
# At ~1.5 s per ID the full run takes about 3 h; 6 h gives ample buffer
# for 429 backoff retries.
#
# Submit from the repo root on a login node:
#   sbatch scripts/structures/fetch_missing_swissprot.sbatch

set -euo pipefail

REPO_ROOT=/project/6111759/chemist/enzymado/enzymado-benchmark
POOL_CSV="${REPO_ROOT}/data/processed/protein_pool.csv"
TARGET_ACC=/scratch/chemist/enzymado/data/afdb_coverage/accessions_api.txt
OUTPUT_DIR=/scratch/chemist/enzymado/data/afdb_api
FILTERED_CSV="${OUTPUT_DIR}/input_sp_missing.csv"
# The system python3 on compute nodes may lack numpy, which fetch_alphafold.py needs.
PY=/scratch/chemist/enzymado/venvs/structures/bin/python

mkdir -p "${OUTPUT_DIR}/logs"

echo "=== Building filtered input CSV ==="
"${PY}" - "${POOL_CSV}" "${TARGET_ACC}" "${FILTERED_CSV}" << 'PYEOF'
import csv
import sys

pool_path, target_path, out_path = sys.argv[1:4]

with open(target_path) as fh:
    target_accs = {line.strip() for line in fh if line.strip()}

def any_match(seq_id: str) -> bool:
    base = seq_id.split("__")[0]
    return any(acc.strip() in target_accs for acc in base.split(","))

with open(pool_path, newline="") as fin, open(out_path, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["sequence_id", "protein_sequence"])
    writer.writeheader()
    n_rows = 0
    for row in reader:
        if any_match(row["sequence_id"]):
            writer.writerow(
                {"sequence_id": row["sequence_id"], "protein_sequence": row["protein_sequence"]}
            )
            n_rows += 1
print(f"Input rows matching target accessions: {n_rows}", flush=True)
print(f"Written to {out_path}", flush=True)
PYEOF

echo "=== Starting fetch_alphafold.py ==="
"${PY}" "${REPO_ROOT}/scripts/structures/fetch_alphafold.py" \
    --input "${FILTERED_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --delay 1.0

echo "=== Done ==="
