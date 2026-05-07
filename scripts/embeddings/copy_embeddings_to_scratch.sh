#!/usr/bin/env bash
#SBATCH --job-name=copy_emb_scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=scripts/logs/copy_emb_scratch_%j.out
#SBATCH --nodelist=gpu4

# Copies per-residue embedding H5 files from NFS to /scratch/embeddings
# for faster read access during training.
#
# Total size: ~9.5 TB — expect several hours.
# Files already copied (size-matched) are skipped automatically.
# Partial files (size mismatch) are deleted and re-copied.
#
# Run from the git root:
#   sbatch scripts/copy_embeddings_to_scratch.sh

SRC="/uss/ralmadamonter/enzymado-benchmark/data/embeddings/protein_sequences_unique"
DST="/scratch/embeddings"

mkdir -p "$DST"

# ── Status check ────────────────────────────────────────────────────────────
echo "=== Status check: $DST ==="
echo ""
for src_file in "$SRC"/*.h5; do
    fname=$(basename "$src_file")
    dst_file="$DST/$fname"
    src_size=$(stat -c%s "$src_file" 2>/dev/null || echo 0)
    src_gb=$(echo "scale=1; $src_size / 1073741824" | bc)

    if [ ! -f "$dst_file" ]; then
        echo "  [missing]  $fname  (${src_gb} GB)"
    else
        dst_size=$(stat -c%s "$dst_file" 2>/dev/null || echo 0)
        dst_gb=$(echo "scale=1; $dst_size / 1073741824" | bc)
        if [ "$src_size" -eq "$dst_size" ]; then
            echo "  [ok]       $fname  (${src_gb} GB)"
        else
            echo "  [partial]  $fname  (${dst_gb}/${src_gb} GB — will re-copy)"
        fi
    fi
done
echo ""

# ── Copy ────────────────────────────────────────────────────────────────────
echo "Started: $(date)"
echo ""

copied=0
skipped=0

for src_file in "$SRC"/*.h5; do
    fname=$(basename "$src_file")
    dst_file="$DST/$fname"
    src_size=$(stat -c%s "$src_file" 2>/dev/null || echo 0)

    if [ -f "$dst_file" ]; then
        dst_size=$(stat -c%s "$dst_file" 2>/dev/null || echo 0)
        if [ "$src_size" -eq "$dst_size" ]; then
            echo "  [skip]    $fname (complete)"
            skipped=$((skipped + 1))
            continue
        else
            echo "  [partial] $fname — removing and re-copying..."
            rm -f "$dst_file"
        fi
    fi

    size_gb=$(echo "scale=1; $src_size / 1073741824" | bc)
    echo "  [copy]    $fname  (${size_gb} GB) ..."
    t0=$SECONDS
    rsync --no-whole-file --inplace "$src_file" "$dst_file"
    elapsed=$((SECONDS - t0))
    echo "            done in ${elapsed}s"

    copied=$((copied + 1))
done

echo ""
echo "Done: $copied copied, $skipped skipped"
echo "Finished: $(date)"
echo ""
echo "Disk usage: $(du -sh $DST)"
