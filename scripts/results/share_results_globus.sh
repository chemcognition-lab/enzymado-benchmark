#!/usr/bin/env bash
#SBATCH --job-name=share_results
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=scripts/logs/share_results_%j.out
#SBATCH --nodelist=gpu4

# Copies completed active_sites and binding_site results to /data/shared
# for Globus sharing via the "GPU4 data" endpoint (3b2993eb-26af-11f0-ae79-02718ce60f75).
#
# Only copies run dirs that have both test_predictions.npz and model.pt/model.joblib.
#
# Run from the git root:
#   sbatch scripts/share_results_globus.sh

SRC="/uss/ralmadamonter/enzymado-benchmark/results/predictions"
DST="/data/shared/big_files/enzymado_results"

mkdir -p "$DST"

echo "Copying completed results to $DST"
echo "Started: $(date)"
echo ""

copied=0
skipped=0

for task in active_sites binding_site; do
    echo "=== $task ==="
    for run_dir in "$SRC"/*__${task}__*; do
        [ -d "$run_dir" ] || continue
        name=$(basename "$run_dir")

        has_preds=false
        has_model=false
        [ -f "$run_dir/test_predictions.npz" ] && has_preds=true
        { [ -f "$run_dir/model.pt" ] || [ -f "$run_dir/model.joblib" ]; } && has_model=true

        if $has_preds && $has_model; then
            dst_dir="$DST/$name"
            if [ -d "$dst_dir" ]; then
                echo "  [skip]  $name (already exists)"
                skipped=$((skipped + 1))
            else
                echo "  [copy]  $name"
                cp -r "$run_dir" "$dst_dir"
                copied=$((copied + 1))
            fi
        else
            echo "  [skip]  $name (incomplete — preds=$has_preds model=$has_model)"
        fi
    done
    echo ""
done

echo "Done: $copied copied, $skipped skipped"
echo "Finished: $(date)"
echo ""
echo "Contents of $DST:"
ls "$DST"
