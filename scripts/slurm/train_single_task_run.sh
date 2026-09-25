#!/usr/bin/env bash
#
# Train and score one task against one mean-pooled embedding store.
#
# Run this after scripts/slurm/setup_single_task_run.sh has finished and its
# embedding job has completed. XGBoost over an 11k x 1280 pooled array needs no
# GPU, so this is meant to run interactively on a login or CPU node.
#
#   bash scripts/slurm/train_single_task_run.sh
#
# Both splits are trained. The random-vs-similarity gap is the point of the
# exercise: a random-split score that collapses under the similarity split was
# largely measuring near-duplicate leakage between train and test.
#
# Overrides:
#   TASK=solubility EMBED_MODEL=ankh_base MODEL_PREFIX=mlp_ bash ...

set -euo pipefail

: "${TASK:=ph_opt}"
: "${EMBED_MODEL:=esm2_650M}"
: "${MODEL_PREFIX:=xgb_}"
: "${TASK_SCOPE:=properties}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO}"

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi

: "${SCRATCH_ROOT:=/scratch/${USER}/enzymado}"
: "${TMPDIR:=${SCRATCH_ROOT}/tmp}"
export SCRATCH_ROOT TMPDIR
export PYTHONNOUSERSITE=1
mkdir -p "${TMPDIR}"

MANIFEST="${REPO}/embeddings_${TASK}.json"
# Kept out of results/ so a personal experiment is never mistaken for, or
# merged into, the published benchmark tree.
OUT_ROOT="${REPO}/results_${TASK}"

step() { printf '\n=== %s ===\n' "$1"; }

for required in "${MANIFEST}" "${REPO}/.venv"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing ${required}. Run setup_single_task_run.sh first." >&2
        exit 1
    fi
done

module load python/3.13.2 >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

step "Checking the embedding store"
SHARDS="$(python - "$MANIFEST" "$EMBED_MODEL" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    print(json.load(fh)[sys.argv[2]]["path"])
PY
)"
if [[ ! -d "${SHARDS}" ]]; then
    echo "ERROR: embedding store not found at ${SHARDS}" >&2
    echo "The embedding job may still be queued or may have failed." >&2
    exit 1
fi
echo "Store: ${SHARDS}"
echo "Size:  $(du -sh "${SHARDS}" | cut -f1)"
echo "Files: $(find "${SHARDS}" -name '*.safetensors' | wc -l | tr -d ' ')"

# No --allowed_ids here. That flag restricts training to the ~269k IDs shared by
# all 14 published embeddings, which only matters when models are compared to
# each other. Omitting it uses every sequence this task has, at the cost of not
# being directly comparable to the published leaderboard.
for SPLIT in random similarity; do
    step "Training ${TASK} / ${EMBED_MODEL} / ${MODEL_PREFIX} / ${SPLIT}"
    python "${REPO}/scripts/run_benchmark.py" "${MANIFEST}" \
        --output_dir "${OUT_ROOT}/predictions" \
        --emb_name "${EMBED_MODEL}" \
        --model_prefix "${MODEL_PREFIX}" \
        --task_name "${TASK}" \
        --task_scope "${TASK_SCOPE}" \
        --split_variant "${SPLIT}" \
        --embedding_mode no_attention \
        || echo "NOTE: ${SPLIT} split did not run (a task may register no ${SPLIT} split)."
done

step "Computing metrics"
python "${REPO}/scripts/results/compute_metrics.py" \
    --predictions-root "${OUT_ROOT}/predictions" \
    --statistics-root "${OUT_ROOT}/statistics" \
    --benchmark-card "${REPO}/benchmark_card.json" \
    --filter-task "${TASK}"

step "Results"
# compute_metrics.py writes metrics.json into each run directory under the
# predictions tree; --statistics-root is only a sufficient-statistics cache for
# classification tasks and stays empty for regression.
find "${OUT_ROOT}/predictions" -name 'metrics.json' | sort | while read -r f; do
    echo "--- ${f#"${OUT_ROOT}/predictions/"}"
    python - "$f" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    payload = json.load(fh)

test = payload.get("test", {})
bootstrap = payload.get("test_bootstrap", {}).get("metrics", {})
print(f"    n = {test.get('n', '?')}")
for key in ("mae", "rmse", "r2", "kendall_tau", "accuracy", "f1", "auroc", "auprc"):
    value = test.get(key)
    if not isinstance(value, (int, float)):
        continue
    interval = bootstrap.get(key, {})
    lo, hi = interval.get("ci_lo"), interval.get("ci_hi")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        print(f"    {key:<12} {value: .4f}   95% CI [{lo: .4f}, {hi: .4f}]")
    else:
        print(f"    {key:<12} {value: .4f}")
PY
done

printf '\nPredictions and metrics: %s\n' "${OUT_ROOT}/predictions"
