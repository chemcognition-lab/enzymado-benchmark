#!/usr/bin/env bash
# Attach test-split probabilities to the published classification runs.
#
# This retrains nothing: each run's stored wrapper is restored and re-predicted,
# and attach_probabilities copies the published hard labels through untouched.
# Only the five classification tasks are targeted; regression wrappers expose no
# probability method and would just be loaded and skipped.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="${REPO}/results/predictions"
LOGS="${REPO}/results/protocol_v3/logs"
ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
ARRAY="${ARRAY_SPEC:-0-13%14}"
MAIL="${MAIL_USER:-josechemistry.qb@gmail.com}"
DRYRUN="${DRYRUN:-1}"
POOLED="${REPO}/embeddings_models_output_mean_pooled.json"
RESIDUE="${REPO}/embeddings_models_output.json"
mkdir -p "${LOGS}"

submit() {
    local name=$1 split=$2 mode=$3 manifest=$4 tasks=$5 prefix=$6 mem=$7 time=$8
    local launcher="${REPO}/scripts/slurm/run_training_cluster.slurm"
    [[ "${mode}" == attention ]] && launcher="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"
    local -a cmd=(
        env "OUTPUT_DIR=${OUT}" "EMBEDDINGS_JSON=${manifest}"
        "ATTENTION_EMBEDDINGS_JSON=${RESIDUE}"
        "TASK_SCOPES=properties" "SPLIT_VARIANT=${split}" "EMBEDDING_MODES=${mode}"
        "TASK_NAMES=${tasks}" "SKIP_LABEL_KINDS=" "MODEL_PREFIX=${prefix}"
        "EMB_NAME=" "SEED=42" "OVERWRITE_RESULTS=0" "ATTACH_PROBABILITIES=1"
        "COMMON_IDS_PATH=${REPO}/results/embedding_intersection/common_ids.txt"
        sbatch --parsable "--account=${ACCOUNT}" "--array=${ARRAY}"
        --gres=gpu:l40s:1 --cpus-per-task=8 "--mem=${mem}" "--time=${time}"
        "--mail-user=${MAIL}" --mail-type=FAIL,END
        "--job-name=pb_${name}" "--output=${LOGS}/%x_%A_%a.out"
        --export=ALL "${launcher}"
    )
    if [[ "${DRYRUN}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n\n'
    else local id; id=$("${cmd[@]}"); printf '%-26s %s\n' "pb_${name}" "${id}"; fi
}

for split in random similarity; do
    s=${split:0:4}
    # Inference over the test split only: one forward pass, no epochs. Walltimes
    # are scaled to n_test (binding 2,850 and solubility 367 against 6.6M residues
    # for binding_site), and memory stays at 124G, where billing still matches the
    # GPU alone. A short request also queues far better under a low fairshare.
    submit "bsite_attn_${s}"  "${split}" attention    "${RESIDUE}" binding_site ''      124G 03:00:00
    submit "bsite_na_${s}"    "${split}" no_attention "${RESIDUE}" binding_site ''      124G 03:00:00
    submit "asite_attn_${s}"  "${split}" attention    "${RESIDUE}" active_sites ''      124G 02:00:00
    submit "asite_na_${s}"    "${split}" no_attention "${RESIDUE}" active_sites ''      124G 02:00:00
    submit "ec_attn_${s}"     "${split}" attention    "${RESIDUE}" ec_number    ''      124G 02:00:00
    submit "ec_na_${s}"       "${split}" no_attention "${POOLED}"  ec_number    ''      124G 01:00:00
    submit "pool_attn_${s}"   "${split}" attention    "${RESIDUE}" binding,solubility '' 124G 01:00:00
    submit "pool_na_${s}"     "${split}" no_attention "${POOLED}"  binding,solubility '' 124G 01:00:00
done
