#!/usr/bin/env bash
#
# Embed one task's sequences with several PLMs in parallel, saving both the
# per-residue (L x D) shards and a mean-pooled (D,) file in a single GPU pass.
#
#   L x D shards  -> ${EMBED_OUTPUT_ROOT}/embeddings/<task>_sequences/<model>_shards   (scratch)
#   mean-pooled   -> ${REPO}/data/embeddings_mean_pooled/<task>_sequences/<model>.safetensors
#
# Run on a LOGIN node (compute nodes have no internet), inside tmux, from the
# repository root. Every stage is skipped when its output already exists, so a
# failed run can simply be restarted.
#
#   tmux new -s embed
#   bash scripts/slurm/submit_task_embeddings.sh
#
# Overrides:
#   TASK=km bash scripts/slurm/submit_task_embeddings.sh
#   EMBED_MODELS="esm2_650M" bash scripts/slurm/submit_task_embeddings.sh
#   NO_SUBMIT=1 bash scripts/slurm/submit_task_embeddings.sh   # prepare only

set -euo pipefail

: "${TASK:=ph_opt}"
: "${WALLTIME:=3:00:00}"
: "${JOB_MEM:=65536M}"
: "${NO_SUBMIT:=0}"

# One model per family by default; pass any subset of these four as a
# space-separated EMBED_MODELS. All four load from the HF hub without a token.
: "${EMBED_MODELS:=prot5_xl_half_uniref50 esm2_650M esmc_600m ankh3_large}"
read -r -a EMBED_MODELS <<< "${EMBED_MODELS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO}"

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi

# Mirrors run_embeddings_cluster.slurm so the job finds the weights cached here.
: "${SCRATCH_ROOT:=/scratch/${USER}/enzymado}"
: "${HF_HOME:=${SCRATCH_ROOT}/hf-cache}"
: "${TORCH_HOME:=${SCRATCH_ROOT}/torch-cache}"
: "${XDG_CACHE_HOME:=${SCRATCH_ROOT}/xdg-cache}"
: "${TMPDIR:=${SCRATCH_ROOT}/tmp}"
: "${EMBED_OUTPUT_ROOT:=${SCRATCH_ROOT}/data}"
export SCRATCH_ROOT HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR EMBED_OUTPUT_ROOT
export PYTHONNOUSERSITE=1
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

step() { printf '\n=== %s ===\n' "$1"; }

SEQUENCE_SET="${TASK}_sequences"
SUBSET_CSV="${REPO}/data/processed/${SEQUENCE_SET}.csv"
FASTA="${REPO}/data/fastas/${SEQUENCE_SET}.fasta"
POOLED_DIR="data/embeddings_mean_pooled/${SEQUENCE_SET}"
RESIDUE_DIR="${EMBED_OUTPUT_ROOT}/embeddings/${SEQUENCE_SET}"
POOLED_MANIFEST="${REPO}/embeddings_${TASK}_pooled.json"
RESIDUE_MANIFEST="${REPO}/embeddings_${TASK}_residue.json"

echo "Repository:   ${REPO}  ($(df -P "${REPO}" | awk 'NR==2 {print $6}'))"
echo "Task:         ${TASK}"
echo "Models:       ${EMBED_MODELS[*]}"
echo "Mean-pooled:  ${REPO}/${POOLED_DIR}"
echo "L x D shards: ${RESIDUE_DIR}"

# name -> python module, venv, embedding dim, position in the MODELS array of
# run_embeddings_cluster.slurm (its array index)
model_env() {
    case "$1" in
        prot5_xl_half_uniref50) echo "python/3.13.2 .venv-prot5 1024 0" ;;
        esm2_650M)              echo "python/3.13.2 .venv-esm2 1280 3" ;;
        esmc_600m)              echo "python/3.12.4 .venv-esm3-esmc 1152 7" ;;
        ankh3_large)            echo "python/3.12.4 .venv-ankh 1536 10" ;;
        *) echo "ERROR: no environment mapping for '$1'." >&2; exit 1 ;;
    esac
}

ARRAY_INDICES=""
for model in "${EMBED_MODELS[@]}"; do
    read -r _ _ _ index <<< "$(model_env "${model}")"
    ARRAY_INDICES="${ARRAY_INDICES:+${ARRAY_INDICES},}${index}"
done

# The pinned envs_files/*.txt were frozen on another machine; several pins
# (numpy, +cu126 torch) do not exist in the Alliance wheelhouse, and pip here
# refuses PyPI builds of those packages. Install the unpinned top-level
# packages from the wheelhouse instead, plus PyPI-only extras.
HF_PACKAGES=(torch transformers sentencepiece protobuf safetensors huggingface_hub pandas numpy)

# Build a venv once. Marker files record what this script started and
# finished, so a venv it left half-built is rebuilt rather than silently
# reused. A venv it did not create, such as the pinned environment behind the
# published embeddings, is reused and never deleted.
build_venv() {
    local py_mod="$1" venv="$2"; shift 2
    local pypi_extras=("$@")
    if [[ -f "${REPO}/${venv}/.enzymado_built" ]]; then
        echo "${venv} present, skipping."
        return 0
    fi
    if [[ -e "${REPO}/${venv}" && ! -f "${REPO}/${venv}/.enzymado_building" ]]; then
        echo "${venv} exists and was not built by this script; reusing it."
        return 0
    fi
    rm -rf "${REPO:?}/${venv}"
    module load "${py_mod}" >/dev/null 2>&1
    python -m venv "${REPO}/${venv}"
    touch "${REPO}/${venv}/.enzymado_building"
    "${REPO}/${venv}/bin/pip" install --no-index "${HF_PACKAGES[@]}"
    if (( ${#pypi_extras[@]} )); then
        "${REPO}/${venv}/bin/pip" install "${pypi_extras[@]}"
    fi
    touch "${REPO}/${venv}/.enzymado_built"
}

# --- 1. environments ---------------------------------------------------------
step "1/5  Environments"
if [[ ! -x "${REPO}/.venv/bin/python" ]]; then
    echo "ERROR: missing training environment ${REPO}/.venv." >&2
    exit 1
fi
"${REPO}/.venv/bin/python" -c "import enzymado" 2>/dev/null \
    || "${REPO}/.venv/bin/pip" install --no-deps -e "${REPO}"

for model in "${EMBED_MODELS[@]}"; do
    read -r py_mod venv _ <<< "$(model_env "${model}")"
    case "${model}" in
        esmc_600m) build_venv "${py_mod}" "${venv}" "esm>=3.1" ;;
        *)         build_venv "${py_mod}" "${venv}" ;;
    esac
done

# --- 2. weights --------------------------------------------------------------
step "2/5  Caching weights into ${HF_HOME}"
# The Xet download backend reassembles multi-GB files in memory, which gets the
# process killed by the login node's resource limits; plain HTTP does not.
export HF_HUB_DISABLE_XET=1
for model in "${EMBED_MODELS[@]}"; do
    read -r py_mod venv _ <<< "$(model_env "${model}")"
    module load "${py_mod}" >/dev/null 2>&1
    echo "--- ${model}"
    "${REPO}/${venv}/bin/python" - "${model}" <<'PY'
import sys
from huggingface_hub import snapshot_download

name = sys.argv[1]
repos = {
    "prot5_xl_half_uniref50": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "ankh3_large": "ElnaggarLab/ankh3-large",
}
if name in repos:
    print(snapshot_download(repos[name], max_workers=2))
else:
    # ESMC resolves its own repo and file layout through the esm SDK.
    from esm.models.esmc import ESMC
    ESMC.from_pretrained(name)
    print("cached via esm SDK")
PY
done

# --- 3. task subset + FASTA --------------------------------------------------
step "3/5  ${TASK} sequence subset"
if [[ -f "${SUBSET_CSV}" ]]; then
    echo "${SUBSET_CSV} exists, skipping."
else
    "${REPO}/.venv/bin/python" "${REPO}/scripts/data_prep/subset_sequence_index_for_task.py" \
        --task_name "${TASK}" --output "${SUBSET_CSV}"
fi
if [[ -f "${FASTA}" ]]; then
    echo "${FASTA} exists, skipping."
else
    "${REPO}/.venv/bin/python" "${REPO}/scripts/data_prep/make_fasta_from_unique_sequences.py" \
        --input "${SUBSET_CSV}" --output "${FASTA}"
fi
echo "Sequences to embed: $(grep -c '^>' "${FASTA}")"

# --- 4. manifests ------------------------------------------------------------
step "4/5  Manifests"
"${REPO}/.venv/bin/python" - "${POOLED_MANIFEST}" "${RESIDUE_MANIFEST}" \
    "${POOLED_DIR}" "${RESIDUE_DIR}" \
    "$(for m in "${EMBED_MODELS[@]}"; do read -r _ _ dim _ <<< "$(model_env "$m")"; printf '%s:%s ' "$m" "$dim"; done)" <<'PY'
import json, sys

pooled_path, residue_path, pooled_dir, residue_dir, specs = sys.argv[1:6]
models = dict(spec.split(":") for spec in specs.split())
pooled = {m: {"path": f"{pooled_dir}/{m}.safetensors", "dim": int(d)} for m, d in models.items()}
residue = {m: {"path": f"{residue_dir}/{m}_shards", "dim": int(d)} for m, d in models.items()}
for path, payload in ((pooled_path, pooled), (residue_path, residue)):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=4)
    print(path)
PY

# --- 5. submit ---------------------------------------------------------------
step "5/5  Submitting embedding array"
if [[ "${NO_SUBMIT}" == "1" ]]; then
    echo "NO_SUBMIT=1 set; prepared but not submitted."
    exit 0
fi

: "${AIP_ACCOUNT:=$(ls -d ~/projects/*/ 2>/dev/null | head -n 1 | sed 's|/$||;s|.*/||')}"
if [[ -z "${AIP_ACCOUNT}" ]]; then
    echo "ERROR: could not infer an account. Export AIP_ACCOUNT and rerun." >&2
    exit 1
fi

# PER_PROTEIN=0 + SAVE_MEAN_POOLED=1: residue shards and the pooled file come
# out of the same forward pass. HF_HUB_OFFLINE stops the job reaching for the
# network it does not have and makes a missing cache fail fast.
JOB_ID="$(sbatch --parsable \
    --account="${AIP_ACCOUNT}" \
    --array="${ARRAY_INDICES}" \
    --time="${WALLTIME}" \
    --mem="${JOB_MEM}" \
    --mail-type=NONE \
    --export=ALL,SEQUENCE_SET="${SEQUENCE_SET}",PER_PROTEIN=0,SAVE_MEAN_POOLED=1,EMBED_OUTPUT_ROOT="${EMBED_OUTPUT_ROOT}",HF_HUB_OFFLINE=1 \
    "${REPO}/scripts/slurm/run_embeddings_cluster.slurm")"

printf '\nSubmitted array job %s (account %s), tasks %s.\n' "${JOB_ID}" "${AIP_ACCOUNT}" "${ARRAY_INDICES}"
echo "Watch:  squeue -u \$USER    |    tail -f enzymado_embed_${JOB_ID}_*.out"
