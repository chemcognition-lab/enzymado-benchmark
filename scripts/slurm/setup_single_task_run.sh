#!/usr/bin/env bash
#
# One-shot cluster setup for a single-task, single-embedding benchmark run.
#
# Defaults reproduce the cheapest useful experiment in this repo: ph_opt
# (11,110 unique sequences) embedded once with esm2_650M and mean-pooled during
# extraction, which keeps the store at ~28 MB instead of the ~13 GB a
# residue-level store of the same sequences would need.
#
# Run this on a LOGIN node, inside tmux, from the repository root:
#
#   tmux new -s setup
#   bash scripts/slurm/setup_single_task_run.sh
#
# It is idempotent: each stage is skipped if its output already exists, so a
# failed run can simply be restarted.
#
# Overrides:
#   TASK=solubility EMBED_MODEL=ankh_base bash scripts/slurm/setup_single_task_run.sh
#   NO_SUBMIT=1 bash scripts/slurm/setup_single_task_run.sh   # prepare only

set -euo pipefail

: "${TASK:=ph_opt}"
: "${EMBED_MODEL:=esm2_650M}"
: "${WALLTIME:=2:00:00}"
: "${JOB_MEM:=32G}"
: "${NO_SUBMIT:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO}"

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi

# Keep every cache off the home quota; these mirror run_embeddings_cluster.slurm
# so the prepared weights are the ones the job will find.
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

# --- model -> environment map ------------------------------------------------
# Mirrors the parallel arrays in run_embeddings_cluster.slurm.
case "${EMBED_MODEL}" in
    esm2_650M)    PY_MOD=python/3.13.2; VENV=.venv-esm2;      REQS=venv-esm2.txt;      HF_REF=facebook/esm2_t33_650M_UR50D ;;
    esm2_3B)      PY_MOD=python/3.13.2; VENV=.venv-esm2;      REQS=venv-esm2.txt;      HF_REF=facebook/esm2_t36_3B_UR50D ;;
    prot5_xl_half_uniref50)
                  PY_MOD=python/3.13.2; VENV=.venv-prot5;     REQS=venv-prot5.txt;     HF_REF=Rostlab/prot_t5_xl_half_uniref50-enc ;;
    prot5_xl_uniref50)
                  PY_MOD=python/3.13.2; VENV=.venv-prot5;     REQS=venv-prot5.txt;     HF_REF=Rostlab/prot_t5_xl_uniref50 ;;
    prost5)       PY_MOD=python/3.13.2; VENV=.venv-prot5;     REQS=venv-prot5.txt;     HF_REF=Rostlab/ProstT5 ;;
    ankh3_large)  PY_MOD=python/3.12.4; VENV=.venv-ankh;      REQS=venv-ankh.txt;      HF_REF=ElnaggarLab/ankh3-large ;;
    ankh3_xl)     PY_MOD=python/3.12.4; VENV=.venv-ankh;      REQS=venv-ankh.txt;      HF_REF=ElnaggarLab/ankh3-xl ;;
    ankh_base|ankh_large)
                  PY_MOD=python/3.12.4; VENV=.venv-ankh;      REQS=venv-ankh.txt;      HF_REF="" ;;
    esm3_open|esmc_300m|esmc_600m)
                  PY_MOD=python/3.12.4; VENV=.venv-esm3-esmc; REQS=venv-esm3-esmc.txt; HF_REF="" ;;
    profluent_e1_300m)
                  PY_MOD=python/3.12.4; VENV=.venv-profluent; REQS=venv-profluent.txt; HF_REF=Profluent-Bio/E1-300m ;;
    profluent_e1_600m)
                  PY_MOD=python/3.12.4; VENV=.venv-profluent; REQS=venv-profluent.txt; HF_REF=Profluent-Bio/E1-600m ;;
    *) echo "ERROR: unknown EMBED_MODEL='${EMBED_MODEL}'." >&2; exit 1 ;;
esac

if [[ "${EMBED_MODEL}" == "esm3_open" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: esm3_open is a gated repo and needs HF_TOKEN exported." >&2
    exit 1
fi

SEQUENCE_SET="${TASK}_sequences"
SUBSET_CSV="${REPO}/data/processed/${SEQUENCE_SET}.csv"
FASTA="${REPO}/data/fastas/${SEQUENCE_SET}.fasta"
SHARDS="${EMBED_OUTPUT_ROOT}/embeddings_mean_pooled/${SEQUENCE_SET}/${EMBED_MODEL}_shards"
MANIFEST="${REPO}/embeddings_${TASK}.json"

echo "Repository:   ${REPO}"
echo "Task:         ${TASK}"
echo "Embedding:    ${EMBED_MODEL}  (mean-pooled at extraction)"
echo "Environment:  ${VENV}  (${PY_MOD})"
echo "Shard target: ${SHARDS}"

module load "${PY_MOD}" git-lfs/3.4.0 >/dev/null 2>&1 || module load "${PY_MOD}" >/dev/null 2>&1

# --- 1. embedding environment ------------------------------------------------
step "1/6  Embedding environment (${VENV})"
if [[ -d "${REPO}/${VENV}" ]]; then
    echo "Already present, skipping."
else
    python -m venv "${REPO}/${VENV}"
    # shellcheck disable=SC1091
    source "${REPO}/${VENV}/bin/activate"
    python -m pip install --upgrade pip >/dev/null
    # Cluster-local prebuilt wheels are far faster and avoid the pinned PyPI
    # CUDA wheels entirely; fall back to PyPI when no local index exists.
    if ! python -m pip install --no-index -r "${REPO}/envs_files/${REQS}" 2>/dev/null; then
        echo "No local wheel index usable; installing from PyPI (slower)."
        python -m pip install -r "${REPO}/envs_files/${REQS}"
    fi
    deactivate
fi

# shellcheck disable=SC1091
source "${REPO}/${VENV}/bin/activate"

# --- 2. model weights --------------------------------------------------------
step "2/6  Pre-downloading weights into ${HF_HOME}"
if [[ -z "${HF_REF}" ]]; then
    echo "${EMBED_MODEL} loads through its own SDK; it will download on first use."
    echo "If compute nodes have no outbound network, run one tiny job on a login"
    echo "node first to warm ${HF_HOME}."
else
    python - "$HF_REF" <<'PY'
import sys
from transformers import AutoModel, AutoTokenizer
ref = sys.argv[1]
print(f"Fetching {ref} ...")
AutoTokenizer.from_pretrained(ref)
AutoModel.from_pretrained(ref)
print("Weights cached.")
PY
fi

# --- 3. task subset + FASTA --------------------------------------------------
step "3/6  Building ${TASK} sequence subset"
if [[ -f "${SUBSET_CSV}" ]]; then
    echo "${SUBSET_CSV} exists, skipping."
else
    python "${REPO}/scripts/data_prep/subset_sequence_index_for_task.py" \
        --task_name "${TASK}" --output "${SUBSET_CSV}"
fi

if [[ -f "${FASTA}" ]]; then
    echo "${FASTA} exists, skipping."
else
    python "${REPO}/scripts/data_prep/make_fasta_from_unique_sequences.py" \
        --input "${SUBSET_CSV}" --output "${FASTA}"
fi
echo "Sequences to embed: $(grep -c '^>' "${FASTA}")"

deactivate

# --- 4. manifest -------------------------------------------------------------
step "4/6  Writing manifest ${MANIFEST}"
case "${EMBED_MODEL}" in
    ankh_base) DIM=768 ;;
    esmc_300m) DIM=960 ;;
    prot5_xl_uniref50|prot5_xl_half_uniref50|prost5|profluent_e1_300m) DIM=1024 ;;
    esmc_600m) DIM=1152 ;;
    esm2_650M|profluent_e1_600m) DIM=1280 ;;
    ankh_large|ankh3_large|esm3_open) DIM=1536 ;;
    esm2_3B|ankh3_xl) DIM=2560 ;;
esac

cat > "${MANIFEST}" <<JSON
{
    "${EMBED_MODEL}": {
        "path": "${SHARDS}",
        "dim": ${DIM}
    }
}
JSON
cat "${MANIFEST}"

# --- 5. submit ---------------------------------------------------------------
step "5/6  Submitting embedding job"
if [[ "${NO_SUBMIT}" == "1" ]]; then
    echo "NO_SUBMIT=1 set; prepared but not submitted."
    exit 0
fi

: "${AIP_ACCOUNT:=$(ls -d ~/projects/*/ 2>/dev/null | head -n 1 | sed 's|/$||;s|.*/||')}"
if [[ -z "${AIP_ACCOUNT}" ]]; then
    echo "ERROR: could not infer an account. Export AIP_ACCOUNT and rerun." >&2
    exit 1
fi
echo "Account: ${AIP_ACCOUNT}"

JOB_ID="$(sbatch --parsable \
    --account="${AIP_ACCOUNT}" \
    --time="${WALLTIME}" \
    --mem="${JOB_MEM}" \
    --export=ALL,EMBED_MODEL="${EMBED_MODEL}",SEQUENCE_SET="${SEQUENCE_SET}",PER_PROTEIN=1,EMBED_OUTPUT_ROOT="${EMBED_OUTPUT_ROOT}" \
    "${REPO}/scripts/slurm/run_embeddings_cluster.slurm")"
echo "Submitted job ${JOB_ID}."

# A job that fails on a missing path or venv dies within seconds. Watch long
# enough to tell "still queued" (fine, leave) from "ran and died" (not fine).
echo "Watching for up to 5 minutes to confirm it does not fail immediately..."
SUBMIT_OK=1
for _ in $(seq 1 60); do
    sleep 5
    STATE="$(squeue -j "${JOB_ID}" -h -o %T 2>/dev/null || true)"
    if [[ -z "${STATE}" ]]; then
        echo "WARNING: job left the queue early - it may have failed on startup."
        sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed 2>/dev/null || true
        tail -n 25 "${REPO}"/enzymado_embed_"${JOB_ID}"*.out 2>/dev/null || true
        SUBMIT_OK=0
        break
    fi
    if [[ "${STATE}" == "RUNNING" ]]; then
        echo "Job ${JOB_ID} is RUNNING."
        sleep 20
        echo "--- log so far ---"
        tail -n 25 "${REPO}"/enzymado_embed_"${JOB_ID}"*.out 2>/dev/null || true
        break
    fi
done

# --- 6. training environment -------------------------------------------------
# Built last, on purpose: the embedding job is already queued or running, so
# this otherwise-idle stretch is free. Nothing below blocks the GPU job.
step "6/6  Training environment (.venv)"
if [[ -d "${REPO}/.venv" ]]; then
    echo "Already present, skipping."
else
    python -m venv "${REPO}/.venv"
    # shellcheck disable=SC1091
    source "${REPO}/.venv/bin/activate"
    python -m pip install --upgrade pip >/dev/null
    if ! python -m pip install --no-index -r "${REPO}/envs_files/venv.txt" 2>/dev/null; then
        echo "No local wheel index usable; installing from PyPI (slower)."
        python -m pip install -r "${REPO}/envs_files/venv.txt"
    fi
    deactivate
fi

printf '\n========================================\n'
if [[ "${SUBMIT_OK}" == "0" ]]; then
    echo "Setup finished, but job ${JOB_ID} did not start cleanly."
    echo "Read enzymado_embed_${JOB_ID}*.out before resubmitting."
    exit 1
fi
echo "Setup complete. Embedding job: ${JOB_ID}"
echo "Safe to leave."
echo
echo "On return:"
echo "  squeue -j ${JOB_ID}"
echo "  du -sh ${SHARDS}        # expect ~28 MB for ph_opt/esm2_650M"
echo "  bash scripts/slurm/train_single_task_run.sh"
printf '========================================\n'
