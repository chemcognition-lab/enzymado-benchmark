#!/usr/bin/env bash

# Shared helpers for cluster job scripts that should read code from the repo
# but write generated artifacts under scratch.

set -euo pipefail

default_scratch_root() {
    printf '%s\n' "/scratch/${USER}/enzymado"
}

init_cluster_runtime_dirs() {
    : "${SCRATCH_ROOT:=$(default_scratch_root)}"
    : "${UV_CACHE_DIR:=${SCRATCH_ROOT}/uv-cache}"
    : "${HF_HOME:=${SCRATCH_ROOT}/hf-cache}"
    : "${TORCH_HOME:=${SCRATCH_ROOT}/torch-cache}"
    : "${XDG_CACHE_HOME:=${SCRATCH_ROOT}/xdg-cache}"
    : "${TMPDIR:=${SCRATCH_ROOT}/tmp}"
    : "${UV_LINK_MODE:=copy}"

    export SCRATCH_ROOT UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR UV_LINK_MODE
    export PYTHONNOUSERSITE=1

    mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"
}

ensure_scratch_card() {
    local repo_card_path="$1"
    local scratch_cards_dir="${2:-${SCRATCH_ROOT}/cards}"
    local scratch_card_path="${scratch_cards_dir}/$(basename "${repo_card_path}")"

    mkdir -p "${scratch_cards_dir}"

    if [[ "${SYNC_CARD_FROM_REPO:-0}" == "1" || ! -f "${scratch_card_path}" ]]; then
        cp "${repo_card_path}" "${scratch_card_path}"
    fi

    printf '%s\n' "${scratch_card_path}"
}

relocate_manifest_to_scratch() {
    local repo_root="$1"
    local source_manifest="$2"
    local data_root="$3"
    local scratch_manifests_dir="${4:-${SCRATCH_ROOT}/manifests}"
    local target_manifest="${scratch_manifests_dir}/$(basename "${source_manifest}" .json)_scratch.json"

    mkdir -p "${scratch_manifests_dir}"

    python "${repo_root}/scripts/enzyme_utils/relocate_embeddings_json.py" \
        "${source_manifest}" \
        "${target_manifest}" \
        --output-root "${data_root}" >/dev/null

    printf '%s\n' "${target_manifest}"
}
