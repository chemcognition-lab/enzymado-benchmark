#!/usr/bin/env bash
set -euo pipefail

# Download reviewed Swiss-Prot proteins with EC annotations from UniProt.
OUT_PATH="${1:-data/raw/swissprot/swissprot_ec.tsv}"
FORMAT="${2:-tsv}"
FIELDS="${3:-accession,id,protein_name,ec,go_id,go,organism_id,organism_name,protein_existence,gene_names,sequence,length,cc_function,cc_catalytic_activity,cc_pathway,cc_cofactor,cc_activity_regulation,ft_act_site,ft_binding,xref_pdb,xref_kegg}"
COMPRESS="${4:-false}"

QUERY="reviewed:true%20AND%20ec:*"
BASE_URL="https://rest.uniprot.org/uniprotkb/stream"

mkdir -p "$(dirname "${OUT_PATH}")"

if [[ "${COMPRESS}" == "true" ]]; then
  curl -L --fail "${BASE_URL}?compressed=true&format=${FORMAT}&fields=${FIELDS}&query=${QUERY}" -o "${OUT_PATH}"
else
  curl -L --fail "${BASE_URL}?format=${FORMAT}&fields=${FIELDS}&query=${QUERY}" -o "${OUT_PATH}"
fi

echo "Downloaded Swiss-Prot entries with EC numbers to ${OUT_PATH}"
