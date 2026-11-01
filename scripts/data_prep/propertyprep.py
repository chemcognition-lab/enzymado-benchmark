"""Prepare all benchmark property datasets from their source files."""

from enzymado import paths_and_constants
import enzymado.propertyprep.biolip as biolip
import enzymado.propertyprep.chemistry as chemistry
import enzymado.propertyprep.contracts as contracts
import enzymado.propertyprep.parsing as parsing
import enzymado.propertyprep.pipeline as pipeline
import enzymado.propertyprep.uniprot as uniprot

_P = paths_and_constants.Paths
_C = paths_and_constants.Constants
_PROJECT_ROOT = _P.project_root

sequence_getter = uniprot.sequence_getter
sequence_to_enzyme_metadata = uniprot.sequence_to_enzyme_metadata
_normalize_protein_sequence = uniprot._normalize_protein_sequence
_split_ec_numbers = uniprot._split_ec_numbers
_load_swissprot_sequence_index = uniprot._load_swissprot_sequence_index
_pick_best_local_match = uniprot._pick_best_local_match
_extract_ec_numbers_from_uniprot_json = uniprot._extract_ec_numbers_from_uniprot_json
_get_uniprot_entry_metadata = uniprot._get_uniprot_entry_metadata
_submit_ebi_fasta_job = uniprot._submit_ebi_fasta_job
_poll_ebi_fasta_job = uniprot._poll_ebi_fasta_job
_parse_best_accession_from_ebi_output = uniprot._parse_best_accession_from_ebi_output
_find_accession_by_sequence_similarity = uniprot._find_accession_by_sequence_similarity
_UNIPROT_SEQUENCE_INDEX_CACHE = uniprot._SEQUENCE_INDEX_CACHE
parse_substitution = parsing.parse_substitution
apply_substitutions = parsing.apply_substitutions
parse_active_sites = parsing.parse_active_sites
parse_loc_safe = parsing.parse_loc_safe
parse_binding = parsing.parse_binding
_canonicalize_smiles_column = chemistry.canonicalize_smiles_column


def prep_biolip_binding(
    paths: contracts.Paths | None = None,
    constants: contracts.Constants | None = None,
) -> dict[str, object] | None:
    """Prepare only BioLiP binding using project defaults."""
    paths = paths or _P()
    constants = constants or _C()
    return biolip.prep_biolip_binding(paths, constants)


def main(
    paths: contracts.Paths | None = None,
    constants: contracts.Constants | None = None,
) -> None:
    """Run the property-preparation pipeline with project defaults."""
    paths = paths or _P()
    constants = constants or _C()
    pipeline.run(paths, constants)


if __name__ == "__main__":
    main()
