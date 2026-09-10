"""Prepare all benchmark property datasets from their source files."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import enzymado.paths_and_constants as paths_and_constants  # noqa: E402
import enzymado.propertyprep.biolip as biolip  # noqa: E402
import enzymado.propertyprep.chemistry as chemistry  # noqa: E402
import enzymado.propertyprep.contracts as contracts  # noqa: E402
import enzymado.propertyprep.parsing as parsing  # noqa: E402
import enzymado.propertyprep.pipeline as pipeline  # noqa: E402
import enzymado.propertyprep.uniprot as uniprot  # noqa: E402

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
    paths = paths or paths_and_constants.Paths()
    constants = constants or paths_and_constants.Constants()
    return biolip.prep_biolip_binding(paths, constants)


def main(
    paths: contracts.Paths | None = None,
    constants: contracts.Constants | None = None,
) -> None:
    """Run the property-preparation pipeline with project defaults."""
    paths = paths or paths_and_constants.Paths()
    constants = constants or paths_and_constants.Constants()
    pipeline.run(paths, constants)


if __name__ == "__main__":
    main()
