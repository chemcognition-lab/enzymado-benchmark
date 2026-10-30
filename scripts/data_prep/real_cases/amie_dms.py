"""Preprocess Pseudomonas aeruginosa AmiE multi-substrate DMS datasets (Layout A).

Dataset & Processing Rationale:
- Sequence construction: Uses the 346-aa Pseudomonas aeruginosa aliphatic
  amidase AmiE wild-type sequence (UniProt: P11436) and applies 1-based
  single-residue substitutions from Wrenbeck et al. (Nat. Commun. 2017).
- Substrate-specific tasks: Because all variants in each selection experiment
  are assayed against a single amide substrate (Acetamide, Propionamide, or
  Isobutyramide), each selection screen is modeled as an independent Layout A
  task predicting normalized_fitness. Synonymous wild-type controls are
  averaged into the wild-type sequence entry.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

AMIE_WT_SEQ = (
    "MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNARKIADMIVGMKQGLPGMDLVVFPEY"
    "SLQGIMYDPAEMMETAVAIPGEETEIFSRACRKANVWGVFSLTGERHEEHPRKAPYNTLVL"
    "IDNNGEIVQKYRKIIPWCPIEGWYPGGQTYVSEGPKGMKISLIICDDGNYPEIWRDCAMK"
    "GAELIVRCQGYMYPAKDQQVMMAKAMAWANNCYVAVANAAGFDGVYSYFGHSAIIGFDGR"
    "TLGECGEEEMGIQYAQLSLSQIRDARANDQSQNHLFKILHRGYSGLQASGDGDRGLAECP"
    "FEFYRTWVTDAEKARENVERLTRSTTGVAQCPVGRLPYEGLEKEA"
)

AMIE_SUBSTRATE_FILES: dict[str, str] = {
    "amie_acetamide_dms": "amiESelectionFitnessData_Acetamide.txt",
    "amie_propionamide_dms": "amiESelectionFitnessData_Propionamide.txt",
    "amie_isobutyramide_dms": "amiESelectionFitnessData_Isobutyramide.txt",
}

FIELDNAMES = [_C.dataset_id, _C.protein_sequence, "normalized_fitness"]


def resolve_amie_tsv(filename: str) -> Path:
    """Locate an AmiE selection fitness TSV file."""
    return utils.resolve_raw_or_docs(f"amie_multisubstrate_dms/{filename}")


def build_amie_substrate_rows(
    tsv_path: Path,
    task_name: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for AmiE DMS on a single amide substrate."""
    records: list[tuple[tuple[str, ...], float]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            fitness = utils.parse_float(row.get("normalized_fitness"))
            pos_val = utils.parse_float(row.get("location"))
            mut_aa = utils.clean_text(row.get("mutation"))
            if fitness is None or pos_val is None or mut_aa not in utils.CANONICAL_AA:
                continue
            pos = int(pos_val)
            wt_aa = AMIE_WT_SEQ[pos - 1]
            seq = (
                AMIE_WT_SEQ
                if mut_aa == wt_aa
                else utils.apply_mutations(AMIE_WT_SEQ, f"{wt_aa}{pos}{mut_aa}")
            )
            records.append(((seq,), fitness))

    return utils.build_layout_a_rows(
        task_name,
        records,
        target_col="normalized_fitness",
    )


def clean_amie_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
) -> dict[str, Path]:
    """Clean AmiE DMS datasets across all 3 substrates and write CSVs."""
    written: dict[str, Path] = {}
    for task_name, filename in AMIE_SUBSTRATE_FILES.items():
        tsv_path = resolve_amie_tsv(filename)
        rows = build_amie_substrate_rows(tsv_path, task_name)
        written[task_name] = utils.write_processed_dataset(
            task_name,
            FIELDNAMES,
            rows,
            output_root=output_root,
        )
    return written


def main() -> None:
    """CLI entrypoint for AmiE multi-substrate DMS preprocessing."""
    written = clean_amie_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in written.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
