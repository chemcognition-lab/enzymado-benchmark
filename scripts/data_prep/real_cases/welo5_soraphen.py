"""Preprocess WelO5* halogenase Soraphen A conversion dataset (Layout A).

Dataset & Processing Rationale:
- Sequence construction: Applies combinatorial active-site point mutations onto
  the wild-type Hapalosiphon welwitschii Fe(II)/alpha-ketoglutarate-dependent
  halogenase WelO5* sequence (welo5_wt.fasta).
- Target transformation: Soraphen A conversion values include zero-conversion
  variants and span several orders of magnitude, so the regression target is
  modeled as log10(activity + 1e-3).
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

TASK_NAME = "welo5_soraphen_activity"
FIELDNAMES = [_C.dataset_id, _C.protein_sequence, "activity", "log10_activity"]
LOG10_OFFSET = 1e-3


def resolve_welo5_inputs(
    variants_csv: Path | None = None,
    wt_fasta: Path | None = None,
) -> tuple[Path, Path]:
    """Locate WelO5* variants CSV and wild-type FASTA paths."""
    csv_path = variants_csv or utils.resolve_raw_or_docs(
        f"{TASK_NAME}/welo5_variants.csv"
    )
    fasta_path = wt_fasta or utils.resolve_raw_or_docs(f"{TASK_NAME}/welo5_wt.fasta")
    return csv_path, fasta_path


def build_welo5_rows(
    variants_csv: Path,
    wt_fasta: Path,
) -> list[dict[str, object]]:
    """Build benchmark rows for WelO5* Soraphen A conversion."""
    wt_seq = utils.read_single_fasta(wt_fasta)
    records: list[tuple[tuple[str, ...], float]] = []

    with variants_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = utils.parse_float(row.get("soraphen_conversion"))
            if val is None:
                continue
            mutations = utils.clean_text(row.get("variant") or row.get("mutant"))
            seq = utils.apply_mutations(wt_seq, mutations)
            records.append(((seq,), val))

    return utils.build_layout_a_rows(
        TASK_NAME,
        records,
        target_col="log10_activity",
        raw_col="activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_welo5_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    variants_csv: Path | None = None,
    wt_fasta: Path | None = None,
) -> Path:
    """Clean WelO5* data and write processed CSV."""
    csv_path, fasta_path = resolve_welo5_inputs(variants_csv, wt_fasta)
    rows = build_welo5_rows(csv_path, fasta_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for WelO5* Soraphen A preprocessing."""
    out_path = clean_welo5_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
