"""Preprocess PafA alkaline phosphatase HT-MEK kinetic datasets (Layout A).

Dataset & Processing Rationale:
- Sequence construction & numbering offset: Loads the mature Elizabethkingia
  meningoseptica PafA sequence from pafa_wt.fasta (appending the C-terminal
  "GR" linker residues present in the HT-MEK construct when absent). Because
  HT-MEK mutation tokens use full-length precursor numbering (including the
  17-aa signal peptide), mutations are applied with offset = -17.
- Endpoints & target transformation: Extracts strictly positive measurements
  for kcat (cMUP, s^-1), Km (cMUP, uM), and kcat/Km (M^-1 s^-1), transforming
  each endpoint onto a base-10 logarithmic scale (log10_kcat, log10_km,
  log10_kcat_km).
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

PAFA_ENDPOINTS: dict[str, tuple[str, str]] = {
    "pafa_htmek_kcat": ("kcat_cMUP_s_inv", "log10_kcat"),
    "pafa_htmek_km": ("km_cMUP_uM", "log10_km"),
    "pafa_htmek_kcat_km": ("kcat_over_km_M_s", "log10_kcat_km"),
}


def resolve_pafa_inputs(
    summary_csv: Path | None = None,
    wt_fasta: Path | None = None,
) -> tuple[Path, Path]:
    """Locate PafA HT-MEK summary CSV and wild-type FASTA paths."""
    csv_path = summary_csv or utils.resolve_raw_or_docs(
        "pafa_htmek_kinetics/pafa_htmek_summary.csv",
        raw_rel="pafa_htmek_kcat/pafa_htmek_summary.csv",
    )
    fasta_path = wt_fasta or utils.resolve_raw_or_docs(
        "pafa_htmek_kinetics/pafa_wt.fasta",
        raw_rel="pafa_htmek_kcat/pafa_wt.fasta",
    )
    return csv_path, fasta_path


def build_pafa_endpoint_rows(
    summary_csv: Path,
    wt_fasta: Path,
    task_name: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for a single PafA HT-MEK kinetic endpoint."""
    raw_col, target_col = PAFA_ENDPOINTS[task_name]
    wt_seq = utils.read_single_fasta(wt_fasta)
    if not wt_seq.endswith("GR"):
        wt_seq = f"{wt_seq}GR"
    records: list[tuple[tuple[str, ...], float]] = []

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = utils.parse_float(row.get(raw_col))
            variant = utils.clean_text(row.get("mutant") or row.get("variant"))
            if val is None or val <= 0.0 or not variant:
                continue
            seq = utils.apply_mutations(wt_seq, variant, offset=-17)
            records.append(((seq,), val))

    return utils.build_layout_a_rows(
        task_name,
        records,
        target_col=target_col,
        log10_offset=0.0,
    )


def clean_pafa_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    summary_csv: Path | None = None,
    wt_fasta: Path | None = None,
) -> dict[str, Path]:
    """Clean all 3 PafA HT-MEK kinetic endpoints and write processed CSVs."""
    csv_path, fasta_path = resolve_pafa_inputs(summary_csv, wt_fasta)
    written: dict[str, Path] = {}
    for task_name, (_, target_col) in PAFA_ENDPOINTS.items():
        rows = build_pafa_endpoint_rows(csv_path, fasta_path, task_name)
        fieldnames = [_C.dataset_id, _C.protein_sequence, target_col]
        written[task_name] = utils.write_processed_dataset(
            task_name,
            fieldnames,
            rows,
            output_root=output_root,
        )
    return written


def main() -> None:
    """CLI entrypoint for PafA HT-MEK kinetic preprocessing."""
    written = clean_pafa_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in written.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
