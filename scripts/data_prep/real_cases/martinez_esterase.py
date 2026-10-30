"""Preprocess Martínez-Martínez microbial esterase substrate panel (Layout B).

Dataset & Processing Rationale:
- Sequence & length filtering: Amino acid sequences are loaded from
  Supplementary Table S1 ("ID enzyme" -> "AMINO ACID SEQUENCE") and filtered to
  canonical sequences shorter than 1,000 residues to fit within PLM context
  limits.
- Substrate matrix: Crosses the filtered microbial esterases against the
  ester substrate panel in Supplementary Table S3 ("Smiles code").
- Target transformation: Specific activities (U/g) include inactive pairs (0.0)
  and span several orders of magnitude, so the target is modeled as
  log10(specific_activity + 1e-3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

TASK_NAME = "martinez_esterase_panel"
FIELDNAMES = [
    _C.dataset_id,
    _C.protein_sequence,
    _C.smiles,
    "specific_activity",
    "log10_specific_activity",
]
LOG10_OFFSET = 1e-3
MAX_SEQ_LEN = 1000


def resolve_esterase_xls(input_xls: Path | None = None) -> Path:
    """Locate Martínez-Martínez esterase supplementary Excel workbook."""
    return input_xls or utils.resolve_raw_or_docs(
        "goldman_original_datasets/esterase/cb7b00996_si_002.xls",
        raw_rel=f"{TASK_NAME}/cb7b00996_si_002.xls",
    )


def _load_esterase_sequences(input_xls: Path) -> dict[str, str]:
    """Load valid enzyme ID to protein sequence mappings from Table S1."""
    seq_df = pd.read_excel(input_xls, sheet_name="Supplementary Table S1")
    id_to_seq: dict[str, str] = {}
    for _, row in seq_df.iterrows():
        enz_id = utils.clean_text(row.get("ID enzyme"))
        seq = utils.clean_protein_sequence(row.get("AMINO ACID SEQUENCE"))
        if enz_id and utils.is_valid_sequence(seq) and len(seq) < MAX_SEQ_LEN:
            id_to_seq[enz_id] = seq
    return id_to_seq


def build_esterase_rows(input_xls: Path) -> list[dict[str, object]]:
    """Build benchmark rows for Martínez-Martínez esterase specificity panel."""
    id_to_seq = _load_esterase_sequences(input_xls)
    val_df = pd.read_excel(input_xls, sheet_name="Supplementary Table S3")
    enz_cols = [col for col in val_df.columns if utils.clean_text(col) in id_to_seq]

    records: list[tuple[tuple[str, ...], float]] = []
    for _, row in val_df.iterrows():
        raw_smi = utils.clean_text(row.get("Smiles code"))
        if not raw_smi:
            continue
        smi = utils.canonicalize_smiles(raw_smi)
        for col in enz_cols:
            seq = id_to_seq[utils.clean_text(col)]
            val = utils.parse_float(row.get(col))
            if val is not None and val >= 0.0:
                records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_specific_activity",
        raw_col="specific_activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_esterase_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    input_xls: Path | None = None,
) -> Path:
    """Clean Martínez-Martínez esterase dataset and write processed CSV."""
    xls_path = resolve_esterase_xls(input_xls)
    rows = build_esterase_rows(xls_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Martínez-Martínez esterase preprocessing."""
    out_path = clean_esterase_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
