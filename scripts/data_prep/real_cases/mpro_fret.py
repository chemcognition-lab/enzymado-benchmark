"""Preprocess SARS-CoV-2 Mpro FRET cleavage DMS dataset (Layout A)."""

from __future__ import annotations

from pathlib import Path

import openpyxl
from scripts.data_prep.real_cases import utils

TASK_NAME = "mpro_fret_cleavage"
FIELDNAMES = ["dataset_id", "protein_sequence", "functional_score"]

# SARS-CoV-2 3CLpro / Mpro wild-type sequence (PDB: 6LU7 Chain A, 306 aa)
MPRO_WT_SEQ = (
    "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIR"
    "KSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNG"
    "SPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGN"
    "FYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYE"
    "PLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQC"
    "SGVTFQ"
)


def resolve_mpro_xlsx(xlsx_path: Path | None = None) -> Path:
    """Locate the SARS-CoV-2 Mpro FRET cleavage Excel workbook."""
    return xlsx_path or utils.resolve_raw_or_docs(
        f"{TASK_NAME}/elife-77433-fig2-data1-v2.xlsx"
    )


def _parse_mpro_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
) -> list[tuple[tuple[str, ...], float]]:
    """Extract (sequence, score) records from a single Mpro FRET replicate sheet."""
    rows_iter = ws.iter_rows(values_only=True)
    headers = [utils.clean_text(c) for c in next(rows_iter)]
    pos_idx = headers.index("Position")
    aa_idx = headers.index("aa")
    score_idx = headers.index("functional score (normalized)")

    records: list[tuple[tuple[str, ...], float]] = []
    for row in rows_iter:
        pos_token = utils.clean_text(row[pos_idx])
        mut_aa = utils.clean_text(row[aa_idx])
        score = utils.parse_float(row[score_idx])
        if not pos_token or score is None:
            continue
        if pos_token.upper() == "WT":
            records.append(((MPRO_WT_SEQ,), score))
            continue
        pos_val = utils.parse_float(pos_token)
        if pos_val is None or mut_aa not in utils.CANONICAL_AA:
            continue
        pos = int(pos_val)
        wt_aa = MPRO_WT_SEQ[pos - 1]
        seq = (
            MPRO_WT_SEQ
            if mut_aa == wt_aa
            else utils.apply_mutations(MPRO_WT_SEQ, f"{wt_aa}{pos}{mut_aa}")
        )
        records.append(((seq,), score))
    return records


def build_mpro_rows(xlsx_path: Path) -> list[dict[str, object]]:
    """Build benchmark rows for SARS-CoV-2 Mpro FRET cleavage DMS."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    records: list[tuple[tuple[str, ...], float]] = []
    for sheet_name in ("FRET scores rep 1", "FRET scores rep 2"):
        records.extend(_parse_mpro_sheet(wb[sheet_name]))
    wb.close()

    return utils.build_layout_a_rows(TASK_NAME, records, target_col="functional_score")


def clean_mpro_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    xlsx_path: Path | None = None,
) -> Path:
    """Clean SARS-CoV-2 Mpro FRET cleavage DMS data and write processed CSV."""
    resolved_xlsx = resolve_mpro_xlsx(xlsx_path)
    rows = build_mpro_rows(resolved_xlsx)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for SARS-CoV-2 Mpro FRET DMS preprocessing."""
    out_path = clean_mpro_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
