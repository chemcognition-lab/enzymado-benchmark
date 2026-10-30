"""Preprocess SARS-CoV-2 PLpro activity and abundance DMS datasets (Layout A)."""

from __future__ import annotations

from pathlib import Path

import openpyxl
from scripts.data_prep.real_cases import utils

PLPRO_ENDPOINTS: dict[str, tuple[str, str]] = {
    "plpro_activity": ("fitness_activity", "fitness_activity"),
    "plpro_abundance": ("fitness_abundance", "fitness_abundance"),
}


def resolve_plpro_xlsx(xlsx_path: Path | None = None) -> Path:
    """Locate the SARS-CoV-2 PLpro supplementary Excel workbook."""
    return xlsx_path or utils.resolve_raw_or_docs(
        "supplementary_tables/plpro_flynn2024_MOESM9_ESM.xlsx"
    )


def _advance_to_plpro_headers(rows_iter: object) -> list[str]:
    """Advance worksheet iterator to the header row containing 'Pos'."""
    for row in rows_iter:  # type: ignore[attr-defined]
        cleaned = [utils.clean_text(c) for c in row]
        if "Pos" in cleaned:
            return cleaned
    raise ValueError("Could not find 'Pos' header row in PLpro worksheet")


def reconstruct_plpro_wt(xlsx_path: Path) -> str:
    """Reconstruct the 315-aa SARS-CoV-2 PLpro wild-type sequence from Fig. 3."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    rows_iter = wb["Fig. 3"].iter_rows(values_only=True)
    headers = _advance_to_plpro_headers(rows_iter)
    pos_idx = headers.index("Pos")
    wt_idx = headers.index("WT_AA")

    pos_to_wt: dict[int, str] = {}
    for row in rows_iter:
        pos_val = utils.parse_float(row[pos_idx])
        wt_aa = utils.clean_text(row[wt_idx])
        if pos_val is not None and wt_aa in utils.CANONICAL_AA:
            pos_to_wt[int(pos_val)] = wt_aa
    wb.close()
    return "".join(pos_to_wt[idx] for idx in range(1, max(pos_to_wt) + 1))


def build_plpro_endpoint_rows(
    xlsx_path: Path,
    task_name: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for SARS-CoV-2 PLpro activity or abundance."""
    raw_col, target_col = PLPRO_ENDPOINTS[task_name]
    wt_seq = reconstruct_plpro_wt(xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    rows_iter = wb["Fig. 3"].iter_rows(values_only=True)
    headers = _advance_to_plpro_headers(rows_iter)
    pos_idx = headers.index("Pos")
    wt_idx = headers.index("WT_AA")
    mut_idx = headers.index("Mut")
    val_idx = headers.index(raw_col)

    records: list[tuple[tuple[str, ...], float]] = []
    for row in rows_iter:
        val = utils.parse_float(row[val_idx])
        pos_val = utils.parse_float(row[pos_idx])
        wt_aa = utils.clean_text(row[wt_idx])
        mut_aa = utils.clean_text(row[mut_idx])
        if val is None or pos_val is None or mut_aa not in utils.CANONICAL_AA:
            continue
        pos = int(pos_val)
        seq = utils.apply_mutations(wt_seq, f"{wt_aa}{pos}{mut_aa}")
        records.append(((seq,), val))
    wb.close()

    return utils.build_layout_a_rows(task_name, records, target_col=target_col)


def clean_plpro_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    xlsx_path: Path | None = None,
) -> dict[str, Path]:
    """Clean SARS-CoV-2 PLpro activity and abundance datasets and write CSVs."""
    resolved_xlsx = resolve_plpro_xlsx(xlsx_path)
    written: dict[str, Path] = {}
    for task_name, (_, target_col) in PLPRO_ENDPOINTS.items():
        rows = build_plpro_endpoint_rows(resolved_xlsx, task_name)
        fieldnames = ["dataset_id", "protein_sequence", target_col]
        written[task_name] = utils.write_processed_dataset(
            task_name,
            fieldnames,
            rows,
            output_root=output_root,
        )
    return written


def main() -> None:
    """CLI entrypoint for SARS-CoV-2 PLpro DMS preprocessing."""
    written = clean_plpro_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in written.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
