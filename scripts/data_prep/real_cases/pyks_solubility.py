"""Preprocess PyKS polyketide synthase soluble expression DMS dataset (Layout A)."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

TASK_NAME = "pyks_solubility"
FIELDNAMES = ["dataset_id", "protein_sequence", "normalized_er"]


def resolve_pyks_csv(csv_path: Path | None = None) -> Path:
    """Locate the PyKS solubility DMS supplementary CSV."""
    return csv_path or utils.resolve_raw_or_docs(
        "supplementary_tables/pyks_wrenbeck2019_NIHMS1058253-supplement-DataS1.csv"
    )


def reconstruct_pyks_wt(csv_path: Path) -> str:
    """Reconstruct the 439-aa PyKS wild-type sequence from synonymous WT rows."""
    pos_to_wt: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if utils.clean_text(row.get("Normalized_ER")) != "0":
                continue
            pos_val = utils.parse_float(row.get("Location"))
            aa = utils.clean_text(row.get("Mutation"))
            if pos_val is not None and aa in utils.CANONICAL_AA:
                pos_to_wt[int(pos_val)] = aa
    return "".join(pos_to_wt[idx] for idx in range(min(pos_to_wt), max(pos_to_wt) + 1))


def build_pyks_rows(csv_path: Path) -> list[dict[str, object]]:
    """Build benchmark rows for PyKS soluble expression DMS."""
    wt_seq = reconstruct_pyks_wt(csv_path)
    records: list[tuple[tuple[str, ...], float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = utils.parse_float(row.get("Normalized_ER"))
            pos_val = utils.parse_float(row.get("Location"))
            mut_aa = utils.clean_text(row.get("Mutation"))
            if val is None or pos_val is None or mut_aa not in utils.CANONICAL_AA:
                continue
            pos_1based = int(pos_val) + 1
            wt_aa = wt_seq[pos_1based - 1]
            seq = (
                wt_seq
                if mut_aa == wt_aa
                else utils.apply_mutations(wt_seq, f"{wt_aa}{pos_1based}{mut_aa}")
            )
            records.append(((seq,), val))

    return utils.build_layout_a_rows(TASK_NAME, records, target_col="normalized_er")


def clean_pyks_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    csv_path: Path | None = None,
) -> Path:
    """Clean PyKS solubility DMS data and write processed CSV."""
    resolved_csv = resolve_pyks_csv(csv_path)
    rows = build_pyks_rows(resolved_csv)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for PyKS solubility DMS preprocessing."""
    out_path = clean_pyks_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
