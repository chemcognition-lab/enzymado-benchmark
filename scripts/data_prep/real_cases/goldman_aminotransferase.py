"""Preprocess Goldman aminotransferase family specificity dataset (Layout B)."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

TASK_NAME = "goldman_aminotransferase"
FIELDNAMES = ["dataset_id", "protein_sequence", "smiles", "log10_specific_activity"]


def resolve_aminotransferase_csv(input_csv: Path | None = None) -> Path:
    """Locate Goldman aminotransferase raw CSV."""
    return input_csv or utils.resolve_raw_or_docs(
        "goldman_family_screens/aminotransferase.csv",
        raw_rel=f"{TASK_NAME}/aminotransferase.csv",
    )


def build_aminotransferase_rows(input_csv: Path) -> list[dict[str, object]]:
    """Build benchmark rows for Goldman aminotransferase screen."""
    records: list[tuple[tuple[str, ...], float]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = utils.clean_protein_sequence(row.get("SEQ"))
            val = utils.parse_float(row.get("LogSpActivity"))
            if not utils.is_valid_sequence(seq) or val is None:
                continue
            smi = utils.canonicalize_smiles(row.get("SUBSTRATES"))
            records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_specific_activity",
    )


def clean_aminotransferase_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    input_csv: Path | None = None,
) -> Path:
    """Clean Goldman aminotransferase dataset and write processed CSV."""
    csv_path = resolve_aminotransferase_csv(input_csv)
    rows = build_aminotransferase_rows(csv_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Goldman aminotransferase preprocessing."""
    out_path = clean_aminotransferase_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
