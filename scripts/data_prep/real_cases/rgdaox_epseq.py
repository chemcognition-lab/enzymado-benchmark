"""Preprocess RgDAOx EP-Seq activity and stability DMS datasets (Layout A).

Dataset & Processing Rationale:
- Sequence reconstruction: The 365-aa Rhodotorula gracilis D-amino acid oxidase
  (RgDAOx) wild-type sequence is reconstructed directly from the "Position" and
  "wildtype_residue" columns of DMS_scores_DAOX.csv (with N-terminal M1).
- Endpoints & filtering: Nonsense/stop mutations are excluded; missense and WT
  rows are split into two tasks corresponding to "Activity score (log)"
  (daox_epseq_activity) and "Expression score (log)" (daox_epseq_stability).
  Scores are already reported on a log scale in the source table.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

DAOX_ENDPOINTS: dict[str, tuple[str, str]] = {
    "daox_epseq_activity": ("Activity score (log)", "log_activity"),
    "daox_epseq_stability": ("Expression score (log)", "log_stability"),
}


def resolve_daox_csv(dms_csv: Path | None = None) -> Path:
    """Locate the RgDAOx DMS scores CSV."""
    return dms_csv or utils.resolve_raw_or_docs(
        "rgdaox_epseq_activity_stability/DMS_scores_DAOX.csv"
    )


def reconstruct_daox_wt(dms_csv: Path) -> str:
    """Reconstruct the 365-aa RgDAOx wild-type sequence from Position and WT residue."""
    pos_to_wt: dict[int, str] = {1: "M"}
    with dms_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pos_str = utils.clean_text(row.get("Position"))
            wt_aa = utils.clean_text(row.get("wildtype_residue"))
            if not pos_str or not wt_aa or pos_str.upper() == "WT":
                continue
            pos_to_wt[int(float(pos_str))] = wt_aa
    return "".join(pos_to_wt[idx] for idx in range(1, max(pos_to_wt) + 1))


def build_daox_endpoint_rows(
    dms_csv: Path,
    task_name: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for RgDAOx EP-Seq activity or stability."""
    raw_col, target_col = DAOX_ENDPOINTS[task_name]
    wt_seq = reconstruct_daox_wt(dms_csv)
    records: list[tuple[tuple[str, ...], float]] = []

    with dms_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = utils.parse_float(row.get(raw_col))
            if val is None:
                continue
            if utils.clean_text(row.get("Mutation")).upper() == "WT":
                records.append(((wt_seq,), val))
                continue
            mut_aa = utils.clean_text(row.get("mutant_residue"))
            wt_aa = utils.clean_text(row.get("wildtype_residue"))
            pos_str = utils.clean_text(row.get("Position"))
            if mut_aa not in utils.CANONICAL_AA or not wt_aa or not pos_str:
                continue
            pos = int(float(pos_str))
            seq = utils.apply_mutations(wt_seq, f"{wt_aa}{pos}{mut_aa}")
            records.append(((seq,), val))

    return utils.build_layout_a_rows(task_name, records, target_col=target_col)


def clean_daox_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    dms_csv: Path | None = None,
) -> dict[str, Path]:
    """Clean RgDAOx EP-Seq activity and stability datasets and write CSVs."""
    csv_path = resolve_daox_csv(dms_csv)
    written: dict[str, Path] = {}
    for task_name, (_, target_col) in DAOX_ENDPOINTS.items():
        rows = build_daox_endpoint_rows(csv_path, task_name)
        fieldnames = [_C.dataset_id, _C.protein_sequence, target_col]
        written[task_name] = utils.write_processed_dataset(
            task_name,
            fieldnames,
            rows,
            output_root=output_root,
        )
    return written


def main() -> None:
    """CLI entrypoint for RgDAOx EP-Seq preprocessing."""
    written = clean_daox_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in written.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
