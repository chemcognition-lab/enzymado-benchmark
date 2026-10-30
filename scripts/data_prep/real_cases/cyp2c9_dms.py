"""Preprocess Human CYP2C9 Click-seq activity and VAMP-seq abundance DMS (Layout A)."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

# Canonical Human Cytochrome P450 2C9 sequence (UniProt: P11712, 490 aa)
CYP2C9_WT_SEQ = (
    "MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKV"
    "YGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKW"
    "KEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICS"
    "IIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFM"
    "KSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTE"
    "TTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYID"
    "LLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFK"
    "KSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVP"
    "PFYQLCFIPV"
)

CYP2C9_TASKS: dict[str, tuple[str, str]] = {
    "cyp2c9_activity": ("cyp2c9_activity_scores.csv", "activity_score"),
    "cyp2c9_abundance": ("cyp2c9_abundance_scores.csv", "abundance_score"),
}


def resolve_cyp2c9_csv(filename: str) -> Path:
    """Locate a CYP2C9 score CSV file."""
    return utils.resolve_raw_or_docs(f"cyp2c9_activity_abundance/{filename}")


def build_cyp2c9_rows(
    scores_csv: Path,
    task_name: str,
    target_col: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for CYP2C9 Click-seq activity or VAMP-seq abundance."""
    records: list[tuple[tuple[str, ...], float]] = []
    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = utils.parse_float(row.get("score"))
            seq = utils.parse_hgvs_missense(str(row.get("hgvs_pro", "")), CYP2C9_WT_SEQ)
            if score is not None and seq is not None:
                records.append(((seq,), score))

    return utils.build_layout_a_rows(task_name, records, target_col=target_col)


def clean_cyp2c9_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
) -> dict[str, Path]:
    """Clean CYP2C9 activity and abundance DMS datasets and write CSVs."""
    written: dict[str, Path] = {}
    for task_name, (filename, target_col) in CYP2C9_TASKS.items():
        csv_path = resolve_cyp2c9_csv(filename)
        rows = build_cyp2c9_rows(csv_path, task_name, target_col)
        fieldnames = ["dataset_id", "protein_sequence", target_col]
        written[task_name] = utils.write_processed_dataset(
            task_name,
            fieldnames,
            rows,
            output_root=output_root,
        )
    return written


def main() -> None:
    """CLI entrypoint for CYP2C9 DMS preprocessing."""
    written = clean_cyp2c9_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in written.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
