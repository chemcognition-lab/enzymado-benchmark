"""Preprocess Huang HAD phosphatase cap-domain specificity panels (Layout B)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from scripts.data_prep.real_cases import utils

TASK_CAP_MAP: dict[str, str] = {
    "C0": "huang_had_phosphatase_c0",
    "C1": "huang_had_phosphatase_c1",
    "C2": "huang_had_phosphatase_c2",
}
FIELDNAMES = [
    "dataset_id",
    "protein_sequence",
    "smiles",
    "conversion",
    "log10_conversion",
]
LOG10_OFFSET = 1e-3


def resolve_had_inputs(
    chiral_csv: Path | None = None,
    seq_tsv: Path | None = None,
    meta_xlsx: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Locate HAD phosphatase chiral matrix, sequence table, and metadata Excel."""
    csv_path = chiral_csv or utils.resolve_raw_or_docs(
        "goldman_family_screens/phosphatase_chiral.csv"
    )
    tsv_path = seq_tsv or utils.resolve_raw_or_docs(
        "goldman_original_datasets/phosphatase_had/phosphatase_sequence_file.txt"
    )
    xlsx_path = meta_xlsx or utils.resolve_raw_or_docs(
        "goldman_original_datasets/phosphatase_had/pnas.1423570112.sd01.xlsx",
        raw_rel="huang_had_phosphatase/pnas.1423570112.sd01.xlsx",
    )
    return csv_path, tsv_path, xlsx_path


def _classify_cap_clade(cap: str, sfld: str) -> str | None:
    """Classify a HAD phosphatase into C0, C1 (C1.5 subgroup), or C2 cap clade."""
    match cap:
        case "C0":
            return "C0"
        case "C2":
            return "C2"
        case "C1" if sfld.startswith("C1.5"):
            return "C1"
        case _:
            return None


def _load_seq_to_cap_clade(seq_tsv: Path, meta_xlsx: Path) -> dict[str, str]:
    """Map cleaned protein sequence to disjoint cap-domain clades (C0, C1, C2)."""
    meta_df = pd.read_excel(meta_xlsx, sheet_name="Supplemental 2")
    uniprot_to_clade: dict[str, str] = {}
    for _, row in meta_df.iterrows():
        uid = utils.clean_text(row.get("Uniprot ID"))
        clade = _classify_cap_clade(
            utils.clean_text(row.get("Cap Type")),
            utils.clean_text(row.get("SFLD Subgroup")),
        )
        if uid and clade is not None:
            uniprot_to_clade[uid] = clade

    seq_to_clade: dict[str, str] = {}
    with seq_tsv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            uid = utils.clean_text(row.get("Entry"))
            seq = utils.clean_protein_sequence(row.get("Sequence"))
            clade = uniprot_to_clade.get(uid)
            if clade is not None and utils.is_valid_sequence(seq):
                seq_to_clade[seq] = clade
    return seq_to_clade


def build_had_rows_by_clade(
    chiral_csv: Path,
    seq_tsv: Path,
    meta_xlsx: Path,
) -> dict[str, list[dict[str, object]]]:
    """Build benchmark rows partitioned into C0, C1 (C1.5), and C2 HAD clades."""
    seq_to_clade = _load_seq_to_cap_clade(seq_tsv, meta_xlsx)
    clade_records: dict[str, list[tuple[tuple[str, ...], float]]] = {
        "C0": [],
        "C1": [],
        "C2": [],
    }
    smi_cache: dict[str, str] = {}

    with chiral_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = utils.clean_protein_sequence(row.get("SEQ"))
            clade = seq_to_clade.get(seq)
            val = utils.parse_float(row.get("Conversion"))
            raw_smi = utils.clean_text(row.get("SUBSTRATES"))
            if clade is None or val is None or val < 0.0 or not raw_smi:
                continue
            smi = smi_cache.get(raw_smi) or utils.canonicalize_smiles(
                raw_smi,
                neutralize_alkoxide=True,
            )
            smi_cache[raw_smi] = smi
            clade_records[clade].append(((seq, smi), val))

    return {
        task_name: utils.build_layout_b_rows(
            task_name,
            clade_records[clade],
            target_col="log10_conversion",
            raw_col="conversion",
            log10_offset=LOG10_OFFSET,
        )
        for clade, task_name in TASK_CAP_MAP.items()
    }


def clean_had_phosphatase_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    chiral_csv: Path | None = None,
    seq_tsv: Path | None = None,
    meta_xlsx: Path | None = None,
) -> dict[str, Path]:
    """Clean HAD phosphatase datasets and write C0, C1, and C2 processed CSVs."""
    csv_path, tsv_path, xlsx_path = resolve_had_inputs(chiral_csv, seq_tsv, meta_xlsx)
    rows_by_task = build_had_rows_by_clade(csv_path, tsv_path, xlsx_path)
    return {
        task_name: utils.write_processed_dataset(
            task_name,
            FIELDNAMES,
            rows,
            output_root=output_root,
        )
        for task_name, rows in rows_by_task.items()
    }


def main() -> None:
    """CLI entrypoint for Huang HAD phosphatase preprocessing."""
    outputs = clean_had_phosphatase_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in outputs.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
