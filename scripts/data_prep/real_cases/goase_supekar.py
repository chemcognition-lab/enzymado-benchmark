"""Preprocess Galactose Oxidase variant-substrate activity matrix (Layout B)."""

from __future__ import annotations

import csv
from pathlib import Path
import zipfile

from scripts.data_prep.real_cases import utils

TASK_NAME = "goase_variant_substrate"
FIELDNAMES = [
    "dataset_id",
    "protein_sequence",
    "smiles",
    "activity",
    "log10_activity",
]
LOG10_OFFSET = 1e-3

PDB_INTERNAL_PATH = "GOase_ML_pred_workflow/dock_and_score/receptors/pdb_f/1GOG.pdb"
SMI_INTERNAL_PATH = "GOase_ML_pred_workflow/dock_and_score/substrates/panel56.smi"
MUT_INTERNAL_PREFIX = "GOase_ML_pred_workflow/dock_and_score/receptors/files/GOh"


def resolve_goase_inputs(
    matrix_csv: Path | None = None,
    workflow_zip: Path | None = None,
) -> tuple[Path, Path]:
    """Locate GOase dataset CSV and workflow ZIP archive."""
    csv_path = matrix_csv or utils.resolve_raw_or_docs(
        "goase_variant_substrates/dataset_56-subs_34-vars.csv",
        raw_rel=f"{TASK_NAME}/dataset_56-subs_34-vars.csv",
    )
    zip_path = workflow_zip or utils.resolve_raw_or_docs(
        "goase_variant_substrates/GOase_ML_pred_workflow.zip",
        raw_rel=f"{TASK_NAME}/GOase_ML_pred_workflow.zip",
    )
    return csv_path, zip_path


def _extract_wt_from_pdb(pdb_text: str) -> str:
    """Extract 1-based wild-type sequence from CA atoms in 1GOG.pdb."""
    residues: dict[int, str] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_name = line[17:20].strip().capitalize()
            res_num = int(line[22:26].strip())
            residues[res_num] = utils.AA3_TO_AA1[res_name]
    return "".join(residues[idx] for idx in sorted(residues))


def _load_goase_zip_metadata(
    workflow_zip: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load variant sequences and canonical substrate SMILES from workflow zip."""
    with zipfile.ZipFile(workflow_zip) as archive:
        pdb_text = archive.read(PDB_INTERNAL_PATH).decode("utf-8")
        wt_seq = _extract_wt_from_pdb(pdb_text)

        smiles_map: dict[str, str] = {}
        smi_text = archive.read(SMI_INTERNAL_PATH).decode("utf-8")
        for line in smi_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw_smi, sub_id = stripped.split()
            smiles_map[sub_id] = utils.canonicalize_smiles(
                raw_smi,
                neutralize_alkoxide=True,
            )

        variant_map: dict[str, str] = {}
        for entry_name in archive.namelist():
            if entry_name.startswith(MUT_INTERNAL_PREFIX) and entry_name.endswith(
                ".txt"
            ):
                var_id = Path(entry_name).stem
                mut_str = archive.read(entry_name).decode("utf-8").strip()
                variant_map[var_id] = utils.apply_mutations(
                    wt_seq,
                    mut_str,
                    verify_wt=False,
                )

    return variant_map, smiles_map


def build_goase_rows(
    matrix_csv: Path,
    workflow_zip: Path,
) -> list[dict[str, object]]:
    """Build benchmark rows for GOase variant-substrate activity."""
    variant_map, smiles_map = _load_goase_zip_metadata(workflow_zip)
    records: list[tuple[tuple[str, ...], float]] = []

    with matrix_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name_token = utils.clean_text(row.get("name"))
            val = utils.parse_float(row.get("labels"))
            if not name_token or val is None or "_" not in name_token:
                continue
            var_id, sub_id = name_token.split("_", maxsplit=1)
            seq = variant_map.get(var_id)
            smi = smiles_map.get(sub_id)
            if seq is not None and smi is not None:
                records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_activity",
        raw_col="activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_goase_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    matrix_csv: Path | None = None,
    workflow_zip: Path | None = None,
) -> Path:
    """Clean GOase variant-substrate dataset and write processed CSV."""
    csv_path, zip_path = resolve_goase_inputs(matrix_csv, workflow_zip)
    rows = build_goase_rows(csv_path, zip_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for GOase variant-substrate preprocessing."""
    out_path = clean_goase_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
