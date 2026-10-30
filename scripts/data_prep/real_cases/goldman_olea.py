"""Preprocess Goldman OleA thiolase family specificity dataset (Layout B).

Dataset & Processing Rationale:
- Sequence & organism normalization: Parses the 73 bacterial OleA sequences
  from 73_OleA_JGI_unaligned.fasta.txt, normalizes JGI organism headers (and 6
  strain-name aliases) to match enzyme_substrate_activity_data.csv, and strips
  the 20-aa N-terminal His-tag (MGSSHHHHHHSSGLVPRGSH) while restoring the
  catalytic cysteine (FDVANASLAF -> FDVANACLAF) on crystal-structure entry 4KU5.
- Compound mapping & target: Reconciles the "oxidazole" -> "oxadiazole"
  spelling variant against final_15_cmpnds_tested.csv and maps the reported
  log_slope directly to log_activity.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

TASK_NAME = "goldman_olea_thiolase"
FIELDNAMES = [_C.dataset_id, _C.protein_sequence, _C.smiles, "log_activity"]

ORG_PREFIX_RE = re.compile(r"^.+\.1")
MANUAL_ORG_SUBS: dict[str, str] = {
    "Xanthomonas campestris": "XC",
    "Kocuria varians strain G6": "Kocuria varians",
    "Halobacteriovorax marinus strain BE01": "Halobacteriovorax marinus",
    "Xanthomonas translucens pv. graminis": "Xanthomonas translucens",
    "Enhygromyxa salina strain SWB007": "Enhygromyxa salina",
    "Pseudoxanthomonas sp. NML171200": "Pseudoxanthomonas",
}
COMPOUND_NAME_SUBS: dict[str, str] = {"oxidazole": "oxadiazole"}
HIS_TAG_PREFIX = "MGSSHHHHHHSSGLVPRGSH"


def resolve_olea_inputs(
    activity_csv: Path | None = None,
    compounds_csv: Path | None = None,
    fasta_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Locate OleA activity table, compound SMILES table, and FASTA paths."""
    act_path = activity_csv or utils.resolve_raw_or_docs(
        "goldman_original_datasets/olea_thiolase/enzyme_substrate_activity_data.csv",
        raw_rel=f"{TASK_NAME}/enzyme_substrate_activity_data.csv",
    )
    cmp_path = compounds_csv or utils.resolve_raw_or_docs(
        "goldman_original_datasets/olea_thiolase/final_15_cmpnds_tested.csv",
        raw_rel=f"{TASK_NAME}/final_15_cmpnds_tested.csv",
    )
    fa_path = fasta_path or utils.resolve_raw_or_docs(
        "goldman_original_datasets/olea_thiolase/73_OleA_JGI_unaligned.fasta.txt",
        raw_rel=f"{TASK_NAME}/73_OleA_JGI_unaligned.fasta.txt",
    )
    return act_path, cmp_path, fa_path


def _normalize_olea_header(raw_header: str) -> str:
    """Normalize an OleA FASTA header into the organism identifier used in the CSV."""
    header = ORG_PREFIX_RE.sub("", raw_header.replace("\ufeff", ""))
    header = header.replace("4KU5_", "").replace("_", " ").strip()
    for src, dst in MANUAL_ORG_SUBS.items():
        header = re.sub(src, dst, header)
    return header


def _clean_olea_seq(raw_seq: str) -> str:
    """Clean an OleA protein sequence and strip the N-terminal His-tag if present."""
    seq = utils.clean_protein_sequence(raw_seq)
    if seq.startswith(HIS_TAG_PREFIX):
        return seq[len(HIS_TAG_PREFIX) :].replace("FDVANASLAF", "FDVANACLAF")
    return seq


def _load_olea_organisms(fasta_path: Path) -> dict[str, str]:
    """Parse organism headers and strip N-terminal His tags where present."""
    org_to_parts: dict[str, list[str]] = {}
    current_org: str | None = None

    with fasta_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_org = _normalize_olea_header(line[1:])
                org_to_parts[current_org] = []
            elif current_org is not None:
                org_to_parts[current_org].append(line)

    return {
        org: _clean_olea_seq("".join(parts))
        for org, parts in org_to_parts.items()
        if parts
    }


def _load_olea_compounds(compounds_csv: Path) -> dict[str, str]:
    """Load canonical SMILES mapped by compound abbreviation."""
    cmp_to_smiles: dict[str, str] = {}
    with compounds_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = utils.clean_text(row.get("cmpnd_abbrev"))
            raw_smi = utils.clean_text(row.get("SMILES"))
            if name and raw_smi:
                mapped_name = COMPOUND_NAME_SUBS.get(name, name)
                cmp_to_smiles[mapped_name] = utils.canonicalize_smiles(raw_smi)
    return cmp_to_smiles


def build_olea_rows(
    activity_csv: Path,
    compounds_csv: Path,
    fasta_path: Path,
) -> list[dict[str, object]]:
    """Build benchmark rows for Goldman OleA thiolase screen."""
    org_to_seq = _load_olea_organisms(fasta_path)
    cmp_to_smiles = _load_olea_compounds(compounds_csv)
    records: list[tuple[tuple[str, ...], float]] = []

    with activity_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = org_to_seq.get(utils.clean_text(row.get("org")))
            smi = cmp_to_smiles.get(utils.clean_text(row.get("cmpnd")))
            val = utils.parse_float(row.get("log_slope"))
            if seq is not None and smi is not None and val is not None:
                records.append(((seq, smi), val))

    return utils.build_layout_b_rows(TASK_NAME, records, target_col="log_activity")


def clean_olea_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    activity_csv: Path | None = None,
    compounds_csv: Path | None = None,
    fasta_path: Path | None = None,
) -> Path:
    """Clean Goldman OleA dataset and write processed CSV."""
    act_path, cmp_path, fa_path = resolve_olea_inputs(
        activity_csv,
        compounds_csv,
        fasta_path,
    )
    rows = build_olea_rows(act_path, cmp_path, fa_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Goldman OleA thiolase preprocessing."""
    out_path = clean_olea_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
