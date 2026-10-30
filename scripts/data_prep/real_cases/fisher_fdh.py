"""Preprocess Fisher flavin-dependent halogenase NaCl and NaBr panels (Layout B)."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

TASK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("fisher_fdh_nacl", "halogenase_NaCl.csv", "Conversion_NaCl"),
    ("fisher_fdh_nabr", "halogenase_NaBr.csv", "Conversion_NaBr"),
)
FIELDNAMES = [
    "dataset_id",
    "protein_sequence",
    "smiles",
    "conversion",
    "log10_conversion",
]
LOG10_OFFSET = 1e-3


def resolve_fdh_csv(task_name: str, filename: str) -> Path:
    """Locate Fisher FDH raw CSV for a halide condition."""
    return utils.resolve_raw_or_docs(
        f"goldman_family_screens/{filename}",
        raw_rel=f"{task_name}/{filename}",
    )


def build_fdh_rows(
    input_csv: Path,
    *,
    task_name: str,
    target_col: str,
) -> list[dict[str, object]]:
    """Build benchmark rows for a Fisher FDH halide screen."""
    records: list[tuple[tuple[str, ...], float]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = utils.clean_protein_sequence(row.get("SEQ"))
            val = utils.parse_float(row.get(target_col))
            if not utils.is_valid_sequence(seq) or val is None or val < 0.0:
                continue
            smi = utils.canonicalize_smiles(row.get("SUBSTRATES"))
            records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        task_name,
        records,
        target_col="log10_conversion",
        raw_col="conversion",
        log10_offset=LOG10_OFFSET,
    )


def clean_fdh_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
) -> dict[str, Path]:
    """Clean Fisher FDH NaCl and NaBr datasets and write processed CSVs."""
    outputs: dict[str, Path] = {}
    for task_name, filename, target_col in TASK_SPECS:
        csv_path = resolve_fdh_csv(task_name, filename)
        rows = build_fdh_rows(csv_path, task_name=task_name, target_col=target_col)
        outputs[task_name] = utils.write_processed_dataset(
            task_name,
            FIELDNAMES,
            rows,
            output_root=output_root,
        )
    return outputs


def main() -> None:
    """CLI entrypoint for Fisher FDH preprocessing."""
    outputs = clean_fdh_data(output_root=utils.parse_output_root(__doc__))
    for task_name, path in outputs.items():
        print(f"Wrote {task_name}: {path}")


if __name__ == "__main__":
    main()
