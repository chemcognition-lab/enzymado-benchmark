"""Preprocess Tm9D8* TrpB 3-site combinatorial epistasis dataset (Layout A).

Dataset & Processing Rationale:
- Sequence construction: Uses the Thermotoga maritima TrpB engineered scaffold
  Tm9D8* (Johnston et al., PNAS 2024, Table S3). Combinatorial Library 3A
  simultaneously mutates three active-site positions: 1-based residues 104 (A),
  105 (E), and 106 (T).
- Filtering & target: Stop-codon (*) combinations are excluded, leaving all
  20^3 = 8,000 canonical 3-site amino-acid combinations with measured linear
  enrichment fitness scores.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

TASK_NAME = "trpb3_epistasis"
FIELDNAMES = [_C.dataset_id, _C.protein_sequence, "fitness"]

TM9D8_WT_SEQ = (
    "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARR"
    "LSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAAL"
    "FGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYY"
    "VFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDS"
    "GVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGV"
    "GPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVV"
    "NLSGRGDKDLESVLNHPYVRERIRLEHHHHHH"
)
LIBRARY_3A_POSITIONS = (104, 105, 106)


def resolve_trpb3_csv(csv_path: Path | None = None) -> Path:
    """Locate the Johnston2024 TrpB3A CSV file."""
    return csv_path or utils.resolve_raw_or_docs(
        "trpb3_3site_epistasis/Johnston2024_TrpB3A.csv"
    )


def _combo_to_sequence(combo: str) -> str:
    """Apply a 3-residue Library 3A combination at positions 104, 105, and 106."""
    mut_tokens = [
        f"{TM9D8_WT_SEQ[pos - 1]}{pos}{mut_aa}"
        for pos, mut_aa in zip(LIBRARY_3A_POSITIONS, combo, strict=True)
        if mut_aa != TM9D8_WT_SEQ[pos - 1]
    ]
    if not mut_tokens:
        return TM9D8_WT_SEQ
    return utils.apply_mutations(TM9D8_WT_SEQ, "/".join(mut_tokens))


def build_trpb3_rows(csv_path: Path) -> list[dict[str, object]]:
    """Build benchmark rows for Tm9D8* TrpB Library 3A combinatorial epistasis."""
    records: list[tuple[tuple[str, ...], float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            combo = utils.clean_text(row.get("sequences") or row.get("AACombo"))
            fitness = utils.parse_float(row.get("fitness"))
            if (
                len(combo) != 3
                or fitness is None
                or not set(combo).issubset(utils.CANONICAL_AA)
            ):
                continue
            records.append(((_combo_to_sequence(combo),), fitness))

    return utils.build_layout_a_rows(TASK_NAME, records, target_col="fitness")


def clean_trpb3_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    csv_path: Path | None = None,
) -> Path:
    """Clean Tm9D8* TrpB 3-site combinatorial epistasis data and write CSV."""
    resolved_csv = resolve_trpb3_csv(csv_path)
    rows = build_trpb3_rows(resolved_csv)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Tm9D8* TrpB 3-site epistasis preprocessing."""
    out_path = clean_trpb3_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
