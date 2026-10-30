"""Preprocess McbA amide synthetase 9-drug single-mutant matrix (Layout B).

Dataset & Processing Rationale:
- Sequence reconstruction: Wild-type McbA (518 aa) is translated directly from
  the codon-optimized coding DNA sequence reported in Table S6 (ATG = M1).
  Single-site saturation variants across active-site residues are reconstructed
  from each worksheet's column headers and amino-acid row labels.
- Substrate panel: Evaluates 9 pharmaceutical amide synthesis targets across
  worksheets Fig3b-d and FigS18-S23 (Moclobemide, Metoclopramide, Cinchocaine,
  Itopride, Declopramide, Trimethobenzamide, (S)-Sulpiride, Procainamide, and
  Troxipide), represented by their canonicalized product SMILES.
- Target transformation: Negative baseline-subtracted relative activities are
  clamped at 0.0 and transformed via log10(activity + 1e-3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scripts.data_prep.real_cases import utils

from enzymado import paths_and_constants

_C = paths_and_constants.Constants

TASK_NAME = "mcba_reaction_activity"
FIELDNAMES = [
    _C.dataset_id,
    _C.protein_sequence,
    _C.smiles,
    "activity",
    "log10_activity",
]
LOG10_OFFSET = 1e-3

WT_MCBA_DNA = (
    "ATGGAGAAAAAAATCTGGAGCCATCCGCAGTTCGAAAAAGGCGGATCCGG"
    "AGAAAACCTGTATTTCCAGGGCGGTTACGCTCGTCGTGTAATGGATGGTAT"
    "CGGTGAAGTAGCGGTAACTGGCGCTGGTGGTTCTGTAACTGGTGCGCGTC"
    "TGCGCCATCAGGTTCGTCTGCTGGCTCATGCTCTGACCGAAGCGGGTATT"
    "CCGCCAGGCCGTGGTGTAGCATGTCTGCATGCTAACACCTGGCGTGCGAT"
    "CGCACTGCGTCTGGCTGTTCAGGCGATTGGTTGCCACTATGTTGGTCTGC"
    "GTCCTACCGCTGCTGTTACTGAACAGGCACGCGCAATTGCGGCTGCTGAT"
    "TCTGCCGCACTGGTTTTCGAACCAAGCGTTGAAGCTCGTGCAGCTGACCT"
    "GCTGGAACGTGTTTCTGTGCCGGTTGTGCTGTCTCTGGGTCCGACCTCTC"
    "GTGGCCGTGATATCCTGGCAGCTAGCGTTCCGGAAGGTACGCCGCTGCG"
    "TTACCGTGAACACCCAGAAGGTATCGCAGTTGTAGCCTTTACTAGCGGCAC"
    "CACTGGCACCCCTAAAGGCGTTGCCCACTCCTCTACCGCTATGAGCGCTT"
    "GTGTGGATGCTGCGGTTTCCATGTACGGTCGCGGTCCTTGGCGTTTCCTG"
    "ATCCCGATCCCTCTGTCTGACCTGGGTGGCGAACTGGCACAGTGTACCCT"
    "GGCTACCGGCGGTACCGTTGTGCTGCTGGAAGAGTTCCAACCGGACGCC"
    "GTTCTGGAAGCTATCGAACGTGAACGTGCCACTCACGTGTTCCTGGCGCC"
    "GAACTGGCTGTACCAGCTGGCTGAACATCCGGCTCTGCCGCGTTCTGATC"
    "TGTCTTCTCTGCGTCGCGTTGTTTACGGCGGTGCACCGGCAGTACCATCT"
    "CGTGTAGCAGCAGCACGTGAACGTATGGGTGCTGTGCTGATGCAGAACTA"
    "CGGCACCCAGGAAGCAGCTTTCATCGCAGCACTGACTCCAGACGATCACG"
    "CACGTCGTGAACTGCTGACCGCTGTAGGTCGTCCTCTGCCACACGTTGAG"
    "GTGGAAATCCGTGATGACTCTGGTGGTACTCTGCCGCGTGGTGCGGTAGG"
    "TGAAGTCTGGGTACGTTCCCCGATGACTATGTCTGGTTACTGGCGTGACC"
    "CGGAACGTACGGCTCAGGTTCTGTCTGGTGGTTGGCTGCGTACTGGTGAT"
    "GTTGGTACCTTCGATGAGGATGGTCACCTGCATCTGACCGATCGTCTGCA"
    "GGACATCATCATCGTTGAAGCATATAACGTCTATTCCCGTCGTGTGGAACA"
    "TGTTCTGACCGAACACCCAGATGTTCGCGCAGCTGCGGTTGTTGGCGTAC"
    "CAGATCCGGACTCTGGTGAAGCTGTTTGCGCTGCGGTTGTAGTCGCGGAT"
    "GGTGCGGATCCTGACCCTGAACACCTGCGTGCTCTGGTTCGTGATCACCT"
    "GGGTGATCTGCACGTTCCTCGCCGTGTTGAGTTCGTTCGCTCCATCCCGG"
    "TAACTCCTGCCGGCAAACCAGATAAAGTGAAAGTGCGTACCTGGTTCACC"
    "GACTAA"
)

SHEET_TO_DRUG_SMILES: dict[str, str] = {
    "Fig3b": "O=C(NCCN1CCOCC1)c1ccc(Cl)cc1",
    "Fig3c": "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",
    "Fig3d": "CCCCOc1cc(C(=O)NCCN(CC)CC)c2ccccc2n1",
    "FigS18": "COc1cc(C(=O)NCc2ccc(OCCN(C)C)cc2)ccc1OC",
    "FigS19": "CCN(CC)CCNC(=O)c1ccc(N)cc1Cl",
    "FigS20": "COc1cc(C(=O)NCc2ccc(OCCN(C)C)cc2)cc(OC)c1OC",
    "FigS21": "CCN1CCC[C@H]1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC",
    "FigS22": "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "FigS23": "COc1cc(C(=O)N[C@H]2CN3CCC2CC3)cc(OC)c1OC",
}


def resolve_mcba_xlsx(input_xlsx: Path | None = None) -> Path:
    """Locate McbA supplementary Excel workbook."""
    return input_xlsx or utils.resolve_raw_or_docs(
        "mcba_amide_synthetase/41467_2024_55399_MOESM4_ESM.xlsx",
        raw_rel=f"{TASK_NAME}/41467_2024_55399_MOESM4_ESM.xlsx",
    )


def _parse_mcba_sheet(
    df: pd.DataFrame,
    wt_seq: str,
    smi: str,
) -> list[tuple[tuple[str, ...], float]]:
    """Extract (sequence, smiles, activity) records from a single McbA drug sheet."""
    site_tokens = [utils.clean_text(c) for c in df.iloc[0, 1:].tolist()]
    records: list[tuple[tuple[str, ...], float]] = []

    for row_idx in range(1, min(21, len(df))):
        mut_aa = utils.clean_text(df.iloc[row_idx, 0])
        if mut_aa not in utils.CANONICAL_AA:
            continue
        for col_idx, site_tok in enumerate(site_tokens, start=1):
            val = utils.parse_float(df.iloc[row_idx, col_idx])
            if val is None or not site_tok:
                continue
            seq = (
                wt_seq
                if mut_aa == site_tok[0]
                else utils.apply_mutations(wt_seq, f"{site_tok}{mut_aa}")
            )
            records.append(((seq, smi), max(0.0, val)))
    return records


def build_mcba_rows(input_xlsx: Path) -> list[dict[str, object]]:
    """Build benchmark rows for McbA across 9 pharmaceutical amide targets."""
    wt_seq = utils.translate_dna(WT_MCBA_DNA)
    excel_file = pd.ExcelFile(input_xlsx)
    records: list[tuple[tuple[str, ...], float]] = []

    for sheet_name, raw_smi in SHEET_TO_DRUG_SMILES.items():
        smi = utils.canonicalize_smiles(raw_smi)
        df = excel_file.parse(sheet_name, header=None)
        records.extend(_parse_mcba_sheet(df, wt_seq, smi))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_activity",
        raw_col="activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_mcba_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    input_xlsx: Path | None = None,
) -> Path:
    """Clean McbA amide synthetase dataset and write processed CSV."""
    xlsx_path = resolve_mcba_xlsx(input_xlsx)
    rows = build_mcba_rows(xlsx_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for McbA amide synthetase preprocessing."""
    out_path = clean_mcba_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
