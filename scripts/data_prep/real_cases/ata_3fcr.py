"""Preprocess Ao 3FCR/3HMU/6SNU amine transaminase chiral amine panel (Layout B)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scripts.data_prep.real_cases import utils

TASK_NAME = "ao_3fcr_transaminase"
FIELDNAMES = [
    "dataset_id",
    "protein_sequence",
    "smiles",
    "specific_activity",
    "log10_specific_activity",
]
LOG10_OFFSET = 1e-3

# Wild-type sequences from Supplementary Information Section 7 (pages S23-S25)
WT_SEQUENCES: dict[str, tuple[str, int]] = {
    "3FCR": (
        "MLKNDQLDQWDRDNFFHPSTHLAQHARGESANRVIKTASGVFIEDRDGTKLLDAFAGLYCV"
        "NVGYGRQEIAEAIADQARELAYYHSYVGHGTEASITLAKMILDRAPKNMSKVYFGLGGSDAN"
        "ETNVKLIWYYNNILGRPEKKKIISRWRGYHGSGLVTGSLTGLELFHKKFDLPVEQVIHTEAPY"
        "YFRREDLNQTEEQFVAHCVAELEALIEREGADTIAAFIGEPILGTGGIVPPPAGYWEAIQTVLN"
        "KHDILLVADEVVTGFGRLGTMFGSDHYGLEPDIITIAKGLTSAYAPLSGSIVSDKVWKVLEQG"
        "TDENGPIGHGWTYSAHPIGAAAGVANLKLLDELNLVSNAGEVGAYLNATMAEALSQHANVG"
        "DVRGEGLLCAVEFVKDRDSRTFFDAADKIGPQISAKLLEQDKIIARAMPQGDILGFAPPFCLT"
        "RAEADQVVEGTLRAVKAVLGSLIGSDGGSGGGSTSRDHMVLHEYVNAAGITLIGSDGGSGG"
        "GSTSHHHHHH",
        0,
    ),
    "3HMU": (
        "MSLATITNHMPTAELQALDAAHHLHPFSANNALGEEGTRVITRARGVWLNDSEGEEILDAMA"
        "GLWCVNIGYGRDELAEVAARQMRELPYYNTFFKTTHVPAIALAQKLAELAPGDLNHVFFAG"
        "GGSEANDTNIRMVRTYWQNKGQPEKTVIISRKNAYHGSTVASSALGGMAGMHAQSGLIPD"
        "VHHINQPNWWAEGGDMDPEEFGLARARELEEAILELGENRVAAFIAEPVQGAGGVIVAPDS"
        "YWPEIQRICDKYDILLIADEVICGFGRTGNWFGTQTMGIRPHIMTIAKGLSSGYAPIGGSIVCD"
        "EVAHVIGKDEFNHGYTYSGHPVAAAVALENLRILEEENILDHVRNVAAPYLKEKWEALTDHP"
        "LVGEAKIVGMMASIALTPNKASRAKFASEPGTIGYICRERCFANNLIMRHVGDRMIISPPLVIT"
        "PAEIDEMFVRIRKSLDEAQAEIEKQGLMKSEGSHHHHHH",
        2,
    ),
    "6SNU": (
        "MGSSHHHHHHSSGLVPRGSHMASQKQRTTSQWRELDAAHHLHPFTDTASLNQAGARVMT"
        "RGEGVYLWDSEGNKIIDGMAGLWCVNVGYGRKDFAEAARRQMEELPFYNTFFKTTHPAVV"
        "ELSSLLAEVTPAGFDRVFYTNSGSESVDTMIRMVRRYWDVQGKPEKKTLIGRWNGYHGSTI"
        "GGASLGGMKYMHEQGDLPIPGMAHIEQPWWYKHGKDMTPDEFGVVAARWLEEKILEIGAD"
        "KVAAFVGEPIQGAGGVIVPPATYWPEIERICRKYDVLLVADEVICGFGRTGEWFGHQHFGFQ"
        "PDLFTAAKGLSSGYLPIGAVFVGKRVAEGLIAGGDFNHGFTYSGHPVCAAVAHANVAALRD"
        "EGIVQRVKDDIGPYMQKRWRETFSRFEHVDDVRGVGMVQAFTLVKNKAKRELFPDFGEIGT"
        "LCRDIFFRNNLIMRACGDHIVSAPPLVMTRAEVDEMLAVAERCLEEFEQTLKARGLA",
        0,
    ),
}

# 30 enantiopure chiral amines (R)-1..(S)-15 from Figure 3 (Angew. Chem. 2023)
SUBSTRATE_SMILES: dict[str, str] = {
    "(R)-1": "C[C@@H](N)c1ccccc1",
    "(S)-1": "C[C@H](N)c1ccccc1",
    "(R)-2": "CC[C@@H](N)c1ccccc1",
    "(S)-2": "CC[C@H](N)c1ccccc1",
    "(R)-3": "OC[C@H](N)c1ccccc1",
    "(S)-3": "OC[C@@H](N)c1ccccc1",
    "(R)-4": "CCC[C@@H](N)c1ccccc1",
    "(S)-4": "CCC[C@H](N)c1ccccc1",
    "(R)-5": "C[C@@H](N)c1ccc(Cl)cc1",
    "(S)-5": "C[C@H](N)c1ccc(Cl)cc1",
    "(R)-6": "C[C@@H](N)c1cccc(Cl)c1",
    "(S)-6": "C[C@H](N)c1cccc(Cl)c1",
    "(R)-7": "C[C@@H](N)c1ccccc1Cl",
    "(S)-7": "C[C@H](N)c1ccccc1Cl",
    "(R)-8": "COc1ccc([C@@H](C)N)cc1",
    "(S)-8": "COc1ccc([C@H](C)N)cc1",
    "(R)-9": "C[C@@H](N)c1cccc2ccccc12",
    "(S)-9": "C[C@H](N)c1cccc2ccccc12",
    "(R)-10": "C[C@@H](N)c1ccc2ccccc2c1",
    "(S)-10": "C[C@H](N)c1ccc2ccccc2c1",
    "(R)-11": "CCCC[C@@H](N)c1ccccc1",
    "(S)-11": "CCCC[C@H](N)c1ccccc1",
    "(R)-12": "N[C@@H](Cc1ccccc1)c1ccccc1",
    "(S)-12": "N[C@H](Cc1ccccc1)c1ccccc1",
    "(R)-13": "N[C@@H](C(=O)O)c1ccccc1",
    "(S)-13": "N[C@H](C(=O)O)c1ccccc1",
    "(R)-14": "N[C@@H](CC(=O)O)c1ccccc1",
    "(S)-14": "N[C@H](CC(=O)O)c1ccccc1",
    "(R)-15": "Cc1ccc([C@@H](C)N)cc1",
    "(S)-15": "Cc1ccc([C@H](C)N)cc1",
}


def resolve_ata_xlsx(input_xlsx: Path | None = None) -> Path:
    """Locate Ao 3FCR supplementary Excel file."""
    return input_xlsx or utils.resolve_raw_or_docs(
        "supplementary_tables/ata_3fcr_ao2023_SI_excel.xlsx",
        raw_rel=f"{TASK_NAME}/ata_3fcr_ao2023_SI_excel.xlsx",
    )


def _resolve_ata_variant(variant_label: str) -> str:
    """Resolve a 3FCR/3HMU/6SNU variant label into a full protein sequence."""
    scaffold, muts = variant_label.split("-", maxsplit=1)
    wt_seq, offset = WT_SEQUENCES[scaffold]
    return utils.apply_mutations(wt_seq, muts, offset=offset)


def build_ata_rows(input_xlsx: Path) -> list[dict[str, object]]:
    """Build benchmark rows for Ao 3FCR/3HMU/6SNU amine transaminase panel."""
    df = pd.read_excel(input_xlsx, sheet_name="detailed data", header=None)
    sub_labels = [utils.clean_text(df.iloc[2, c]) for c in range(9, 39)]
    canon_smiles = {
        label: utils.canonicalize_smiles(smi) for label, smi in SUBSTRATE_SMILES.items()
    }

    records: list[tuple[tuple[str, ...], float]] = []
    # Rows 3..81 contain the 79 enzyme variants in the specific activity sub-table
    for row_idx in range(3, 82):
        var_label = utils.clean_text(df.iloc[row_idx, 1])
        if not var_label or "-" not in var_label:
            continue
        seq = _resolve_ata_variant(var_label)
        for col_offset, sub_label in enumerate(sub_labels):
            smi = canon_smiles.get(sub_label)
            val = utils.parse_float(df.iloc[row_idx, 9 + col_offset])
            if smi is not None and val is not None and val >= 0.0:
                records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_specific_activity",
        raw_col="specific_activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_ata_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    input_xlsx: Path | None = None,
) -> Path:
    """Clean Ao 3FCR transaminase dataset and write processed CSV."""
    xlsx_path = resolve_ata_xlsx(input_xlsx)
    rows = build_ata_rows(xlsx_path)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Ao 3FCR transaminase preprocessing."""
    out_path = clean_ata_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
