"""Preprocess Koudelakova haloalkane dehalogenase (HLD) specificity panel (Layout B)."""

from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.data_prep.real_cases import utils

TASK_NAME = "koudelakova_hld_panel"
FIELDNAMES = [
    "dataset_id",
    "protein_sequence",
    "smiles",
    "transformed_activity",
    "log10_activity",
]
LOG10_OFFSET = 1e-4

# 9 wild-type HLD sequences from Table S1 GenBank/UniProt/PDB accessions
WT_HLD_SEQUENCES: dict[str, str] = {
    "DatA": (
        "MKEHRHMTEKSPHSAFGDGAKAYDVPAFGLQIHTVEHGSGAPIVFLHGNPTSSYLWRHIFRRL"
        "HGHGRLLAVDLIGYGQSSKPDIEYTLENQQRYVDAWFDALDLRNVTLVLQDYGAAFGLNWASR"
        "NPDRVRAVAFFEPVLRNIDSVDLSPEFVTRRAKLRQPGEGEIFVQQENRFLTELFPWFFLTPL"
        "APEDLRQYQTPFPTPHSRKAILAGPRNLPVDGEPASTVAFLEQAVNWLNTSDTPKLLLTFKPG"
        "FLLTDAILKWSQVTIRNLEIEAAGAGIHFVQEEQPETIARLLDAWLTRIAGN"
    ),
    "DbeA": (
        "MTISADISLHHRAVLGSTMAYRETGRSDAPHVLFLHGNPTSSYIWRNIMPLVAPVGHCIAPDL"
        "IGYGQSGKPDISYRFFDQADYLDALIDELGIASAYLVAQDWGTALAFHLAARRPQLVRGLAFM"
        "EFIRPMRDWSDFHQHDAARETFRKFRTPGVGEAMILDNNAFVERVLPGSILRTLSEEEMAAYR"
        "APFATRESRMPTLMLPRELPIAGEPADVTQALTAAHAALAASTYPKLLFVGSPGALVSPAFAA"
        "EFAKTLKHCAVIQLGAGGHYLQEDHPEAIGRSVAGWIAGIEAASAQRHAACRAKRART"
    ),
    "DbjA": (
        "MSKPIEIEIRRAPVLGSSMAYRETGAQDAPVVLFLHGNPTSSHIWRNILPLVSPVAHCIAPDL"
        "IGFGQSGKPDIAYRFFDHVRYLDAFIEQRGVTSAYLVAQDWGTALAFHLAARRPDFVRGLAFM"
        "EFIRPMPTWQDFHHTEVAEEQDHAEAARAVFRKFRTPGEGEAMILEANAFVERVLPGGIVRKL"
        "GDEEMAPYRTPFPTPESRRPVLAFPRELPIAGEPADVYEALQSAHAALAASSYPKLLFTGEPG"
        "ALVSPEFAERFAASLTRCALIRLGAGLHYLQEDHADAIGRSVAGWIAGIEAVRPQLAA"
    ),
    "DhaA": (
        "MSEIGTGFPFDPHYVEVLGERMHYVDVGPRDGTPVLFLHGNPTSSYLWRNIIPHVAPSHRCIA"
        "PDLIGMGKSDKPDLDYFFDDHVRYLDAFIEALGLEEVVLVIHDWGSALGFHWAKRNPERVKGI"
        "ACMEFIRPIPTWDEWPEFARETFQAFRTADVGRELIIDQNAFIEGALPKCVVRPLTEVEMDHY"
        "REPFLKPVDREPLWRFPNELPIAGEPANIVALVEAYMNWLHQSPVPKLLFWGTPGVLIPPAEA"
        "ARLAESLPNCKTVDIGPGLHYLQEDNPDLIGSEIARWLPAL"
    ),
    "DhlA": (
        "MINAIRTPDQRFSNLDQYPFSPNYLDDLPGYPGLRAHYLDEGNSDAEDVFLCLHGEPTWSYLY"
        "RKMIPVFAESGARVIAPDFFGFGKSDKPVDEEDYTFEFHRNFLLALIERLDLRNITLVVQDWG"
        "GFLGLTLPMADPSRFKRLIIMNACLMTDPVTQPAFSAFVTQPADGFTAWKYDLVTPSDLRLDQ"
        "FMKRWAPTLTEAEASAYAAPFPDTSYQAGVRKFPKMVAQRDQACIDISTEAISFWQNDWNGQT"
        "FMAIGMKDKLLGPDVMYPMKALINGCPEPLEIADAGHFVQEFGEQVAREALKHFAETE"
    ),
    "DmbA": (
        "MTAFGVEPYGQPKYLEIAGKRMAYIDEGKGDAIVFQHGNPTSSYLWRNIMPHLEGLGRLVACD"
        "LIGMGASDKLSPSGPDRYSYGEQRDFLFALWDALDLGDHVVLVLHDWGSALGFDWANQHRDRV"
        "QGIAFMEAIVTPMTWADWPPAVRGVFQGFRSPQGEPMALEHNIFVERVLPGAILRQLSDEEMN"
        "HYRRPFVNGGEDRRPTLSWPRNLPIDGEPAEVVALVNEYRSWLEETDMPKLFINAEPGAIITG"
        "RIRDYVRSWPNQTEITVPGVHFVQEDSPEEIGAAIAQFVRRLRSAAGV"
    ),
    "DmbC": (
        "MSIDFTPDPQLYPFESRWFDSSRGRIHYVDEGTGPPILLCHGNPTWSFLYRDIIVALRDRFRC"
        "VAPDYLGFGLSERPSGFGYQIDEHARVIGEFVDHLGLDRYLSMGQDWGGPISMAVAVERADRV"
        "RGVVLGNTWFWPADTLAMKAFSRVMSSPPVQYAILRRNFFVERLIPAGTEHRPSSAVMAHYRA"
        "VQPNAAARRGVAEMPKQILAARPLLARLAREVPATLGTKPTLLIWGMKDVAFRPKTIIPRLSA"
        "TFPDHVLVELPNAKHFIQEDAPDRIAAAIIERFG"
    ),
    "DrbA": (
        "MSCRLSSNRRGSSKLAAMTNLASDLFPHPSSELSIDGHTLRYIDTAASSDIPSSAVGSSDGEP"
        "TFLCVHGNPTWSFYYRRIIERYGKQQRVIAVDHIGCGRSDKPSEDEFPYTMAAHRDNLIRLVD"
        "ELDLKNVILIAHDWGGAIGLSAMHARRDRLAGIGLLNTAAFPPPYMPQRIAACRMPVLGTPAV"
        "RGLNLFARAAVTMAMSRTKMKPDVAAGLLAPYDNWKNRVAIDRFVRDIPLNDSHPTMKTLRQL"
        "ESDLPDLASLPISLIWGMKDWCFRPECLRRFQSVWPDAEVTELATTGHYVIEDSPEETLAAID"
        "SLLARVKERIGAA"
    ),
    "LinB": (
        "MSLGAKPFGEKKFIEIKGRRMAYIDEGTGDPILFQHGNPTSSYLWRNIMPHCAGLGRLIACDL"
        "IGMGDSDKLDPSGPERYAYAEHRDYLDALWEALDLGDRVVLVVHDWGSALGFDWARRHRERVQ"
        "GIAYMEAIAMPIEWADFPEQDRDLFQAFRSQAGEELVLQDNVFVEQVLPGLILRPLSEAEMAA"
        "YREPFLAAGEARRPTLSWPRQIPIAGTPADVVAIARDYAGWLSESPIPKLFINAEPGALTTGR"
        "MRDFCRTWPNQTEITVAGAHFIQEDSPDEIGAAIAAFVRRLRPA"
    ),
}

ENZYME_ORDER: tuple[str, ...] = (
    "DatA",
    "DbeA",
    "DbjA",
    "DhaA",
    "DhlA",
    "DmbA",
    "DmbC",
    "DrbA",
    "LinB",
    "DbeA1",
    "DbeA2",
    "DbjA_delta",
    "DhaA31",
)

# 30 halogenated substrates in the exact order of Table S3 & Table S5
SUBSTRATE_ORDER_SMILES: tuple[tuple[str, str], ...] = (
    ("1-Chlorobutane", "CCCCCl"),
    ("1-Chlorohexane", "CCCCCCCl"),
    ("1-Bromobutane", "CCCCBr"),
    ("1-Bromohexane", "CCCCCCBr"),
    ("1-Iodopropane", "CCCI"),
    ("1-Iodobutane", "CCCCI"),
    ("1-Iodohexane", "CCCCCCI"),
    ("1,2-Dichloroethane", "ClCCCl"),
    ("1,3-Dichloropropane", "ClCCCCl"),
    ("1,5-Dichloropentane", "ClCCCCCCl"),
    ("1,2-Dibromoethane", "BrCCBr"),
    ("1,3-Dibromopropane", "BrCCCBr"),
    ("1-Bromo-3-chloropropane", "ClCCCBr"),
    ("1,3-Diiodopropane", "ICCCI"),
    ("2-Iodobutane", "CCC(C)I"),
    ("1,2-Dichloropropane", "CC(Cl)CCl"),
    ("1,2-Dibromopropane", "CC(Br)CBr"),
    ("2-Bromo-1-chloropropane", "CC(Br)CCl"),
    ("1,2,3-Trichloropropane", "ClCC(Cl)CCl"),
    ("1-Chloro-2-(2-chloroethoxy)ethane", "ClCCOCCCl"),
    ("Chlorocyclohexane", "ClC1CCCCC1"),
    ("Bromocyclohexane", "BrC1CCCCC1"),
    ("(Bromomethyl)cyclohexane", "BrCC1CCCCC1"),
    ("1-Bromo-2-chloroethane", "ClCCBr"),
    ("Chlorocyclopentane", "ClC1CCCC1"),
    ("4-Bromobutanenitrile", "N#CCCCBr"),
    ("1,2,3-Tribromopropane", "BrCC(Br)CBr"),
    ("1,2-Dibromo-3-chloropropane", "ClCC(Br)CBr"),
    ("3-Chloro-2-methylprop-1-ene", "C=C(C)CCl"),
    ("2,3-Dichloroprop-1-ene", "C=C(Cl)CCl"),
)


def resolve_hld_pdf(pdf_path: Path | None = None) -> Path:
    """Locate Koudelakova 2011 supplementary PDF."""
    if pdf_path is not None:
        return pdf_path
    raw_pdf = utils.RAW_REAL_CASE_DIR / TASK_NAME / "hld_koudelakova2011_biochemj.pdf"
    if raw_pdf.exists():
        return raw_pdf
    return utils.PROJECT_ROOT / "docs" / "pdfs" / "hld_koudelakova2011_biochemj.pdf"


def build_hld_sequences() -> dict[str, str]:
    """Build all 13 HLD sequences (9 wild-type + 4 engineered variants) per Table S2."""
    seqs = dict(WT_HLD_SEQUENCES)
    dbea = seqs["DbeA"]
    dbja = seqs["DbjA"]
    dhaa = seqs["DhaA"]
    # Table S2: DbeA1 insertion between residues 142 and 143: VAEEQDHAE
    seqs["DbeA1"] = dbea[:142] + "VAEEQDHAE" + dbea[142:]
    # Table S2: DbeA2 insertion between residues 141 and 142: EVAEEQDHA
    seqs["DbeA2"] = dbea[:141] + "EVAEEQDHA" + dbea[141:]
    # Table S2: DbjA_delta deletion of H140..E146 (HTEVAEE)
    seqs["DbjA_delta"] = dbja[:139] + dbja[146:]
    # Table S2: DhaA31 substitutions I135F, C176Y, V245F, L246I, Y273F
    seqs["DhaA31"] = utils.apply_mutations(dhaa, "I135F/C176Y/V245F/L246I/Y273F")
    return seqs


def _parse_table_s5_from_pdf(pdf_path: Path) -> list[float]:
    """Extract the 390 numeric cells (13 enzymes x 30 substrates) from Table S5."""
    text = subprocess.check_output(
        ["pdftotext", "-f", "17", "-l", "17", str(pdf_path), "-"],
    ).decode("utf-8", errors="ignore")

    values: list[float] = []
    after_last_substrate = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "2,3-Dichloroprop-1-ene":
            after_last_substrate = True
            continue
        if after_last_substrate and "." in line:
            val = utils.parse_float(line)
            if val is not None:
                values.append(val)
    if len(values) != len(ENZYME_ORDER) * len(SUBSTRATE_ORDER_SMILES):
        raise ValueError(f"Expected 390 Table S5 values, parsed {len(values)}")
    return values


def build_hld_rows(pdf_path: Path) -> list[dict[str, object]]:
    """Build benchmark rows for the 13 x 30 Koudelakova HLD specificity matrix."""
    seqs = build_hld_sequences()
    canon_substrates = [
        utils.canonicalize_smiles(smi) for _, smi in SUBSTRATE_ORDER_SMILES
    ]
    values = _parse_table_s5_from_pdf(pdf_path)

    n_subs = len(canon_substrates)
    records: list[tuple[tuple[str, ...], float]] = []
    for enz_idx, enz_name in enumerate(ENZYME_ORDER):
        seq = seqs[enz_name]
        col_vals = values[enz_idx * n_subs : (enz_idx + 1) * n_subs]
        for smi, val in zip(canon_substrates, col_vals, strict=True):
            records.append(((seq, smi), val))

    return utils.build_layout_b_rows(
        TASK_NAME,
        records,
        target_col="log10_activity",
        raw_col="transformed_activity",
        log10_offset=LOG10_OFFSET,
    )


def clean_hld_data(
    output_root: Path = utils.PROCESSED_REAL_CASE_DIR,
    pdf_path: Path | None = None,
) -> Path:
    """Clean Koudelakova HLD dataset and write processed CSV."""
    resolved_pdf = resolve_hld_pdf(pdf_path)
    rows = build_hld_rows(resolved_pdf)
    return utils.write_processed_dataset(
        TASK_NAME,
        FIELDNAMES,
        rows,
        output_root=output_root,
    )


def main() -> None:
    """CLI entrypoint for Koudelakova HLD preprocessing."""
    out_path = clean_hld_data(output_root=utils.parse_output_root(__doc__))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
