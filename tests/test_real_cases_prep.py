"""Unit tests for real-case dataset preprocessing utilities and scripts."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.data_prep.real_cases import (
    ata_3fcr,
    koudelakova_hld,
    mcba_amide_synthetase,
    utils,
    welo5_soraphen,
)


def test_clean_and_validate_protein_sequence() -> None:
    """Verify gap stripping and canonical amino-acid validation."""
    base = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    gapped = f" {base[:20]}--..*{base[20:]}\n"
    cleaned = utils.clean_protein_sequence(gapped)
    assert cleaned == base
    assert utils.is_valid_sequence(cleaned)
    assert not utils.is_valid_sequence("ACDEFG", min_len=30)
    assert not utils.is_valid_sequence(f"{base}X")


def test_apply_mutations_standard_and_colon() -> None:
    """Verify 1-based point mutation application in standard and colon formats."""
    wt = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    mutated = utils.apply_mutations(wt, "A1G/C2S")
    assert mutated.startswith("GSDEF")
    assert utils.apply_mutations(wt, "WT") == wt
    assert utils.apply_mutations(wt, "A:1:G,A:2:S", verify_wt=False) == mutated

    with pytest.raises(ValueError, match="Wild-type mismatch"):
        utils.apply_mutations(wt, "V1G")


def test_parse_hgvs_missense() -> None:
    """Verify HGVS protein missense parsing and synonymous/nonsense filtering."""
    wt = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    res = utils.parse_hgvs_missense("p.Ala1Gly", wt)
    assert res is not None
    assert res.startswith("GCDEF")
    assert utils.parse_hgvs_missense("p.Ala1=", wt) is None
    assert utils.parse_hgvs_missense("p.Ala1Ter", wt) is None
    assert utils.parse_hgvs_missense("WT", wt) == wt


def test_canonicalize_smiles_and_alkoxide() -> None:
    """Verify RDKit SMILES canonicalization and alkoxide neutralization."""
    smi = utils.canonicalize_smiles(
        "C[C@H]([O-])CC[C@H](C)O",
        neutralize_alkoxide=True,
    )
    assert "[O-]" not in smi
    multi = utils.canonicalize_multi_smiles(["CCO", "CC(=O)O"])
    assert "." in multi


def test_translate_dna_and_mcba_wt() -> None:
    """Verify DNA translation produces the 518-aa wt-McbA sequence."""
    seq = utils.translate_dna(mcba_amide_synthetase.WT_MCBA_DNA)
    assert len(seq) == 518
    assert utils.is_valid_sequence(seq)
    assert seq[96] == "Y"
    assert seq[507] == "D"


def test_ata_and_hld_sequence_builders() -> None:
    """Verify Ao 3FCR/3HMU/6SNU and Koudelakova HLD sequence builders."""
    s_3fcr = ata_3fcr._resolve_ata_variant("3FCR-Y87F/T231A/Y59W")
    s_3hmu = ata_3fcr._resolve_ata_variant("3HMU-F91L/W63Y/F25W")
    s_6snu = ata_3fcr._resolve_ata_variant("6SNU-WT")
    assert utils.is_valid_sequence(s_3fcr)
    assert utils.is_valid_sequence(s_3hmu)
    assert utils.is_valid_sequence(s_6snu)

    hld_seqs = koudelakova_hld.build_hld_sequences()
    assert len(hld_seqs) == 13
    for seq in hld_seqs.values():
        assert utils.is_valid_sequence(seq)


def test_layout_builders_and_validation(tmp_path: Path) -> None:
    """Verify replicate averaging, Layout A/B row builders, and validation."""
    assert utils.parse_float("1.25") == 1.25
    assert utils.parse_float("ND") is None
    assert utils.compute_log10(0.0, offset=1e-3) == pytest.approx(-3.0)

    rows_a = utils.build_layout_a_rows(
        "task_a",
        [(("SEQ1",), 1.0), (("SEQ1",), 3.0)],
        target_col="log10_act",
        raw_col="act",
        log10_offset=0.0,
    )
    assert rows_a[0]["act"] == "2.000000"

    rows_b = utils.build_layout_b_rows(
        "task_b",
        [(("SEQ1", "CCO"), 0.0)],
        target_col="log10_act",
        raw_col="act",
        log10_offset=1e-3,
    )
    assert rows_b[0]["log10_act"] == "-3.000000"

    resolved = utils.resolve_raw_or_docs("welo5_soraphen_activity/welo5_variants.csv")
    assert resolved.exists()

    out_path = welo5_soraphen.clean_welo5_data(output_root=tmp_path)
    assert out_path.exists()

    with pytest.raises(ValueError, match="outside bounds"):
        utils.validate_dataset_rows([], task_name="empty_task")

    base_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    written = utils.write_processed_dataset(
        "tiny_valid",
        ["dataset_id", "protein_sequence", "val"],
        [
            {
                "dataset_id": "tiny_valid_000001",
                "protein_sequence": f"{base_seq}A",
                "val": "1.0",
            },
            {
                "dataset_id": "tiny_valid_000002",
                "protein_sequence": f"{base_seq}C",
                "val": "2.0",
            },
        ],
        output_root=tmp_path,
        min_rows=2,
        min_unique_seqs=2,
    )
    assert written.exists()
