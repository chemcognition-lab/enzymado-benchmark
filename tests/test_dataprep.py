import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from scripts.data_prep.dataprep import (
    add_negatives,
    has_explicit_ec_number,
    parse_ph,
    prep_fireprot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_has_explicit_ec_number_accepts_supported_biolip_formats():
    values = pd.Series(
        [
            "2.1.1.37",
            "3.1.-.-",
            "4.2.2.n1",
            "?,3.1.3.16",
            "'3.6.1.11,3.6.1.40'",
        ]
    )

    assert has_explicit_ec_number(values).tolist() == [True] * len(values)


def test_has_explicit_ec_number_rejects_unknown_or_malformed_values():
    values = pd.Series(
        [
            "?",
            "",
            None,
            pd.NA,
            "not assigned",
            "GO:0003824",
            "1.2.3",
            "12.3.4.5",
            "A1.2.3.4B",
        ]
    )

    assert has_explicit_ec_number(values).tolist() == [False] * len(values)


def test_biolip_negatives_follow_frozen_assignment_order():
    """Generate the same negative pairs independently of Python hash ordering."""
    positives = pd.DataFrame(
        [
            {"protein_sequence": "AAAA", "smiles": "C", "uniprot_id": "P1"},
            {"protein_sequence": "BBBB", "smiles": "O", "uniprot_id": "P2"},
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "protein_sequence_sha256": hashlib.sha256(b"BBBB").hexdigest(),
                "smiles": "C",
            },
            {
                "protein_sequence_sha256": hashlib.sha256(b"AAAA").hexdigest(),
                "smiles": "O",
            },
        ]
    )

    output = add_negatives(positives, assignments)

    assert output[["protein_sequence", "smiles", "label"]].to_dict(
        orient="records"
    ) == [
        {"protein_sequence": "AAAA", "smiles": "C", "label": 1},
        {"protein_sequence": "BBBB", "smiles": "O", "label": 1},
        {"protein_sequence": "BBBB", "smiles": "C", "label": 0},
        {"protein_sequence": "AAAA", "smiles": "O", "label": 0},
    ]
    assert output.loc[2:, "uniprot_id"].isna().all()


def test_biolip_reproduction_manifests_are_frozen():
    """Lock the recovered assignments and historical ligand representations."""
    assignment_path = (
        PROJECT_ROOT / "data" / "interim" / "biolip" / "biolip_negative_assignments.csv"
    )
    mapping_path = (
        PROJECT_ROOT / "data" / "interim" / "biolip" / "biolip_smiles_mapping.csv"
    )
    assignments = pd.read_csv(assignment_path)
    mapping = pd.read_csv(mapping_path)

    assert len(assignments) == 14_279
    assert not assignments.duplicated().any()
    assert len(mapping) == 3_501
    assert not mapping["raw_smiles"].duplicated().any()
    assert hashlib.sha256(assignment_path.read_bytes()).hexdigest() == (
        "19a1bb08e401e70db67f2624869042c164e4bbc159c65339750e1968a0a65983"
    )
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == (
        "04f24c6b19c6443ee606b15ec9f55924252679676cc64f177d440b088692d04a"
    )


def test_parse_ph_preserves_supported_values_and_invalid_fallbacks():
    """Freeze current single-value, range, missing, and invalid parsing."""
    assert parse_ph("5.5") == 5.5
    assert parse_ph("6-8") == 7.0
    assert parse_ph(" 6 - 8 ") == 7.0
    assert parse_ph(None) is None
    assert parse_ph(pd.NA) is None
    assert parse_ph("not reported") is None


def test_prep_fireprot_preserves_rows_files_and_removal_statistics(tmp_path):
    """Freeze FireProtDB filtering, output schemas, row order, and statistics."""
    raw_dir = tmp_path / "raw" / "fireprotdb"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset_id": "A", "DDG": "1.5", "DTM": "10"},
            {"dataset_id": "B", "DDG": "bad", "DTM": "20"},
            {"dataset_id": "C", "DDG": "", "DTM": "bad"},
        ]
    ).to_csv(raw_dir / "fireprotdb_ec.tsv", sep="\t", index=False)
    paths = SimpleNamespace(
        data_raw_fireprotdb_dir=raw_dir,
        data_interim_dir=interim_dir,
    )

    stats = prep_fireprot(paths)

    assert (interim_dir / "fireprotdb/fireprotdb_ddg.csv").read_bytes() == (
        b"dataset_id,DDG,DTM\nA,1.5,10.0\n"
    )
    assert (interim_dir / "fireprotdb/fireprotdb_dtm.csv").read_bytes() == (
        b"dataset_id,DDG,DTM\nA,1.5,10.0\nB,,20.0\n"
    )
    assert stats == {
        "amount_of_removed_data": {
            "datapoints_before_cleaning": 6,
            "removed_by_reason": {
                "missing_values": 3,
                "duplicates": 0,
                "invalid_sequence": 0,
                "was_not_enzyme": 0,
                "other": 0,
            },
            "datapoints_after_cleaning": 3,
            "datapoints_removed_total": 3,
            "removed_percentage": 50.0,
        }
    }
