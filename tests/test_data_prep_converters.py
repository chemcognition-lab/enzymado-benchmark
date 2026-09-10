from __future__ import annotations

import csv
import gzip
from pathlib import Path
import sys

import scripts.data_prep.convert_biolip_to_csv as biolip
import scripts.data_prep.convert_fireprotdb_to_csv as fireprotdb
import scripts.data_prep.generate_real_case_splits as real_case_splits

BIOLIP_HEADER = [
    "pdb_id",
    "chain_id",
    "resolution",
    "binding_site_id",
    "ligand_id",
    "ligand_chain_id",
    "ligand_serial_number",
    "binding_site_residues",
    "nearby_residues",
    "ligand_contact_residues",
    "catalytic_site_residues",
    "ec_number",
    "go_terms",
    "binding_affinity_1",
    "binding_affinity_2",
    "binding_affinity_3",
    "binding_affinity_4",
    "uniprot_id",
    "pubmed_id",
    "binding_site_range",
    "protein_sequence",
]


def run_main(monkeypatch, module, *arguments: str) -> None:
    """Run a data-preparation CLI with an isolated argument list."""
    monkeypatch.setattr(sys, "argv", [module.__file__, *arguments])
    module.main()


def read_csv(path: Path) -> list[list[str]]:
    """Read CSV rows without normalizing their values or order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_biolip_default_schema_and_gzip_rows_are_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    """Freeze BioLiP's default columns, row order, and gzip support."""
    input_path = tmp_path / "BioLiP_nr.txt.gz"
    output_path = tmp_path / "nested" / "BioLiP_nr.csv"
    first_row = [f"first_{index}" for index in range(len(BIOLIP_HEADER))]
    second_row = [f"second_{index}" for index in range(len(BIOLIP_HEADER))]
    with gzip.open(input_path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(first_row) + "\n\n")
        handle.write("\t".join(second_row) + "\n")

    run_main(
        monkeypatch,
        biolip,
        "--in",
        str(input_path),
        "--out",
        str(output_path),
    )

    assert read_csv(output_path) == [BIOLIP_HEADER, first_row, second_row]


def test_biolip_comment_header_and_quoted_csv_values_are_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    """Freeze custom header parsing and CSV quoting in the output artifact."""
    input_path = tmp_path / "biolip.csv"
    output_path = tmp_path / "biolip_out.csv"
    input_path.write_text(
        '#pdb_id,description\n1,"alpha,beta"\n2,gamma\n',
        encoding="utf-8",
    )

    run_main(
        monkeypatch,
        biolip,
        "--in",
        str(input_path),
        "--out",
        str(output_path),
    )

    assert output_path.read_bytes() == (
        b'pdb_id,description\r\n1,"alpha,beta"\r\n2,gamma\r\n'
    )


def test_fireprotdb_tabular_rows_and_blank_line_handling_are_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    """Freeze FireProtDB's delimiter conversion, values, and row order."""
    input_path = tmp_path / "fireprotdb.tsv"
    output_path = tmp_path / "nested" / "fireprotdb.csv"
    input_path.write_text(
        "experiment_id\tvalue\nA\t1\n\nB\t2\n",
        encoding="utf-8",
    )

    run_main(
        monkeypatch,
        fireprotdb,
        "--in",
        str(input_path),
        "--out",
        str(output_path),
    )

    assert output_path.read_bytes() == b"experiment_id,value\r\nA,1\r\nB,2\r\n"


def test_real_case_dataset_paths_and_targets_are_stable() -> None:
    """Freeze the eight real-case inputs and their stratification targets."""
    assert real_case_splits.REAL_CASE_DATASETS == (
        (
            Path(
                "data/processed/real_case/align_amylase_activity/"
                "alpha_amylase__specific_activity.csv"
            ),
            "specific_activity",
        ),
        (
            Path(
                "data/processed/real_case/align_amylase_expression/"
                "alpha_amylase__expression.csv"
            ),
            "expression",
        ),
        (
            Path(
                "data/processed/real_case/align_amylase_stability/"
                "alpha_amylase__thermostability.csv"
            ),
            "thermostability",
        ),
        (
            Path(
                "data/processed/real_case/align_glucosidase_activity/"
                "b_glucosidase__activity.csv"
            ),
            "activity",
        ),
        (
            Path(
                "data/processed/real_case/align_glucosidase_expression/"
                "b_glucosidase__expression.csv"
            ),
            "expression",
        ),
        (
            Path(
                "data/processed/real_case/align_glucosidase_melting/"
                "b_glucosidase__melting_point.csv"
            ),
            "melting_point",
        ),
        (
            Path("data/processed/real_case/cleeveland_galox/cleeveland.csv"),
            "specific_activity",
        ),
        (
            Path("data/processed/real_case/erickson_petase/erickson.csv"),
            "sum_aromatic_products",
        ),
    )


def test_real_case_metadata_paths_keep_canonical_then_legacy_order(
    tmp_path: Path, monkeypatch
) -> None:
    """Freeze metadata discovery order and canonical-only fallback behavior."""
    source_path = tmp_path / "example.csv"
    canonical_path = tmp_path / "example_random_80_10_10.metadata.json"
    legacy_path = tmp_path / "example_random_80_10_10_splits.metadata.json"
    canonical_path.touch()
    legacy_path.touch()
    calls = []

    def fake_build_metadata_path(path: Path, **kwargs) -> Path:
        calls.append((path, kwargs))
        return canonical_path

    monkeypatch.setattr(
        real_case_splits, "build_metadata_path", fake_build_metadata_path
    )

    assert real_case_splits._metadata_paths(source_path) == (
        canonical_path,
        legacy_path,
    )
    assert calls == [
        (
            source_path,
            {
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        )
    ]

    canonical_path.unlink()
    legacy_path.unlink()
    assert real_case_splits._metadata_paths(source_path) == (canonical_path,)


def test_real_case_driver_keeps_exact_split_contract(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Freeze all scientific and artifact arguments delegated by the driver."""
    source_path = Path("data/processed/real_case/example/example.csv")
    output_path = Path("example_random_80_10_10_standardized.csv")
    metadata_paths = (Path("canonical.json"), Path("legacy.json"))
    summary = {
        "train": {"rows": 8},
        "val": {"rows": 1},
        "test": {"rows": 1},
    }
    output_calls = []
    split_calls = []
    chdir_calls = []

    def fake_build_output_csv_path(path: Path, **kwargs) -> Path:
        output_calls.append((path, kwargs))
        return output_path

    def fake_split_dataset_csv(*args, **kwargs):
        split_calls.append((args, kwargs))
        return summary

    monkeypatch.setattr(real_case_splits, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        real_case_splits,
        "REAL_CASE_DATASETS",
        ((source_path, "activity"),),
    )
    monkeypatch.setattr(
        real_case_splits, "build_output_csv_path", fake_build_output_csv_path
    )
    monkeypatch.setattr(
        real_case_splits,
        "_metadata_paths",
        lambda path: metadata_paths,
    )
    monkeypatch.setattr(real_case_splits, "split_dataset_csv", fake_split_dataset_csv)
    monkeypatch.setattr(
        real_case_splits.os, "chdir", lambda path: chdir_calls.append(path)
    )

    real_case_splits.main()

    assert chdir_calls == [tmp_path]
    assert output_calls == [
        (
            source_path,
            {
                "mode": "random",
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        )
    ]
    expected_kwargs = {
        "mode": "random",
        "train_fraction": 0.8,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "sequence_col": "protein_sequence",
        "random_split_unit": "point",
        "point_id_col": "dataset_id",
        "stratify_col": "activity",
        "stratify_bins": 10,
        "seed": 42,
        "standardization_cols": ("activity",),
    }
    assert split_calls == [
        ((source_path, output_path), {**expected_kwargs, "metadata_path": path})
        for path in metadata_paths
    ]
    assert capsys.readouterr().out == (
        f"{source_path} -> {output_path}\n"
        "{\n"
        '  "train": {\n'
        '    "rows": 8\n'
        "  },\n"
        '  "val": {\n'
        '    "rows": 1\n'
        "  },\n"
        '  "test": {\n'
        '    "rows": 1\n'
        "  }\n"
        "}\n"
    )
