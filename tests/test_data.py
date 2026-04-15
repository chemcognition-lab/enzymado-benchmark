from src.dataset_splitting import (
    build_mmseqs_cluster_command,
    build_output_csv_path,
    build_split_assignment_rows,
    generate_random_split_rows,
    generate_similarity_split_from_clusters,
    validate_split_fractions,
)
from pathlib import Path


def build_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = ["dataset_id", "uniprot_id", "protein_sequence", "target"]
    rows = [
        {
            "dataset_id": "row_1",
            "uniprot_id": "P1",
            "protein_sequence": "AAAA",
            "target": "1.0",
        },
        {
            "dataset_id": "row_2",
            "uniprot_id": "P1",
            "protein_sequence": "AAAA",
            "target": "2.0",
        },
        {
            "dataset_id": "row_3",
            "uniprot_id": "P2",
            "protein_sequence": "CCCC",
            "target": "3.0",
        },
        {
            "dataset_id": "row_4",
            "uniprot_id": "P3",
            "protein_sequence": "DDDD",
            "target": "4.0",
        },
        {
            "dataset_id": "row_5",
            "uniprot_id": "P4",
            "protein_sequence": "EEEE",
            "target": "5.0",
        },
        {
            "dataset_id": "row_6",
            "uniprot_id": "P5",
            "protein_sequence": "FFFF",
            "target": "6.0",
        },
    ]
    return fieldnames, rows


def test_validate_split_fractions_rejects_invalid_total():
    try:
        validate_split_fractions(0.6, 0.2, 0.3)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("validate_split_fractions should fail for invalid totals.")


def test_random_split_keeps_grouped_rows_together():
    fieldnames, rows = build_rows()

    _, split_rows, summary = generate_random_split_rows(
        rows,
        fieldnames,
        0.6,
        0.2,
        0.2,
        sequence_col="protein_sequence",
        group_col="uniprot_id",
        seed=7,
    )

    split_by_protein = {
        row["uniprot_id"]: row["split"]
        for row in split_rows
    }
    assert split_by_protein["P1"] == split_rows[1]["split"]
    assert set(summary) == {"train", "val", "test"}
    assert sum(split["rows"] for split in summary.values()) == len(rows)


def test_similarity_split_assigns_whole_clusters_to_one_split():
    fieldnames, rows = build_rows()
    group_clusters = {
        "P1": "cluster_a",
        "P2": "cluster_a",
        "P3": "cluster_b",
        "P4": "cluster_c",
        "P5": "cluster_d",
    }

    _, split_rows, _ = generate_similarity_split_from_clusters(
        rows,
        fieldnames,
        0.6,
        0.2,
        0.2,
        sequence_col="protein_sequence",
        group_clusters=group_clusters,
        group_col="uniprot_id",
        seed=3,
    )

    split_by_protein = {
        row["uniprot_id"]: row["split"]
        for row in split_rows
    }
    assert split_by_protein["P1"] == split_by_protein["P2"]


def test_build_split_assignment_rows_keeps_only_id_and_split():
    fieldnames, rows = build_rows()
    _, split_rows, _ = generate_random_split_rows(
        rows,
        fieldnames,
        0.6,
        0.2,
        0.2,
        sequence_col="protein_sequence",
        group_col="uniprot_id",
        seed=7,
    )

    output_fieldnames, output_rows = build_split_assignment_rows(
        split_rows,
        fieldnames,
        id_col="uniprot_id",
    )

    assert output_fieldnames == ["uniprot_id", "split"]
    assert len(output_rows) == 5
    assert all(set(row) == {"uniprot_id", "split"} for row in output_rows)


def test_build_output_csv_path_appends_output_tag():
    output_path = build_output_csv_path(
        Path("data/example.csv"),
        mode="similarity",
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
        output_tag="id030 cov080 mode1",
    )

    assert (
        output_path.name
        == "example_similarity_mmseqs2_80_10_10_id030_cov080_mode1_splits.csv"
    )


def test_build_mmseqs_cluster_command_includes_optional_flags():
    command = build_mmseqs_cluster_command(
        Path("db"),
        Path("cluster_db"),
        Path("tmp"),
        identity_threshold=0.3,
        coverage_threshold=0.8,
        coverage_mode=1,
        cluster_mode=2,
        similarity_type=2,
        sensitivity=7.5,
        e_value=1e-4,
        alignment_mode=3,
        seq_id_mode=1,
        max_seqs=200,
        single_step_clustering=True,
        cluster_reassign=True,
    )

    assert command == [
        "mmseqs",
        "cluster",
        "db",
        "cluster_db",
        "tmp",
        "--min-seq-id",
        "0.3",
        "-c",
        "0.8",
        "--cov-mode",
        "1",
        "--cluster-mode",
        "2",
        "--similarity-type",
        "2",
        "-s",
        "7.5",
        "-e",
        "0.0001",
        "--alignment-mode",
        "3",
        "--seq-id-mode",
        "1",
        "--max-seqs",
        "200",
        "--single-step-clustering",
        "1",
        "--cluster-reassign",
        "1",
    ]
