from src.dataset_splitting import (
    generate_random_split_rows,
    generate_similarity_split_from_clusters,
    validate_split_fractions,
)


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
