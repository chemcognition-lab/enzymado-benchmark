import ast
import csv
import json
from pathlib import Path

from src.dataset_splitting import (
    build_metadata_path,
    build_mmseqs_cluster_command,
    build_output_csv_path,
    build_random_split_specs,
    build_standardized_dataset_rows,
    generate_point_stratified_split_rows,
    generate_random_split_rows,
    generate_similarity_split_from_clusters,
    resolve_standardization_columns,
    split_dataset_csv,
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


def test_point_stratified_split_mixes_proteins_and_preserves_target_range():
    fieldnames = ["dataset_id", "protein_sequence", "target"]
    rows = [
        {
            "dataset_id": f"row_{index}",
            "protein_sequence": "AAAA" if index < 50 else "CCCC",
            "target": str(index),
        }
        for index in range(100)
    ]

    _, split_rows, summary = generate_point_stratified_split_rows(
        rows,
        fieldnames,
        0.8,
        0.1,
        0.1,
        sequence_col="protein_sequence",
        stratify_col="target",
        stratify_kind="regression",
        n_bins=5,
        seed=42,
    )

    assert {split["rows"] for split in summary.values()} == {10, 80}
    for split_name in ("train", "val", "test"):
        selected = [row for row in split_rows if row["split"] == split_name]
        assert {row["protein_sequence"] for row in selected} == {"AAAA", "CCCC"}
        targets = [float(row["target"]) for row in selected]
        assert min(targets) < 20
        assert max(targets) > 80


def test_point_stratified_split_preserves_binary_class_counts():
    fieldnames = ["dataset_id", "protein_sequence", "target"]
    rows = [
        {
            "dataset_id": f"row_{index}",
            "protein_sequence": f"SEQ{index}",
            "target": "1" if index < 20 else "0",
        }
        for index in range(100)
    ]

    _, split_rows, summary = generate_point_stratified_split_rows(
        rows,
        fieldnames,
        0.8,
        0.1,
        0.1,
        sequence_col="protein_sequence",
        stratify_col="target",
        stratify_kind="binary",
        seed=42,
    )

    assert {
        split: values["rows"]
        for split, values in summary.items()
    } == {"train": 80, "val": 10, "test": 10}
    positives = {
        split: sum(
            row["target"] == "1"
            for row in split_rows
            if row["split"] == split
        )
        for split in ("train", "val", "test")
    }
    assert positives == {"train": 16, "val": 2, "test": 2}


def test_point_stratified_split_preserves_hierarchical_ec_marginals():
    fieldnames = ["uniprot_id", "protein_sequence", "ec_number"]
    rows = []
    for index in range(100):
        ec_number = "1.1.1.1" if index < 50 else "2.2.2.2"
        if index % 10 == 0:
            ec_number += "; 3.3.3.3"
        rows.append(
            {
                "uniprot_id": f"P{index}",
                "protein_sequence": f"SEQ{index}",
                "ec_number": ec_number,
            }
        )

    _, split_rows, summary = generate_point_stratified_split_rows(
        rows,
        fieldnames,
        0.8,
        0.1,
        0.1,
        sequence_col="protein_sequence",
        stratify_col="ec_number",
        stratify_kind="multilabel_ec",
        seed=42,
    )

    assert {
        split: values["rows"]
        for split, values in summary.items()
    } == {"train": 80, "val": 10, "test": 10}
    rare_counts = {
        split: sum(
            "3.3.3.3" in row["ec_number"]
            for row in split_rows
            if row["split"] == split
        )
        for split in ("train", "val", "test")
    }
    assert rare_counts == {"train": 8, "val": 1, "test": 1}


def test_point_stratified_split_preserves_residue_density_range():
    fieldnames = ["uniprot_id", "protein_sequence", "active_site"]
    rows = [
        {
            "uniprot_id": f"P{index}",
            "protein_sequence": "A" * 100,
            "active_site": str(list(range(1, index % 10 + 2))),
        }
        for index in range(100)
    ]

    _, split_rows, summary = generate_point_stratified_split_rows(
        rows,
        fieldnames,
        0.8,
        0.1,
        0.1,
        sequence_col="protein_sequence",
        stratify_col="active_site",
        stratify_kind="residue_density",
        n_bins=5,
        seed=42,
    )

    assert {
        split: values["rows"]
        for split, values in summary.items()
    } == {"train": 80, "val": 10, "test": 10}
    for split in ("train", "val", "test"):
        counts = [
            len(ast.literal_eval(row["active_site"]))
            for row in split_rows
            if row["split"] == split
        ]
        assert min(counts) <= 2
        assert max(counts) >= 9


def test_random_split_specs_cover_all_benchmark_tasks():
    specs = build_random_split_specs()
    tasks = {spec.task for spec in specs}

    assert len(specs) == 18
    assert tasks == {
        "active_sites",
        "align_amylase_activity",
        "align_amylase_expression",
        "align_amylase_stability",
        "align_glucosidase_activity",
        "align_glucosidase_expression",
        "align_glucosidase_melting",
        "binding",
        "binding_site",
        "cleeveland_galox",
        "ddg",
        "dtm",
        "ec_number",
        "erickson_petase",
        "kcat",
        "km",
        "ph_opt",
        "solubility",
    }


def test_random_split_specs_match_task_label_kinds():
    specs = {spec.task: spec for spec in build_random_split_specs()}

    assert specs["solubility"].stratify_kind == "binary"
    assert specs["binding"].stratify_kind == "binary"
    assert specs["ec_number"].stratify_kind == "multilabel_ec"
    assert specs["active_sites"].stratify_kind == "residue_density"
    assert specs["binding_site"].stratify_kind == "residue_density"

    regression_tasks = set(specs) - {
        "active_sites",
        "binding",
        "binding_site",
        "ec_number",
        "solubility",
    }
    assert all(
        specs[task].stratify_kind == "regression"
        for task in regression_tasks
    )


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
        == "example_similarity_mmseqs2_80_10_10_id030_cov080_mode1_standardized.csv"
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


def test_standardization_fits_train_only_and_applies_to_all_splits():
    fieldnames = ["dataset_id", "protein_sequence", "target", "split"]
    rows = [
        {"dataset_id": "train_1", "protein_sequence": "AAAA", "target": "2", "split": "train"},
        {"dataset_id": "train_2", "protein_sequence": "CCCC", "target": "4", "split": "train"},
        {"dataset_id": "val_1", "protein_sequence": "DDDD", "target": "6", "split": "val"},
        {"dataset_id": "test_1", "protein_sequence": "EEEE", "target": "0", "split": "test"},
    ]

    output_fieldnames, standardized_rows, metadata = build_standardized_dataset_rows(
        rows,
        fieldnames,
        ["target"],
    )

    assert output_fieldnames[-1] == "target_standardized"
    fit_stats = metadata["columns"]["target"]["fit_statistics"]
    assert fit_stats["mean"] == 3.0
    assert fit_stats["std"] == 1.0
    values_by_id = {
        row["dataset_id"]: float(row["target_standardized"])
        for row in standardized_rows
    }
    assert values_by_id == {
        "train_1": -1.0,
        "train_2": 1.0,
        "val_1": 3.0,
        "test_1": -3.0,
    }


def test_standardization_autodetect_skips_binary_targets():
    fieldnames = ["dataset_id", "protein_sequence", "binding"]
    rows = [
        {"dataset_id": "row_1", "protein_sequence": "AAAA", "binding": "1"},
        {"dataset_id": "row_2", "protein_sequence": "CCCC", "binding": "0"},
    ]

    columns, skipped_columns, source = resolve_standardization_columns(fieldnames, rows)

    assert source == "auto"
    assert columns == []
    assert skipped_columns == {"binding": "binary_numeric_column"}


def test_split_dataset_csv_writes_wide_csv_and_unified_metadata(tmp_path):
    input_path = tmp_path / "toy.csv"
    fieldnames, rows = build_rows()
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    random_path = build_output_csv_path(
        input_path,
        mode="random",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )
    metadata_path = build_metadata_path(
        input_path,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )

    split_dataset_csv(
        input_path,
        random_path,
        mode="random",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        sequence_col="protein_sequence",
        group_col="uniprot_id",
        seed=7,
        standardization_cols=("target",),
        metadata_path=metadata_path,
    )

    assert random_path.exists()
    assert random_path.name == "toy_random_50_25_25_standardized.csv"
    with random_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames_out = reader.fieldnames or []
        rows_out = list(reader)
    assert "split" in fieldnames_out
    assert "target_standardized" in fieldnames_out
    assert {row["split"] for row in rows_out} == {"train", "val", "test"}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert set(metadata["modes"]) == {"random"}
    random_meta = metadata["modes"]["random"]
    assert random_meta["output_csv_path"] == str(random_path)
    assert random_meta["standardization"]["enabled"] is True
    assert random_meta["standardization"]["source"] == "explicit"
    assert random_meta["standardization"]["columns"]["target"]["fit_split"] == "train"


def test_split_dataset_csv_merges_modes_into_single_metadata(tmp_path):
    input_path = tmp_path / "toy.csv"
    fieldnames, rows = build_rows()
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = build_metadata_path(
        input_path,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )
    random_path = build_output_csv_path(
        input_path,
        mode="random",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )
    similarity_path = build_output_csv_path(
        input_path,
        mode="similarity",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )

    split_dataset_csv(
        input_path,
        random_path,
        mode="random",
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        sequence_col="protein_sequence",
        group_col="uniprot_id",
        seed=7,
        standardization_cols=("target",),
        metadata_path=metadata_path,
    )

    group_clusters = {f"P{i}": f"cluster_{i}" for i in range(1, 6)}
    from src.dataset_splitting import (
        build_standardized_dataset_rows as _build_std_rows,
        generate_similarity_split_from_clusters as _sim_from_clusters,
        write_split_dataset,
        write_split_metadata,
    )

    _, split_rows, summary = _sim_from_clusters(
        rows,
        fieldnames,
        0.5,
        0.25,
        0.25,
        sequence_col="protein_sequence",
        group_clusters=group_clusters,
        group_col="uniprot_id",
        seed=3,
    )
    output_fieldnames, std_rows, std_meta = _build_std_rows(
        split_rows, [*fieldnames, "split"], ["target"]
    )
    write_split_dataset(similarity_path, std_rows, output_fieldnames)
    write_split_metadata(
        metadata_path,
        mode="similarity",
        shared_fields={"input_csv_path": str(input_path)},
        mode_payload={
            "output_csv_path": str(similarity_path),
            "summary": summary,
            "standardization": {"enabled": True, **std_meta},
        },
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert set(metadata["modes"]) == {"random", "similarity"}
    assert metadata["modes"]["random"]["output_csv_path"] == str(random_path)
    assert metadata["modes"]["similarity"]["output_csv_path"] == str(similarity_path)
