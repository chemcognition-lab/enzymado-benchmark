import csv
import json
import math
from pathlib import Path

import pytest

from scripts.data_prep.clean_real import (
    ERICKSON_BUFFER_COLUMNS,
    log10_offset,
    build_align_rows,
    build_argument_parser,
    build_erickson_rows,
    resolve_align_processed_dir,
    resolve_align_task_processed_dir,
    sync_align_benchmark_card,
)

from enzymado.paths_and_constants import Paths


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_build_align_rows_splits_each_task_into_its_own_output(tmp_path):
    align_root = tmp_path / "align"
    family_dir = align_root / "alpha-amylase"
    fieldnames = [
        "mutant",
        "dataset",
        "mutated_sequence",
        "expression",
        "thermostability",
        "specific activity",
    ]
    write_csv(
        family_dir / "train.csv",
        fieldnames,
        [
            {
                "mutant": "mut_1",
                "dataset": "alpha_source",
                "mutated_sequence": "AA AA",
                "expression": "1.5",
                "thermostability": "2.5",
                "specific activity": "3.5",
            }
        ],
    )

    interim_fieldnames, interim_rows, processed_outputs = build_align_rows(
        "alpha-amylase",
        align_root,
    )

    assert interim_fieldnames == [
        "dataset_id",
        "source_split",
        "mutant",
        "protein_sequence",
        "source_dataset",
        "expression",
        "thermostability",
        "specific_activity",
    ]
    assert interim_rows == [
        {
            "dataset_id": "align_alpha_amylase_000001",
            "source_split": "train",
            "mutant": "mut_1",
            "protein_sequence": "AAAA",
            "source_dataset": "alpha_source",
            "expression": "1.5",
            "thermostability": "2.5",
            "specific_activity": "3.5",
        }
    ]

    task_names = [task_name for _, task_name, _, _ in processed_outputs]
    assert task_names == ["expression", "thermostability", "specific_activity"]

    by_task = {
        task_name: (fieldnames, rows)
        for _, task_name, fieldnames, rows in processed_outputs
    }
    # One row per variant: repeated measurements are averaged and counted.
    assert by_task["expression"] == (
        ["dataset_id", "protein_sequence", "n_measurements", "expression"],
        [
            {
                "dataset_id": "align_alpha_amylase_000001",
                "protein_sequence": "AAAA",
                "n_measurements": "1",
                "expression": "1.5",
            }
        ],
    )
    assert by_task["thermostability"][1][0]["thermostability"] == "2.5"
    assert by_task["specific_activity"][1][0]["specific_activity"] == "3.5"


def test_align_defaults_cover_supported_families_and_real_case_output_dir():
    parser = build_argument_parser()
    args = parser.parse_args(["--dataset", "align"])

    assert args.align_family == "all"
    align_choices = parser._option_string_actions["--align-family"].choices
    assert "alpha-amylase" in align_choices
    assert "b-glucosidase" in align_choices
    assert "alkaline phosphatase" not in align_choices

    paths = Paths()
    resolved_dir = resolve_align_processed_dir(
        args.out_processed,
        paths.data_processed_properties / "aromatic_products" / "erickson.csv",
        paths,
    )
    assert resolved_dir == paths.data_processed_real_case


def test_align_task_outputs_use_current_directory_layout(tmp_path):
    processed_dir = tmp_path / "real_case"

    assert resolve_align_task_processed_dir(
        processed_dir, "alpha-amylase", "specific_activity"
    ) == (processed_dir / "align_amylase_activity")
    assert resolve_align_task_processed_dir(
        processed_dir, "alpha-amylase", "expression"
    ) == (processed_dir / "align_amylase_expression")
    assert resolve_align_task_processed_dir(
        processed_dir, "alpha-amylase", "thermostability"
    ) == (processed_dir / "align_amylase_stability")
    assert resolve_align_task_processed_dir(
        processed_dir, "b-glucosidase", "activity"
    ) == (processed_dir / "align_glucosidase_activity")
    assert resolve_align_task_processed_dir(
        processed_dir, "b-glucosidase", "expression"
    ) == (processed_dir / "align_glucosidase_expression")
    assert resolve_align_task_processed_dir(
        processed_dir, "b-glucosidase", "melting_point"
    ) == (processed_dir / "align_glucosidase_melting")


def test_sync_align_benchmark_card_replaces_legacy_align_entries(tmp_path):
    project_root = tmp_path
    processed_dir = project_root / "data" / "processed" / "real_case"
    benchmark_card_path = project_root / "benchmark_card.json"

    write_csv(
        processed_dir / "align_amylase_expression" / "alpha_amylase__expression.csv",
        [
            "dataset_id",
            "protein_sequence",
            "source_split",
            "source_dataset",
            "expression",
        ],
        [
            {
                "dataset_id": "align_alpha_amylase_000001",
                "protein_sequence": "AAAA",
                "source_split": "train",
                "source_dataset": "alpha_source",
                "expression": "1.5",
            }
        ],
    )
    write_csv(
        processed_dir
        / "align_glucosidase_melting"
        / "b_glucosidase__melting_point.csv",
        ["dataset_id", "protein_sequence", "source_split", "melting_point"],
        [
            {
                "dataset_id": "align_b_glucosidase_000001",
                "protein_sequence": "BBBB",
                "source_split": "test",
                "melting_point": "47.0",
            }
        ],
    )

    benchmark_card_path.write_text(
        json.dumps(
            {
                "benchmark_name": "test",
                "benchmark_version": "0",
                "root": ".",
                "data_root": "data",
                "properties": {},
                "real_case": {
                    "align": {"datasets": [{"path": "old/align.csv"}]},
                    "align_phosphatase_carboxy": {
                        "datasets": [{"path": "old/phosphatase.csv"}]
                    },
                    "cleeveland_galox": {
                        "datasets": [
                            {
                                "path": "data/processed/real_case/"
                                "cleeveland_galox/cleeveland.csv"
                            }
                        ]
                    },
                },
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    updated_sections = sync_align_benchmark_card(
        benchmark_card_path,
        processed_dir,
        project_root,
    )

    assert updated_sections == [
        "align_amylase_expression",
        "align_glucosidase_melting",
    ]

    updated_card = json.loads(benchmark_card_path.read_text(encoding="utf-8"))
    assert "align" not in updated_card["real_case"]
    assert "align_phosphatase_carboxy" not in updated_card["real_case"]
    assert "cleeveland_galox" in updated_card["real_case"]

    amylase_entry = updated_card["real_case"]["align_amylase_expression"]["datasets"][0]
    assert amylase_entry["dataset_id"] == "alpha_amylase__expression"
    assert amylase_entry["path"] == (
        "data/processed/real_case/align_amylase_expression/alpha_amylase__expression.csv"
    )
    assert amylase_entry["schema"]["target_col"] == "expression"

    glucosidase_entry = updated_card["real_case"]["align_glucosidase_melting"][
        "datasets"
    ][0]
    assert glucosidase_entry["dataset_id"] == "b_glucosidase__melting_point"
    assert glucosidase_entry["path"] == (
        "data/processed/real_case/align_glucosidase_melting/b_glucosidase__melting_point.csv"
    )
    assert glucosidase_entry["schema"]["target_col"] == "melting_point"


def _write_erickson_values(path: Path, rows: list[list[str]]) -> None:
    header = [["Enzyme ID", "", "101"], ["Rxn Temperature (°C)", "Rxn pH", "Sum (mg/L)"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(header + rows)


def test_erickson_rows_carry_ph_and_buffer_and_fix_the_relabelled_row(tmp_path):
    sequences = tmp_path / "sequences.csv"
    write_csv(sequences, ["Enzyme ID", "Protein Sequence"], [
        {"Enzyme ID": "101", "Protein Sequence": "MKV"},
    ])
    conditions = ["C6", "NP7", "NP7.5", "H7.5", "B8", "G9"]
    rows = [["30°C", c, "1"] for c in conditions] + [["30°C", "C6", "2"]]
    rows += [["40°C", c, "3"] for c in conditions[1:]]
    values = tmp_path / "values.csv"
    _write_erickson_values(values, rows)

    _, processed = build_erickson_rows(sequences, values)

    by_id = {row["dataset_id"]: row for row in processed}
    assert len(by_id) == len(processed) == 12
    fixed = by_id["101_40_c_c6"]
    assert (fixed["temperature"], fixed["ph"], fixed["sum_aromatic_products"]) == (
        "40.0", "6.0", "2.0",
    )
    assert float(fixed["log10_sum_aromatic_products_plus_one"]) == pytest.approx(
        math.log10(3.0)
    )
    hepes = by_id["101_30_c_h7_5"]
    assert hepes["ph"] == "7.5" and hepes["buffer_h"] == "1.0"
    for row in processed:
        assert sum(float(row[column]) for column in ERICKSON_BUFFER_COLUMNS) == 1.0


def test_erickson_row_fix_refuses_an_unexpected_label(tmp_path):
    sequences = tmp_path / "sequences.csv"
    write_csv(sequences, ["Enzyme ID", "Protein Sequence"], [
        {"Enzyme ID": "101", "Protein Sequence": "MKV"},
    ])
    rows = [["30°C", c, "1"] for c in ["C6", "NP7", "NP7.5", "H7.5", "B8", "G9", "NP7"]]
    values = tmp_path / "values.csv"
    _write_erickson_values(values, rows)

    with pytest.raises(ValueError, match="expected 30°C, C6"):
        build_erickson_rows(sequences, values)


def test_log10_offset_is_the_smallest_positive_value_only_when_zeros_exist():
    assert log10_offset([0.5, 2.0, 30.0]) == 0.0
    assert log10_offset([0.0, 0.006487, 7053.0]) == 0.006
    with pytest.raises(ValueError, match="negative"):
        log10_offset([-1.0, 2.0])
