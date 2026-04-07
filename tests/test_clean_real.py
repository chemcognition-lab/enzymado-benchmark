import csv
from pathlib import Path

from scripts.clean_utils.clean_real import (
    build_align_rows,
    build_argument_parser,
    resolve_align_family_processed_dir,
    resolve_align_processed_dir,
)
from src.paths_and_constants import Paths


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
    assert by_task["expression"] == (
        ["dataset_id", "protein_sequence", "source_split", "source_dataset", "expression"],
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
    assert by_task["thermostability"][1][0]["thermostability"] == "2.5"
    assert by_task["specific_activity"][1][0]["specific_activity"] == "3.5"


def test_align_defaults_cover_all_families_and_real_case_output_dir():
    parser = build_argument_parser()
    args = parser.parse_args(["--dataset", "align"])

    assert args.align_family == "all"

    paths = Paths()
    resolved_dir = resolve_align_processed_dir(
        args.out_processed,
        paths.data_processed_properties / "aromatic_products" / "erickson.csv",
        paths,
    )
    assert resolved_dir == paths.data_processed_real_case / "align"


def test_align_zero_shot_outputs_are_split_by_family_subdirectory(tmp_path):
    zero_shot_dir = tmp_path / "zero_shot"

    assert resolve_align_family_processed_dir(zero_shot_dir, "alpha-amylase") == (
        zero_shot_dir / "align_amylase"
    )
    assert resolve_align_family_processed_dir(zero_shot_dir, "b-glucosidase") == (
        zero_shot_dir / "align_glucosidase"
    )
    assert resolve_align_family_processed_dir(zero_shot_dir, "alkaline phosphatase") == (
        zero_shot_dir / "align_phosphatase"
    )


def test_align_non_zero_shot_outputs_keep_single_directory(tmp_path):
    processed_dir = tmp_path / "align"

    assert resolve_align_family_processed_dir(processed_dir, "alpha-amylase") == processed_dir
