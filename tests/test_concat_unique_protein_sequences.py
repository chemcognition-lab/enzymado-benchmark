import csv
from pathlib import Path

from scripts.clean_utils.concat_unique_protein_sequences import build_unique_sequences


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_build_unique_sequences_includes_wildtype_and_real_case(tmp_path):
    properties_dir = tmp_path / "properties"
    real_case_dir = tmp_path / "real_case"

    write_csv(
        properties_dir / "ddg_mut" / "mutations.csv",
        ["dataset_id", "protein_sequence", "wildtype_sequence", "mutation"],
        [
            {
                "dataset_id": "mut_1",
                "protein_sequence": "M U T",
                "wildtype_sequence": "WILD",
                "mutation": "A1B",
            }
        ],
    )
    write_csv(
        properties_dir / "ddg_mut" / "mutations_random_80_10_10_splits.csv",
        ["dataset_id", "protein_sequence"],
        [{"dataset_id": "split_only", "protein_sequence": "SHOULDNOTLOAD"}],
    )
    write_csv(
        properties_dir / "protein_sequences_unique.csv",
        ["sequence_id", "protein_sequence"],
        [{"sequence_id": "old", "protein_sequence": "OLD"}],
    )
    write_csv(
        real_case_dir / "align" / "alpha_amylase__expression.csv",
        ["dataset_id", "protein_sequence", "expression"],
        [{"dataset_id": "real_1", "protein_sequence": "REAL", "expression": "1.0"}],
    )

    unique_rows, original_count = build_unique_sequences([properties_dir, real_case_dir])

    assert original_count == 3
    assert {row["sequence_id"]: row["protein_sequence"] for row in unique_rows} == {
        "mut_1": "MUT",
        "mut_1__wildtype_sequence": "WILD",
        "real_1": "REAL",
    }
