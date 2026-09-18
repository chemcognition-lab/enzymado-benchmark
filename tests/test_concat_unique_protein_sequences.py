from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import scripts.data_prep.concat_unique_protein_sequences as concat_sequences
import scripts.data_prep.make_fasta_from_unique_sequences as fasta_sequences


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ConcatUniqueProteinSequencesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_conflicting_ids_across_datasets_get_dataset_suffixes(self) -> None:
        write_csv(
            self.root / "properties" / "active_sites" / "swissprot_active_site.csv",
            ["uniprot_id", "protein_sequence"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA"},
                {"uniprot_id": "P2", "protein_sequence": "CCCC"},
            ],
        )
        write_csv(
            self.root / "properties" / "binding_site" / "swissprot_binding_site.csv",
            ["uniprot_id", "protein_sequence"],
            [
                {"uniprot_id": "P1", "protein_sequence": "BBBB"},
            ],
        )

        unique_sequences, original_count = concat_sequences.build_unique_sequences(
            [self.root / "properties"]
        )

        self.assertEqual(original_count, 3)
        rows_by_sequence = {
            row["protein_sequence"]: row["sequence_id"] for row in unique_sequences
        }
        self.assertEqual(rows_by_sequence["AAAA"], "P1__as")
        self.assertEqual(rows_by_sequence["BBBB"], "P1__bs")
        self.assertEqual(rows_by_sequence["CCCC"], "P2")

    def test_conflicting_ids_within_same_dataset_get_stable_indexes(self) -> None:
        write_csv(
            self.root / "properties" / "binding" / "biolip_binding.csv",
            ["uniprot_id", "protein_sequence"],
            [
                {"uniprot_id": "P1", "protein_sequence": "BBBB"},
                {"uniprot_id": "P1", "protein_sequence": "AAAA"},
                {"uniprot_id": "P1", "protein_sequence": "BBBB"},
            ],
        )

        unique_sequences, original_count = concat_sequences.build_unique_sequences(
            [self.root / "properties"]
        )

        self.assertEqual(original_count, 3)
        rows_by_sequence = {
            row["protein_sequence"]: row["sequence_id"] for row in unique_sequences
        }
        self.assertEqual(rows_by_sequence["AAAA"], "P1__bi_1")
        self.assertEqual(rows_by_sequence["BBBB"], "P1__bi_2")

    def test_uniprot_ids_take_priority_when_sequences_repeat(self) -> None:
        write_csv(
            self.root / "properties" / "ddg_mut" / "fireprotdb_ddg.csv",
            ["uniprot_id", "protein_sequence", "wildtype_sequence"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": " mu t ",
                    "wildtype_sequence": "wild",
                }
            ],
        )
        write_csv(
            self.root / "real_case" / "example" / "example.csv",
            ["dataset_id", "protein_sequence"],
            [{"dataset_id": "D1", "protein_sequence": "MUT"}],
        )

        unique_sequences, original_count = concat_sequences.build_unique_sequences(
            [self.root / "properties", self.root / "real_case"]
        )

        self.assertEqual(original_count, 3)
        self.assertEqual(
            unique_sequences,
            [
                {"sequence_id": "P1", "protein_sequence": "MUT"},
                {
                    "sequence_id": "P1__wildtype_sequence",
                    "protein_sequence": "WILD",
                },
            ],
        )


def test_fasta_output_order_deduplication_and_header_sanitization(tmp_path) -> None:
    """Lock the exact FASTA text produced from the canonical sequence index."""
    input_csv = tmp_path / "sequences.csv"
    output_fasta = tmp_path / "nested" / "sequences.fasta"
    write_csv(
        input_csv,
        ["sequence_id", "protein_sequence"],
        [
            {"sequence_id": "A/B", "protein_sequence": "AAAA"},
            {"sequence_id": "A/B", "protein_sequence": "CCCC"},
            {"sequence_id": "C.D", "protein_sequence": " AA AA "},
            {"sequence_id": "", "protein_sequence": "EEEE"},
        ],
    )

    written = fasta_sequences.build_fasta(input_csv, output_fasta)

    assert written == 2
    assert output_fasta.read_text(encoding="utf-8") == (">A_B\nAAAA\n>C_D\nAA AA\n")


def test_fasta_parser_keeps_required_path_flags() -> None:
    """Lock the current FASTA CLI flag names and parsed Path values."""
    args = fasta_sequences.create_parser().parse_args(
        ["--input", "input.csv", "--output", "output.fasta"]
    )

    assert args.input == Path("input.csv")
    assert args.output == Path("output.fasta")


def test_sequence_index_parser_keeps_defaults_and_repeatable_inputs() -> None:
    """Lock canonical-index CLI defaults and repeated input directory parsing."""
    parser = concat_sequences.create_parser()
    defaults = parser.parse_args([])
    selected = parser.parse_args(
        [
            "--input-dir",
            "properties",
            "--input-dir",
            "real_case",
            "--output",
            "sequences.csv",
        ]
    )

    assert defaults.input_dir is None
    assert defaults.output == concat_sequences.DEFAULT_OUTPUT
    assert selected.input_dir == [Path("properties"), Path("real_case")]
    assert selected.output == Path("sequences.csv")


if __name__ == "__main__":
    unittest.main()
