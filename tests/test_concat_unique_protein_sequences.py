from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data_prep.concat_unique_protein_sequences import build_unique_sequences


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

        unique_sequences, original_count = build_unique_sequences([self.root / "properties"])

        self.assertEqual(original_count, 3)
        rows_by_sequence = {
            row["protein_sequence"]: row["sequence_id"]
            for row in unique_sequences
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

        unique_sequences, original_count = build_unique_sequences([self.root / "properties"])

        self.assertEqual(original_count, 3)
        rows_by_sequence = {
            row["protein_sequence"]: row["sequence_id"]
            for row in unique_sequences
        }
        self.assertEqual(rows_by_sequence["AAAA"], "P1__bi_1")
        self.assertEqual(rows_by_sequence["BBBB"], "P1__bi_2")


if __name__ == "__main__":
    unittest.main()
