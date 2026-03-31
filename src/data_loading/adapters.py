from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .splits import (
    clean_text,
    filter_rows_by_split,
    load_split_assignments,
    normalize_sequence,
    read_csv_rows,
)


@dataclass
class DatasetAdapter:
    dataset_path: Path
    sample_id_candidates: tuple[str, ...]
    split_name: str | None = None
    split_column: str | None = None
    split_path: Path | None = None
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")

    def load_rows(self) -> list[dict[str, str]]:
        _, rows = read_csv_rows(self.dataset_path)
        normalized_rows = [self._normalize_row(row) for row in rows]

        split_assignments = None
        if self.split_path is not None and self.split_name is not None:
            _, split_assignments = load_split_assignments(self.split_path)

        filtered_rows = filter_rows_by_split(
            normalized_rows,
            split_name=self.split_name,
            split_column=self.split_column,
            split_assignments=split_assignments,
            split_id_candidates=self.split_id_candidates,
        )
        return [self._attach_sample_id(row) for row in filtered_rows]

    def _normalize_row(self, row: dict[str, str]) -> dict[str, str]:
        normalized = {key: clean_text(value) for key, value in row.items()}
        for sequence_column in ("protein_sequence", "wildtype_sequence"):
            if sequence_column in normalized:
                normalized[sequence_column] = normalize_sequence(normalized[sequence_column])
        return normalized

    def _attach_sample_id(self, row: dict[str, str]) -> dict[str, str]:
        sample_id = self._resolve_sample_id(row)
        normalized = dict(row)
        normalized["sample_id"] = sample_id
        return normalized

    def _resolve_sample_id(self, row: dict[str, str]) -> str:
        for candidate in self.sample_id_candidates:
            value = clean_text(row.get(candidate))
            if value:
                return value
        raise ValueError(
            f"Unable to resolve sample_id using candidates {self.sample_id_candidates}. "
            f"Available columns: {sorted(row)}"
        )


class ProteinGlobalAdapter(DatasetAdapter):
    pass


class ProteinLigandAdapter(DatasetAdapter):
    pass


class MutationDeltaAdapter(DatasetAdapter):
    pass


class ProteinResidueAdapter(DatasetAdapter):
    pass
