from __future__ import annotations
"""Dataset adapter that normalizes raw CSV rows before dataset construction."""

from dataclasses import dataclass
from pathlib import Path

from .splits import (
    clean_text,
    filter_rows_by_split,
    normalize_sequence,
    read_csv_rows,
)


@dataclass
class DatasetAdapter:
    """Load a task CSV and filter rows by its inline split column."""
    dataset_path: Path
    sample_id_candidates: tuple[str, ...]
    split_name: str | None = None
    split_column: str | None = None

    def load_rows(self) -> list[dict[str, str]]:
        """Read, normalize, filter, and tag rows with a stable `sample_id`."""
        _, rows = read_csv_rows(self.dataset_path)
        normalized_rows = [self._normalize_row(row) for row in rows]

        filtered_rows = filter_rows_by_split(
            normalized_rows,
            split_name=self.split_name,
            split_column=self.split_column or "split",
        )
        return [self._attach_sample_id(row) for row in filtered_rows]

    def _normalize_row(self, row: dict[str, str]) -> dict[str, str]:
        """Normalize text fields and uppercase sequence columns."""
        normalized = {key: clean_text(value) for key, value in row.items()}
        for sequence_column in ("protein_sequence", "wildtype_sequence"):
            if sequence_column in normalized:
                normalized[sequence_column] = normalize_sequence(normalized[sequence_column])
        return normalized

    def _attach_sample_id(self, row: dict[str, str]) -> dict[str, str]:
        """Attach the resolved sample identifier used throughout training."""
        sample_id = self._resolve_sample_id(row)
        normalized = dict(row)
        normalized["sample_id"] = sample_id
        return normalized

    def _resolve_sample_id(self, row: dict[str, str]) -> str:
        """Pick the first populated identifier column configured for the task."""
        for candidate in self.sample_id_candidates:
            value = clean_text(row.get(candidate))
            if value:
                return value
        raise ValueError(
            f"Unable to resolve sample_id using candidates {self.sample_id_candidates}. "
            f"Available columns: {sorted(row)}"
        )
