from __future__ import annotations
"""Load, filter, and tag rows from a task CSV file."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def normalize_sequence(value: str | None) -> str:
    """Return a normalized sequence key for embedding store lookups."""
    if not value:
        return ""
    return "".join(value.split()).upper()


def normalize_identifier(column_name: str, value: str | None) -> str:
    """Normalize identifiers, treating sequence-like columns as sequences."""
    if "sequence" in column_name:
        return normalize_sequence(value)
    return (value or "").strip()


def clean_text(value: str | None) -> str:
    """Return a stripped string, using an empty string for missing values."""
    return (value or "").strip()


def filter_rows_by_split(
    rows: list[dict[str, str]],
    split_name: str | None = None,
    split_column: str = "split",
) -> list[dict[str, str]]:
    """Return only rows matching split_name, or all rows if split_name is None."""
    if split_name is None:
        return list(rows)
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    if split_column not in frame.columns:
        raise ValueError(
            f"Missing split column '{split_column}'. Available columns: {list(frame.columns)}"
        )
    split_values = frame[split_column].str.strip()
    if split_values.eq("").any():
        raise ValueError(f"Missing split value in column '{split_column}'.")

    filtered = frame.loc[split_values == split_name].copy()
    filtered["resolved_split"] = split_name
    return filtered.to_dict("records")


@dataclass
class TaskCSVLoader:
    """Load a task CSV and return normalized rows tagged with a stable sample_id."""
    dataset_path: Path
    index_columns: tuple[str, ...]
    split_name: str | None = None
    split_column: str = "split"
    sequence_column: str = "protein_sequence"

    def load_rows(self) -> list[dict[str, str]]:
        frame = pd.read_csv(self.dataset_path, dtype=str, keep_default_na=False)
        rows = frame.to_dict("records")
        filtered = filter_rows_by_split(rows, self.split_name, self.split_column)
        return [self._attach_sample_id(row) for row in filtered]

    def _attach_sample_id(self, row: dict[str, str]) -> dict[str, str]:
        row = dict(row)
        row["sample_id"] = self._resolve_sample_id(row)
        return row

    def _resolve_sample_id(self, row: dict[str, str]) -> str:
        for candidate in self.index_columns:
            value = (row.get(candidate) or "").strip()
            if value:
                return value
        sequence = (row.get(self.sequence_column) or "").strip()
        if sequence:
            return sequence
        raise ValueError(
            f"Unable to resolve sample_id using index columns {self.index_columns} "
            f"or sequence column '{self.sequence_column}'. "
            f"Available columns: {sorted(row)}"
        )
