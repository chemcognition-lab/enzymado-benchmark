"""Load, filter, and tag rows from a task CSV file."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from enzymado import paths_and_constants

_C = paths_and_constants.Constants


def normalize_sequence(value: str | None) -> str:
    """Return a normalized sequence key for embedding store lookups."""
    if not value:
        return ""
    return "".join(value.split()).upper()


def clean_text(value: str | None) -> str:
    """Return a stripped string, using an empty string for missing values."""
    return (value or "").strip()


def filter_rows_by_split(
    rows: list[dict[str, str]],
    split_name: str | None = None,
    split_column: str = _C.split,
) -> list[dict[str, str]]:
    """Return only rows matching split_name, or all rows if split_name is None."""
    if split_name is None:
        return list(rows)
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    if split_column not in frame.columns:
        raise ValueError(
            f"Missing split column '{split_column}'. "
            f"Available columns: {list(frame.columns)}"
        )
    split_values = frame[split_column].str.strip()
    if split_values.eq("").any():
        raise ValueError(f"Missing split value in column '{split_column}'.")

    filtered = frame.loc[split_values == split_name].copy()
    filtered["resolved_split"] = split_name
    return filtered.to_dict("records")


@dataclass
class TaskCSVLoader:
    """Load normalized rows with readable and prediction-unique identifiers."""

    dataset_path: Path
    index_columns: tuple[str, ...]
    split_name: str | None = None
    split_column: str = _C.split
    sequence_column: str = _C.protein_sequence

    def load_rows(self) -> list[dict[str, str]]:
        frame = pd.read_csv(self.dataset_path, dtype=str, keep_default_na=False)
        rows = frame.to_dict("records")
        filtered = filter_rows_by_split(rows, self.split_name, self.split_column)
        return [self._attach_sample_id(row) for row in filtered]

    def _attach_sample_id(self, row: dict[str, str]) -> dict[str, str]:
        row = dict(row)
        row["sample_id"] = self._resolve_sample_id(row)
        row["prediction_id"] = self._resolve_prediction_id(row)
        return row

    def _resolve_prediction_id(self, row: dict[str, str]) -> str:
        """Build a stable row identity while retaining a readable sample prefix."""
        identity_payload = {
            key: str(value)
            for key, value in row.items()
            if key not in {self.split_column, "resolved_split", "prediction_id"}
        }
        encoded = json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"{row['sample_id']}:{digest}"

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
