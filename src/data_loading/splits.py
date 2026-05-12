from __future__ import annotations
"""Helpers for normalizing CSV rows and filtering inline split labels."""

from pathlib import Path

import pandas as pd


def clean_text(value: str | None) -> str:
    """Return a stripped string, using an empty string for missing values."""
    if value is None:
        return ""
    return value.strip()


def normalize_sequence(value: str | None) -> str:
    """Normalize sequences so split and embedding lookups use a stable key."""
    return "".join(clean_text(value).split()).upper()


def normalize_identifier(column_name: str, value: str | None) -> str:
    """Normalize identifiers, treating sequence-like columns specially."""
    if "sequence" in column_name:
        return normalize_sequence(value)
    return clean_text(value)


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file as strings and eagerly strip whitespace from every field."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame = frame.map(clean_text)
    return list(frame.columns), frame.to_dict("records")


def filter_rows_by_split(
    rows: list[dict[str, str]],
    split_name: str | None = None,
    split_column: str = "split",
) -> list[dict[str, str]]:
    """Filter rows to a requested split while preserving the split label."""
    if split_name is None:
        return list(rows)
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    if split_column not in frame.columns:
        raise ValueError(
            f"Missing split column '{split_column}'. Available columns: {list(frame.columns)}"
        )

    split_values = frame[split_column].map(clean_text)
    if split_values.eq("").any():
        raise ValueError(f"Missing split value in column '{split_column}'.")

    filtered = frame.loc[split_values == split_name].copy()
    filtered["resolved_split"] = split_name
    return filtered.to_dict("records")
