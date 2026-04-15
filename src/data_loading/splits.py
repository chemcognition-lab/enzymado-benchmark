from __future__ import annotations
"""Helpers for normalizing CSV rows and resolving split assignments."""

import csv
from pathlib import Path


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
    """Read a CSV file and eagerly strip whitespace from every field."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: clean_text(value) for key, value in row.items()} for row in reader]
    return fieldnames, rows


def detect_split_id_column(fieldnames: list[str]) -> str:
    """Return the non-`split` column used to join a split assignment file."""
    candidates = [fieldname for fieldname in fieldnames if fieldname != "split"]
    if not candidates:
        raise ValueError("Split assignment file must contain one id column besides 'split'.")
    return candidates[0]


def load_split_assignments(split_path: Path) -> tuple[str, dict[str, str]]:
    """Load a split CSV into a normalized identifier-to-split mapping."""
    fieldnames, rows = read_csv_rows(split_path)
    id_column = detect_split_id_column(fieldnames)

    assignments: dict[str, str] = {}
    for row in rows:
        key = normalize_identifier(id_column, row.get(id_column))
        split_name = clean_text(row.get("split"))
        if not key or not split_name:
            continue
        assignments[key] = split_name
    return id_column, assignments


def resolve_row_split(
    row: dict[str, str],
    split_column: str | None = None,
    split_assignments: dict[str, str] | None = None,
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence"),
) -> str:
    """Resolve a row's split from an inline column or an external split table."""
    if split_column and split_column in row and clean_text(row.get(split_column)):
        return clean_text(row.get(split_column))

    if split_assignments is None:
        raise ValueError("No split column or split assignments were provided.")

    for candidate in split_id_candidates:
        if candidate not in row:
            continue
        key = normalize_identifier(candidate, row.get(candidate))
        if key in split_assignments:
            return split_assignments[key]

    raise KeyError(
        "Unable to resolve split assignment for row using candidates "
        f"{split_id_candidates}. Row keys: {sorted(row)}"
    )


def filter_rows_by_split(
    rows: list[dict[str, str]],
    split_name: str | None = None,
    split_column: str | None = None,
    split_assignments: dict[str, str] | None = None,
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence"),
) -> list[dict[str, str]]:
    """Filter rows to a requested split while preserving the resolved split label."""
    if split_name is None:
        return list(rows)

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        resolved_split = resolve_row_split(
            row,
            split_column=split_column,
            split_assignments=split_assignments,
            split_id_candidates=split_id_candidates,
        )
        if resolved_split != split_name:
            continue
        copied_row = dict(row)
        copied_row["resolved_split"] = resolved_split
        filtered_rows.append(copied_row)
    return filtered_rows
