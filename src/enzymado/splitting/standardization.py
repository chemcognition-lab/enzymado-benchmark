"""Train-fitted target standardization for split datasets."""

from __future__ import annotations

import math

from . import core

STANDARDIZATION_EXCLUDED_COLUMNS = {
    "active_site",
    "binding_positions",
    "ec_number",
    "id",
    "ligand_id",
    "ligand_name",
    "mutation",
    "n_measurements",
    "ph",
    "resolved_split",
    "smiles",
    "source_split",
    "split",
    "substrate_concentration",
    "temperature",
}
STANDARDIZATION_EXCLUDED_SUFFIXES = (
    "_count",
    "_id",
    "_sem",
    "_std",
)
KNOWN_REGRESSION_TARGET_COLUMNS = {
    "activity",
    "ddg",
    "dtm",
    "expression",
    "kcat",
    "km",
    "melting_point",
    "phopt_median",
    "specific_activity",
    "sum_aromatic_products",
    "target",
    "thermostability",
}


def parse_finite_float(value: str | None) -> float | None:
    """Parse a finite float, returning ``None`` for blank or invalid values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Return whether all present values in a column are finite numbers."""
    saw_value = False
    for row in rows:
        raw_value = row.get(column)
        if raw_value is None or not str(raw_value).strip():
            continue
        if parse_finite_float(raw_value) is None:
            return False
        saw_value = True
    return saw_value


def is_binary_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Return whether a numeric column contains only binary 0/1 values."""
    values: set[float] = set()
    for row in rows:
        parsed = parse_finite_float(row.get(column))
        if parsed is None:
            continue
        if parsed not in {0.0, 1.0}:
            return False
        values.add(parsed)
    return bool(values)


def should_exclude_standardization_column(column: str) -> bool:
    """Return whether a column should be excluded from target standardization."""
    normalized = column.lower()
    if normalized in STANDARDIZATION_EXCLUDED_COLUMNS:
        return True
    if "sequence" in normalized:
        return True
    return normalized.endswith(STANDARDIZATION_EXCLUDED_SUFFIXES)


def resolve_standardization_columns(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    standardization_cols: tuple[str, ...] | None = None,
) -> tuple[list[str], dict[str, str], str]:
    """Choose scalar columns to standardize, preferring explicit user input."""
    if standardization_cols is not None:
        resolved_cols = list(dict.fromkeys(standardization_cols))
        missing = [column for column in resolved_cols if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Standardization columns are missing from the dataset: {missing}."
            )
        non_numeric = [
            column for column in resolved_cols if not is_numeric_column(rows, column)
        ]
        if non_numeric:
            raise ValueError(
                "Standardization columns must contain only finite numeric values "
                f"or blanks. Non-numeric columns: {non_numeric}."
            )
        return resolved_cols, {}, "explicit"

    numeric_candidates: list[str] = []
    skipped_columns: dict[str, str] = {}
    for column in fieldnames:
        if not is_numeric_column(rows, column):
            continue
        if is_binary_numeric_column(rows, column):
            skipped_columns[column] = "binary_numeric_column"
            continue
        if should_exclude_standardization_column(column):
            skipped_columns[column] = "excluded_id_feature_or_metadata_column"
            continue
        numeric_candidates.append(column)

    known_targets = [
        column
        for column in numeric_candidates
        if column.lower() in KNOWN_REGRESSION_TARGET_COLUMNS
    ]
    if known_targets:
        return known_targets, skipped_columns, "auto_known_target"
    if len(numeric_candidates) == 1:
        return numeric_candidates, skipped_columns, "auto_single_numeric_column"

    for column in numeric_candidates:
        skipped_columns[column] = "ambiguous_numeric_column_use_standardize_col"
    return [], skipped_columns, "auto"


def make_standardized_column_name(
    column: str,
    used_fieldnames: set[str],
) -> str:
    """Return a non-conflicting standardized target column name."""
    base_name = f"{column}_standardized"
    candidate = base_name
    suffix = 2
    while candidate in used_fieldnames:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_fieldnames.add(candidate)
    return candidate


def format_standardized_value(value: float) -> str:
    "Format standardized values compactly while keeping round-trip precision useful."
    return format(value, ".12g")


def build_standardized_dataset_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    target_columns: list[str],
) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    """Append z-scored target columns fit only on train rows."""
    output_fieldnames = list(fieldnames)
    used_fieldnames = set(output_fieldnames)
    standardized_column_by_target = {
        column: make_standardized_column_name(column, used_fieldnames)
        for column in target_columns
    }
    output_fieldnames.extend(standardized_column_by_target.values())

    standardized_rows = [dict(row) for row in rows]
    columns_metadata: dict[str, object] = {}
    for column in target_columns:
        train_values = [
            parsed
            for row in rows
            if row.get("split") == "train"
            for parsed in [parse_finite_float(row.get(column))]
            if parsed is not None
        ]
        if not train_values:
            raise ValueError(
                f"Cannot standardize column '{column}' because train has no "
                "numeric values."
            )

        train_mean = sum(train_values) / len(train_values)
        train_variance = sum((value - train_mean) ** 2 for value in train_values)
        train_variance /= len(train_values)
        train_std = math.sqrt(train_variance)
        scale = 1.0 if train_std == 0.0 else train_std
        standardized_column = standardized_column_by_target[column]

        value_counts_by_split = {split_name: 0 for split_name in core.SPLIT_NAMES}
        missing_counts_by_split = {split_name: 0 for split_name in core.SPLIT_NAMES}
        for row in standardized_rows:
            split_name = row.get("split")
            parsed = parse_finite_float(row.get(column))
            if split_name in value_counts_by_split:
                if parsed is None:
                    missing_counts_by_split[split_name] += 1
                else:
                    value_counts_by_split[split_name] += 1
            row[standardized_column] = (
                ""
                if parsed is None
                else format_standardized_value((parsed - train_mean) / scale)
            )

        columns_metadata[column] = {
            "standardized_column": standardized_column,
            "fit_split": "train",
            "fit_statistics": {
                "mean": train_mean,
                "std": train_std,
                "uncertainty": train_std,
                "scale": scale,
                "ddof": 0,
                "n": len(train_values),
                "zero_std_replaced_with_one": train_std == 0.0,
            },
            "applied_to_splits": list(core.SPLIT_NAMES),
            "numeric_values_by_split": value_counts_by_split,
            "missing_or_invalid_by_split": missing_counts_by_split,
        }

    metadata = {
        "method": "z_score",
        "fit_split": "train",
        "leakage_guard": (
            "Mean and uncertainty/std are fit on train rows only; the same "
            "train-derived values are applied to val and test."
        ),
        "columns": columns_metadata,
    }
    return output_fieldnames, standardized_rows, metadata
