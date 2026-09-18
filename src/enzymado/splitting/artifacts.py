"""CSV, metadata, and output-path handling for dataset splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import core


def write_split_dataset(
    output_csv_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write the full split-labeled dataset (with optional standardized columns)."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_split_metadata(
    metadata_path: Path,
    *,
    mode: str,
    shared_fields: dict[str, object],
    mode_payload: dict[str, object],
) -> Path:
    """Merge mode-specific metadata into the unified per-dataset JSON."""
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        # Older files stored one mode at the top level. Replace that obsolete
        # shape instead of preserving contradictory provenance beside "modes".
        if (
            isinstance(existing, dict)
            and "mode" in existing
            and "modes" not in existing
        ):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}

    for key, value in shared_fields.items():
        existing[key] = value

    modes = existing.setdefault("modes", {})
    if not isinstance(modes, dict):
        modes = {}
        existing["modes"] = modes
    modes[mode] = mode_payload

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def resolve_csv_inputs(input_path: Path) -> list[Path]:
    """Expand a file or directory argument into source CSV datasets."""

    def is_source_dataset(path: Path) -> bool:
        return (
            path.suffix == ".csv"
            and not path.name.endswith("_splits.csv")
            and not path.name.endswith("_standardized.csv")
        )

    if input_path.is_file():
        return [input_path] if is_source_dataset(input_path) else []
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*.csv") if is_source_dataset(path)
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def format_fraction_label(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> str:
    """Format split fractions into the filename suffix used in the repo."""
    return (
        f"{int(round(train_fraction * 100))}_"
        f"{int(round(val_fraction * 100))}_"
        f"{int(round(test_fraction * 100))}"
    )


def build_output_csv_path(
    input_csv_path: Path,
    *,
    mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    output_root: Path | None = None,
    output_tag: str | None = None,
) -> Path:
    """Build the output path for the wide split-labeled CSV."""
    fraction_label = format_fraction_label(train_fraction, val_fraction, test_fraction)
    resolved_output_tag = core.sanitize_output_tag(output_tag)
    tag_suffix = f"_{resolved_output_tag}" if resolved_output_tag else ""
    if mode == "random":
        filename = (
            f"{input_csv_path.stem}_random_{fraction_label}"
            f"{tag_suffix}_standardized.csv"
        )
    elif mode == "similarity":
        filename = (
            f"{input_csv_path.stem}_similarity_mmseqs2_"
            f"{fraction_label}{tag_suffix}_standardized.csv"
        )
    else:
        raise ValueError(f"Unsupported split mode: {mode}.")

    if output_root is None:
        return input_csv_path.parent / filename

    return output_root / filename


def build_metadata_path(
    input_csv_path: Path,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    output_root: Path | None = None,
    output_tag: str | None = None,
) -> Path:
    """Build the unified metadata path shared across split modes for one dataset."""
    fraction_label = format_fraction_label(train_fraction, val_fraction, test_fraction)
    resolved_output_tag = core.sanitize_output_tag(output_tag)
    tag_suffix = f"_{resolved_output_tag}" if resolved_output_tag else ""
    filename = (
        f"{input_csv_path.stem}_{fraction_label}{tag_suffix}.splits.metadata.json"
    )
    if output_root is None:
        return input_csv_path.parent / filename
    return output_root / filename
