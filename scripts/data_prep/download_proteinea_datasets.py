#!/usr/bin/env python3
"""Export ProteinEA Hugging Face datasets into the repository's CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

SEQUENCE_ALIASES = (
    "protein_sequence",
    "sequence",
    "sequences",
    "Sequence",
    "seq",
    "SEQ",
    "primary",
)
ID_ALIASES = ("dataset_id", "id", "ID", "sequence_id")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "eval": "validation",
    "test": "test",
    "testing": "test",
}
DEFAULT_SPLIT_ORDER = ("train", "validation", "test")

DATASET_SPECS = {
    "fluorescence": {
        "hf_name": "proteinea/fluorescence",
        "target_column": "fluorescence",
        "target_aliases": (
            "fluorescence",
            "log_fluorescence",
            "log_fluorescence_intensity",
            "target",
            "label",
            "labels",
        ),
        "output_path": Path("data/raw/proteinea/proteinea_fluorescence_raw.csv"),
        "id_prefix": "fluorescence",
    },
    "solubility": {
        "hf_name": "proteinea/solubility",
        "target_column": "solubility",
        "target_aliases": (
            "solubility",
            "Solubility",
            "target",
            "label",
            "labels",
        ),
        "output_path": Path("data/raw/proteinea/proteinea_solubility_raw.csv"),
        "id_prefix": "solubility",
    },
}


def _load_dataset_module() -> tuple[Any, Any]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install it or use the Ankh environment "
            "defined in envs_embeddings/ankh_environment.yml."
        ) from exc
    return DatasetDict, load_dataset


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_sequence(value: Any) -> str:
    return "".join(clean_text(value).split()).upper()


def first_present(record: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key not in record:
            continue

        value = record[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value

    return None


def normalize_split(value: Any, fallback: str) -> str:
    cleaned = clean_text(value).lower()
    if cleaned in SPLIT_ALIASES:
        return SPLIT_ALIASES[cleaned]
    return fallback


def coerce_float(value: Any) -> str:
    return str(float(value))


def coerce_binary_label(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))

    if isinstance(value, (int, float)):
        return str(int(value))

    cleaned = clean_text(value).lower()
    if cleaned in {"1", "true", "yes", "soluble", "positive"}:
        return "1"
    if cleaned in {"0", "false", "no", "insoluble", "negative"}:
        return "0"

    return str(int(float(cleaned)))


def ordered_split_names(split_names: list[str]) -> list[str]:
    seen = set(split_names)
    ordered = [name for name in DEFAULT_SPLIT_ORDER if name in seen]
    ordered.extend(
        sorted(name for name in split_names if name not in DEFAULT_SPLIT_ORDER)
    )
    return ordered


def inspect_dataset(
    dataset_name: str,
    dataset_dict_cls: Any,
    load_dataset_fn: Any,
) -> None:
    spec = DATASET_SPECS[dataset_name]
    dataset_bundle = load_dataset_fn(spec["hf_name"])
    if isinstance(dataset_bundle, dataset_dict_cls):
        split_names = ordered_split_names(list(dataset_bundle.keys()))
        split_datasets = [
            (split_name, dataset_bundle[split_name]) for split_name in split_names
        ]
    else:
        split_datasets = [("train", dataset_bundle)]

    print(f"Dataset: {dataset_name} ({spec['hf_name']})")
    for split_name, split_dataset in split_datasets:
        print(f"  Split: {split_name} | rows={len(split_dataset)}")
        print(f"  Columns: {list(split_dataset.column_names)}")
        if len(split_dataset):
            print(f"  Example: {split_dataset[0]}")


def export_dataset(
    dataset_name: str,
    out_root: Path,
    overwrite: bool,  # noqa: FBT001 - retained for backward compatibility.
    dataset_dict_cls: Any,
    load_dataset_fn: Any,
) -> Path:
    spec = DATASET_SPECS[dataset_name]
    output_path = out_root / spec["output_path"]

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Re-run with --overwrite to replace it."
        )

    dataset_bundle = load_dataset_fn(spec["hf_name"])
    if isinstance(dataset_bundle, dataset_dict_cls):
        split_names = ordered_split_names(list(dataset_bundle.keys()))
        split_datasets = [
            (split_name, dataset_bundle[split_name]) for split_name in split_names
        ]
    else:
        split_datasets = [("train", dataset_bundle)]

    rows: list[dict[str, str]] = []
    next_id = 0
    for split_name, split_dataset in split_datasets:
        fallback_split = normalize_split(split_name, split_name)
        for record in split_dataset:
            sequence = normalize_sequence(first_present(record, SEQUENCE_ALIASES))
            target_value = first_present(record, spec["target_aliases"])
            row_split = normalize_split(record.get("split"), fallback_split)

            if not sequence or target_value is None or not row_split:
                continue

            record_id = first_present(record, ID_ALIASES)
            if record_id is None:
                record_id = f"{spec['id_prefix']}_{next_id}"

            if dataset_name == "fluorescence":
                target = coerce_float(target_value)
            else:
                target = coerce_binary_label(target_value)

            rows.append(
                {
                    "dataset_id": clean_text(record_id),
                    "protein_sequence": sequence,
                    spec["target_column"]: target,
                    "split": row_split,
                }
            )
            next_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "protein_sequence",
                spec["target_column"],
                "split",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ProteinEA fluorescence and/or solubility from Hugging Face "
            "and export them to the benchmark CSV format."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["fluorescence", "solubility", "all"],
        default="all",
        help="Dataset to download. Default: all",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("."),
        help=(
            "Repository root where the CSV paths will be created. Default: current "
            "directory"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CSV files if they already exist.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print split/column information instead of exporting CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dict_cls, load_dataset_fn = _load_dataset_module()

    dataset_names = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        if args.inspect:
            inspect_dataset(
                dataset_name=dataset_name,
                dataset_dict_cls=dataset_dict_cls,
                load_dataset_fn=load_dataset_fn,
            )
            continue

        export_dataset(
            dataset_name=dataset_name,
            out_root=args.out_root,
            overwrite=args.overwrite,
            dataset_dict_cls=dataset_dict_cls,
            load_dataset_fn=load_dataset_fn,
        )


if __name__ == "__main__":
    main()
