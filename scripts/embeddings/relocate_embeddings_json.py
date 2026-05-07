"""Rewrite embedding manifest paths to a different storage root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relocated_path(original_path: str, output_root: Path, *, storage_format: str = "as-is") -> str:
    path = Path(original_path)
    parts = list(path.parts)
    try:
        data_index = parts.index("data")
    except ValueError as exc:
        raise ValueError(
            f"Cannot relocate path '{original_path}' because it does not contain a 'data' segment."
        ) from exc
    relative_under_data = Path(*parts[data_index + 1 :])
    relocated = output_root / relative_under_data
    if storage_format == "safetensors" and relocated.suffix == ".h5":
        if "embeddings_mean_pooled" in relocated.parts:
            relocated = relocated.with_suffix(".safetensors")
        else:
            relocated = relocated.with_name(f"{relocated.stem}_shards")
    return str(relocated)


def main(input_json: str, output_json: str, output_root: str, storage_format: str = "as-is") -> None:
    input_path = Path(input_json)
    output_path = Path(output_json)
    output_root_path = Path(output_root)

    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    relocated = {}
    for model_name, info in manifest.items():
        rewritten = dict(info)
        rewritten["path"] = relocated_path(
            str(info["path"]),
            output_root_path,
            storage_format=storage_format,
        )
        relocated[model_name] = rewritten

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(relocated, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote relocated manifest to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Source embeddings JSON manifest")
    parser.add_argument("output_json", help="Output JSON manifest with relocated paths")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory that should replace the repo's data root, e.g. /scratch/$USER/enzymado/data",
    )
    parser.add_argument(
        "--storage-format",
        choices=("as-is", "safetensors"),
        default="as-is",
        help=(
            "Rewrite .h5 manifest paths to safetensors locations. Mean-pooled paths become "
            ".safetensors files; per-residue paths become <model>_shards directories."
        ),
    )
    args = parser.parse_args()
    main(args.input_json, args.output_json, args.output_root, storage_format=args.storage_format)
