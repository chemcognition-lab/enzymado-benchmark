"""Rewrite embedding manifest paths to a different storage root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relocated_path(original_path: str, output_root: Path) -> str:
    path = Path(original_path)
    parts = list(path.parts)
    try:
        data_index = parts.index("data")
    except ValueError as exc:
        raise ValueError(
            f"Cannot relocate path '{original_path}' because it does not contain a 'data' segment."
        ) from exc
    relative_under_data = Path(*parts[data_index + 1 :])
    return str(output_root / relative_under_data)


def main(input_json: str, output_json: str, output_root: str) -> None:
    input_path = Path(input_json)
    output_path = Path(output_json)
    output_root_path = Path(output_root)

    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    relocated = {}
    for model_name, info in manifest.items():
        rewritten = dict(info)
        rewritten["path"] = relocated_path(str(info["path"]), output_root_path)
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
    args = parser.parse_args()
    main(args.input_json, args.output_json, args.output_root)
