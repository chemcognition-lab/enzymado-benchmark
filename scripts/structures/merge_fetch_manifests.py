"""Merge the local-match manifest with the AFDB API manifest.

match_afdb_local.py resolves most pool rows from the bulk files; the API run
(fetch_alphafold.py) covers the accessions those files lack. A row can appear
in both, for example an ID with two accessions where one is local. The local
ok row always wins, so no protein is prepared twice from different models.

Writes, under --output_dir (the directory of the local manifest and raw/):
    fetch_manifest.csv        every pool row, one final status each
    fetch_manifest_api.csv    only the API ok rows that are new, to prepare

The CIF of each new API row is hard-linked (or copied) from the API raw/
into --output_dir/raw/, so cif_file stays relative to --output_dir.

Example:
    python scripts/structures/merge_fetch_manifests.py \\
        --local_manifest /scratch/.../structures_prod/fetch_manifest_local.csv \\
        --api_manifest /scratch/.../afdb_api/fetch_manifest.csv \\
        --output_dir /scratch/.../structures_prod
"""

import argparse
from collections import Counter
import csv
from pathlib import Path

import fetch_alphafold
import match_afdb_local


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fetch_alphafold._MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def merge(
    local_rows: list[dict], api_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return (all rows in local order, new API ok rows).

    Raises ValueError if the API manifest has an ID the local one lacks or
    an ID twice, or if an API row has status error (re-run the fetch first).
    """
    api_by_id: dict[str, dict] = {}
    for row in api_rows:
        if row["sequence_id"] in api_by_id:
            raise ValueError(f"duplicate ID in API manifest: {row['sequence_id']}")
        if row["status"] == "error":
            raise ValueError(f"API row still has status error: {row['sequence_id']}")
        api_by_id[row["sequence_id"]] = row
    local_ids = {row["sequence_id"] for row in local_rows}
    extra = set(api_by_id) - local_ids
    if extra:
        raise ValueError(f"{len(extra)} API IDs are not in the local manifest")

    merged, new_api = [], []
    for row in local_rows:
        api_row = api_by_id.get(row["sequence_id"])
        if row["status"] == "ok" or api_row is None:
            merged.append(row)
        elif api_row["status"] == "ok":
            merged.append(api_row)
            new_api.append(api_row)
        elif api_row["status"] == "no_match" or row["status"] == "no_match":
            merged.append(api_row | {"status": "no_match"})
        else:
            merged.append(api_row)
    return merged, new_api


def run(local_manifest: Path, api_manifest: Path, output_dir: Path) -> None:
    merged, new_api = merge(_read(local_manifest), _read(api_manifest))
    api_root = api_manifest.parent
    raw_dir = output_dir / "raw"
    for row in new_api:
        src = api_root / row["cif_file"]
        dest = output_dir / row["cif_file"]
        if dest.parent != raw_dir:
            raise ValueError(f"unexpected cif_file location: {row['cif_file']}")
        if not dest.exists():
            match_afdb_local._link_or_copy(src, dest)
    _write(output_dir / "fetch_manifest.csv", merged)
    _write(output_dir / "fetch_manifest_api.csv", new_api)
    print(f"merged statuses: {dict(Counter(r['status'] for r in merged))}")
    print(f"new API rows to prepare: {len(new_api)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge local-match and AFDB API fetch manifests."
    )
    parser.add_argument("--local_manifest", required=True, type=str,
                        help="fetch_manifest_local.csv from match_afdb_local.py.")
    parser.add_argument("--api_manifest", required=True, type=str,
                        help="fetch_manifest.csv from fetch_alphafold.py.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory holding the local manifest and raw/.")
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(Path(_args.local_manifest), Path(_args.api_manifest), Path(_args.output_dir))
