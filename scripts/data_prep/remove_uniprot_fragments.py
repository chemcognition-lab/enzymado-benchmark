#!/usr/bin/env python
"""Remove UniProt fragments from the tasks whose label describes the whole protein.

A UniProt entry flagged Fragment holds only part of a protein. For EC number,
kinetics, pH optimum, ligand binding and solubility the label belongs to the
whole protein, so a fragment row is dropped from every split. The residue tasks
keep fragments: their labels are positions on the deposited sequence, and every
one of them falls inside it.

Splits are not regenerated. Rows keep their split and regression targets are
re-standardized on the train rows that remain.

    python scripts/data_prep/remove_uniprot_fragments.py --fetch   # query UniProt
    python scripts/data_prep/remove_uniprot_fragments.py           # filter tasks
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from enzymado.data_loading import data_registry  # noqa: E402
from enzymado.splitting.standardization import build_standardized_dataset_rows  # noqa: E402

csv.field_size_limit(sys.maxsize)
TASKS = ("binding", "ec_number", "kcat", "km", "ph_opt", "solubility")
POOL = _REPO / "data" / "processed" / "protein_pool.csv"
FRAGMENTS = _REPO / "data" / "processed" / "uniprot_fragments.csv"
UNIPROT = "https://rest.uniprot.org"
ACCESSION = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})"
)


def base_accession(sequence_id: str) -> str | None:
    match = ACCESSION.match(sequence_id)
    return match.group(0) if match else None


def read_pool() -> list[tuple[str, str]]:
    with POOL.open(newline="") as handle:
        return [
            (row["sequence_id"], row["protein_sequence"].strip().upper())
            for row in csv.DictReader(handle)
        ]


def _request(url: str, data: bytes | None = None) -> tuple[str, dict[str, str]]:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, data=data, timeout=120) as response:
                return response.read().decode(), dict(response.headers)
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"UniProt request failed: {url}")


def _tsv_page(url: str) -> tuple[list[list[str]], dict[str, str], bool]:
    """Fetch one TSV page; return its good rows and whether it came back whole.

    UniProt sometimes cuts a page with a streaming-error line, in some cases at
    the same entry every time. After retries the rows before the cut are kept.
    """
    good: list[list[str]] = []
    headers: dict[str, str] = {}
    for attempt in range(4):
        text, headers = _request(url)
        lines = text.splitlines()
        if not lines or not lines[0].startswith(("From\t", "Entry\t")):
            time.sleep(10 * (attempt + 1))
            continue
        rows = [line.split("\t") for line in lines[1:]]
        good = [row for row in rows if len(row) >= 4 and row[0] and not row[0].startswith("Error")]
        if len(good) == len(rows):
            return good, headers, True
        time.sleep(10 * (attempt + 1))
    return good, headers, False


def _fragment_record(accession: str, fragment: str, length: str, non_ter: str) -> dict | None:
    if fragment.strip().lower() != "fragment":
        return None
    positions = {int(x) for x in re.findall(r"NON_TER (\d+)", non_ter)}
    return {
        "accession": accession,
        "length": int(length),
        "missing_n_terminus": 1 in positions,
        "missing_c_terminus": int(length) in positions,
    }


def fetch_fragments(accessions: list[str], chunk: int = 90000) -> tuple[list[dict], dict]:
    """Return the UniProt entries flagged Fragment among the given accessions."""
    fields = "accession,fragment,length,ft_non_ter"
    fragments, seen, release = {}, set(), None
    for start in range(0, len(accessions), chunk):
        ids = ",".join(accessions[start : start + chunk])
        body = urllib.parse.urlencode(
            {"ids": ids, "from": "UniProtKB_AC-ID", "to": "UniProtKB"}
        ).encode()
        job = json.loads(_request(f"{UNIPROT}/idmapping/run", body)[0])["jobId"]
        while json.loads(_request(f"{UNIPROT}/idmapping/status/{job}")[0]).get(
            "jobStatus"
        ) in ("NEW", "RUNNING"):
            time.sleep(10)
        url = f"{UNIPROT}/idmapping/uniprotkb/results/{job}?fields={fields}&format=tsv&size=500"
        while url:
            rows, headers, _ = _tsv_page(url)
            release = headers.get("X-UniProt-Release", release)
            for source, _, fragment, length, *rest in rows:
                if length.isdigit():
                    seen.add(source)
                    record = _fragment_record(source, fragment, length, (rest or [""])[0])
                    if record:
                        fragments[source] = record
            found = re.search(r'<([^>]+)>; rel="next"', headers.get("Link", ""))
            url = found.group(1) if found else ""

    # Accessions the mapping service left out are asked for directly, halving a
    # batch that UniProt cuts until the entry that breaks it stands alone.
    unresolved: list[str] = []

    def lookup(ids: list[str]) -> None:
        url = f"{UNIPROT}/uniprotkb/accessions?accessions={','.join(ids)}&fields={fields}&format=tsv"
        rows, _, whole = _tsv_page(url)
        if not whole and len(ids) > 1:
            middle = len(ids) // 2
            lookup(ids[:middle])
            lookup(ids[middle:])
            return
        if not whole:
            unresolved.extend(ids)
        for accession, fragment, length, *rest in rows:
            if length.isdigit():
                seen.add(accession)
                record = _fragment_record(accession, fragment, length, (rest or [""])[0])
                if record:
                    fragments[accession] = record

    missing = sorted(set(accessions) - seen)
    for start in range(0, len(missing), 500):
        lookup(missing[start : start + 500])
    summary = {
        "queried_accessions": len(accessions),
        "active_in_uniprot": len(seen),
        "inactive": len(accessions) - len(seen) - len(unresolved),
        "unresolved": sorted(unresolved),
        "fragments": len(fragments),
        "uniprot_release": release,
        "retrieved": datetime.date.today().isoformat(),
    }
    return list(fragments.values()), summary


def fragment_sequences() -> set[str]:
    """Sequences that only occur under accessions UniProt flags as Fragment."""
    with FRAGMENTS.open(newline="") as handle:
        flagged = {row["accession"] for row in csv.DictReader(handle)}
    fragment, complete = set(), set()
    for sequence_id, sequence in read_pool():
        accession = base_accession(sequence_id)
        (fragment if accession in flagged else complete).add(sequence)
    return fragment - complete


def filter_dataset(path: Path, metadata: dict, fragments: set[str]) -> dict[str, int]:
    """Drop fragment rows from one split file and refit its standardization."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = list(reader.fieldnames), list(reader)
    standardization = metadata["standardization"]
    targets = list(standardization.get("columns", {})) if standardization.get("enabled") else []
    standardized = {standardization["columns"][t]["standardized_column"] for t in targets}

    kept = [row for row in rows if row["protein_sequence"].strip().upper() not in fragments]
    removed = {split: 0 for split in ("train", "val", "test")}
    for row in rows:
        if row["protein_sequence"].strip().upper() in fragments:
            removed[row["split"]] += 1
    if targets:
        base_fields = [field for field in fieldnames if field not in standardized]
        base_rows = [{field: row[field] for field in base_fields} for row in kept]
        fieldnames, kept, fitted = build_standardized_dataset_rows(base_rows, base_fields, targets)
        standardization.update(fitted)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    metadata["summary"] = {
        split: {
            "rows": sum(row["split"] == split for row in kept),
            "proteins": len({row["protein_sequence"] for row in kept if row["split"] == split}),
        }
        for split in ("train", "val", "test")
    }
    previous = metadata.get("uniprot_fragments_removed", {})
    metadata["uniprot_fragments_removed"] = {
        split: previous.get(split, 0) + count for split, count in removed.items()
    }
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Query UniProt and write the fragment list.")
    args = parser.parse_args()

    if args.fetch:
        accessions = sorted({a for i, _ in read_pool() if (a := base_accession(i))})
        fragments, summary = fetch_fragments(accessions)
        with FRAGMENTS.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fragments[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(fragments, key=lambda row: row["accession"]))
        FRAGMENTS.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return

    fragments = fragment_sequences()
    print(f"{len(fragments)} fragment sequences")
    for task in TASKS:
        config = data_registry.TASK_REGISTRY[task]
        for variant, path in sorted(config.dataset_paths.items()):
            stem = path.name.split(f"_{variant}_")[0]
            metadata_path = path.with_name(f"{stem}_80_10_10.splits.metadata.json")
            metadata = json.loads(metadata_path.read_text())
            removed = filter_dataset(path, metadata["modes"][str(variant)], fragments)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"  {task} {variant}: removed {removed}")


if __name__ == "__main__":
    main()
