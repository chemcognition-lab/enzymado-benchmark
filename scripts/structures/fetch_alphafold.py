"""Download AlphaFold DB structures for a protein pool.

Reads a CSV with sequence_id and protein_sequence columns, queries the
AlphaFold DB API for each accession, verifies the pool sequence is a
substring of the AFDB sequence, and downloads the compressed CIF file.
Writes (and resumes from) fetch_manifest.csv under the output directory.

Example:
    python scripts/structures/fetch_alphafold.py \
        --input data/processed/protein_pool.csv \
        --output_dir /scratch/chemist/enzymado/data/structures
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_USER_AGENT = "enzymado-benchmark/1.0"
_AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
_MANIFEST_FIELDS = [
    "sequence_id",
    "accession",
    "version",
    "start",
    "end",
    "status",
    "cif_file",
]
_MAX_RETRIES = 5
_BASE_BACKOFF = 60.0  # seconds; grows by 2x on each attempt, capped at 600 s


def parse_accessions(sequence_id: str) -> list[str]:
    """Extract UniProt accessions from a sequence_id string.

    Handles both comma-separated accessions and double-underscore suffixes.
    'A0A009IHW8__bi' -> ['A0A009IHW8']
    'A0A097J792,Q9WTK1' -> ['A0A097J792', 'Q9WTK1']
    """
    base = sequence_id.split("__")[0]
    return [a.strip() for a in base.split(",") if a.strip()]


def find_match(pool_seq: str, afdb_seq: str) -> tuple[int, int] | None:
    """Return (start, end) if pool_seq is an exact substring of afdb_seq.

    Indices are base-0 with end exclusive, matching the residue slice
    recorded in fetch_manifest.csv.
    """
    idx = afdb_seq.find(pool_seq)
    if idx == -1:
        return None
    return idx, idx + len(pool_seq)


def read_plddt(cif_text: str) -> np.ndarray:
    """Parse pLDDT values from AFDB mmCIF text via CA B_iso_or_equiv.

    Reads the _atom_site loop, finds label_atom_id, label_seq_id, and
    B_iso_or_equiv columns, and returns a float32 array of length max_seq_id
    indexed 0-based (seq_id 1 -> index 0). Raises ValueError naming the
    first missing column if any of the three required columns are absent.
    """
    col_names: list[str] = []
    data_lines: list[str] = []
    state = "scan"  # "scan" | "header" | "data"

    for raw_line in cif_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if state == "data":
                break
            continue
        if line == "loop_":
            col_names = []
            data_lines = []
            state = "scan"
            continue
        if line.startswith("_atom_site."):
            state = "header"
            col_names.append(line)
            continue
        if state == "header" and not line.startswith("_"):
            state = "data"
        if state == "data":
            if line.startswith("_") or line == "loop_":
                break
            data_lines.append(line)

    col_map = {name: i for i, name in enumerate(col_names)}
    required_cols = (
        "_atom_site.label_atom_id",
        "_atom_site.label_seq_id",
        "_atom_site.B_iso_or_equiv",
    )
    for col in required_cols:
        if col not in col_map:
            raise ValueError(f"read_plddt: required mmCIF column {col!r} not found")

    atom_id_col = col_map["_atom_site.label_atom_id"]
    seq_id_col = col_map["_atom_site.label_seq_id"]
    bfactor_col = col_map["_atom_site.B_iso_or_equiv"]

    plddt_dict: dict[int, float] = {}
    for data_line in data_lines:
        tokens = data_line.split()
        if len(tokens) <= max(atom_id_col, seq_id_col, bfactor_col):
            continue
        if tokens[atom_id_col] != "CA":
            continue
        try:
            seq_id = int(tokens[seq_id_col])
            plddt_dict[seq_id] = float(tokens[bfactor_col])
        except ValueError:
            continue

    if not plddt_dict:
        return np.zeros(0, dtype=np.float32)

    max_seq = max(plddt_dict.keys())
    result = np.zeros(max_seq, dtype=np.float32)
    for seq_id, val in plddt_dict.items():
        result[seq_id - 1] = val  # 1-based seq_id -> 0-based index
    return result


def _load_done(manifest_path: Path) -> set[str]:
    """Return sequence_ids with a finalized (non-error) status.

    Rows with status='error' are excluded so they are retried on resume.
    """
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as fh:
        return {
            row["sequence_id"]
            for row in csv.DictReader(fh)
            if row.get("status", "") != "error"
        }


def _append_row(manifest_path: Path, row: dict) -> None:
    new_file = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _request_bytes(url: str, timeout: int = 60) -> bytes:
    """GET a URL, retrying on 429, 503 and connection failures.

    Raises urllib.error.HTTPError unchanged for 404 and any non-retryable
    HTTP status that survives all retry attempts. Respects the Retry-After
    header on 429; uses exponential backoff otherwise, starting at
    _BASE_BACKOFF seconds and capped at 600 s.
    """
    wait = _BASE_BACKOFF
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise  # 404: the resource does not exist, no retry
            if exc.code in (429, 503) and attempt < _MAX_RETRIES:
                retry_after = exc.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else wait
                time.sleep(wait_time)
                wait = min(wait * 2, 600.0)
                continue
            raise  # other HTTP errors or retries exhausted
        except urllib.error.URLError:
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
                wait = min(wait * 2, 600.0)
                continue
            raise
    raise RuntimeError(f"exhausted {_MAX_RETRIES} retries for {url}")  # defensive


def _fetch_api(accession: str, session: dict, delay: float) -> list[dict]:
    """Fetch AFDB prediction metadata for one accession (cached per run).

    Returns [] only for a confirmed 404. Any other network failure raises
    so the caller can record an error status instead of silently losing the
    protein.
    """
    if accession in session:
        return session[accession]
    url = _AFDB_API.format(accession)
    try:
        raw = _request_bytes(url, timeout=30)
        data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise  # 429/503 exhausted retries, or unexpected HTTP status
        data = []  # 404: accession genuinely absent from AFDB
    time.sleep(delay)
    session[accession] = data
    return data


_GZIP_MAGIC = b"\x1f\x8b"


def _write_cif_gz(raw: bytes, dest: Path) -> None:
    """Store CIF bytes gzip-compressed exactly once, atomically.

    The AFDB server sends some CIFs already gzipped (seen 2026-11-11 for 216
    of 1,426 files, all AF-00... model IDs); those bytes are written as they
    are instead of being compressed again. The file appears under its final
    name only when complete, so a job killed mid-write leaves no truncated CIF
    that the existence check would treat as downloaded. Bytes that are not a
    (gzipped) mmCIF raise RuntimeError, which the fetch loop records as an
    error row to retry.
    """
    data = raw if raw[:2] == _GZIP_MAGIC else gzip.compress(raw)
    try:
        text = gzip.decompress(data)
    except (OSError, EOFError) as exc:  # BadGzipFile is an OSError
        raise RuntimeError(f"{dest.name}: corrupt gzip download: {exc}") from exc
    if not text.lstrip().startswith(b"data_"):
        raise RuntimeError(f"{dest.name}: downloaded bytes are not an mmCIF file")
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _download_cif(cif_url: str, dest: Path) -> None:
    """Download a CIF file and store it gzip-compressed."""
    _write_cif_gz(_request_bytes(cif_url, timeout=60), dest)


def _resolve_match(
    pool_seq: str,
    accessions: list[str],
    session: dict,
    delay: float,
    raw_dir: Path,
) -> dict | None:
    """Search accessions for the first AFDB entry containing pool_seq.

    Returns a manifest row dict on success (including cif_file), or None if
    nothing matches. Uses modelEntityId for the filename so that polyprotein
    entries with many models sharing the same accession and version do not
    overwrite each other.
    """
    for acc in accessions:
        entries = _fetch_api(acc, session, delay)
        for entry in entries:
            match = find_match(pool_seq, entry.get("sequence", ""))
            if match is None:
                continue
            start, end = match
            version = entry.get("latestVersion", 0)
            model_id = entry.get("modelEntityId", f"{acc}-F1")
            cif_url = entry.get("cifUrl", "")
            cif_name = f"{model_id}-model_v{version}.cif.gz"
            cif_path = raw_dir / cif_name
            if not cif_path.exists():
                _download_cif(cif_url, cif_path)
            return {
                "accession": acc,
                "version": version,
                "start": start,
                "end": end,
                "status": "ok",
                "cif_file": f"raw/{cif_name}",
            }
    return None


def run(input_path: Path, output_dir: Path, delay: float) -> None:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fetch_manifest.csv"

    pool = pd.read_csv(input_path)
    done = _load_done(manifest_path)
    session: dict = {}

    for _, pool_row in pool.iterrows():
        seq_id = str(pool_row["sequence_id"])
        pool_seq = str(pool_row["protein_sequence"])
        if seq_id in done:
            continue

        accessions = parse_accessions(seq_id)
        try:
            match_row = _resolve_match(pool_seq, accessions, session, delay, raw_dir)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"{seq_id}: network error, writing error status: {exc}")
            _append_row(
                manifest_path,
                {
                    "sequence_id": seq_id,
                    "accession": accessions[0],
                    "version": "",
                    "start": "",
                    "end": "",
                    "status": "error",
                    "cif_file": "",
                },
            )
            continue

        if match_row is not None:
            _append_row(manifest_path, {"sequence_id": seq_id} | match_row)
        else:
            entries_flat: list[dict] = []
            for acc in accessions:
                entries_flat.extend(_fetch_api(acc, session, delay))
            status = "no_match" if entries_flat else "no_afdb"
            _append_row(
                manifest_path,
                {
                    "sequence_id": seq_id,
                    "accession": accessions[0],
                    "version": "",
                    "start": "",
                    "end": "",
                    "status": status,
                    "cif_file": "",
                },
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download AlphaFold DB structures for a protein pool."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="CSV with sequence_id and protein_sequence columns.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Root output directory. raw/ and fetch_manifest.csv go here.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between AFDB API requests (default 0.2).",
    )
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(
        input_path=Path(_args.input),
        output_dir=Path(_args.output_dir),
        delay=_args.delay,
    )
