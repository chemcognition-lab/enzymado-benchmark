"""Match pool sequences to AlphaFold DB files already on disk.

Uses the Swiss-Prot bulk tar (swissprot_cif_v6.tar) and a directory of
per-accession F1 files (the rest/ download) instead of the AFDB API. For each
pool row, the accessions are tried in order and the first model whose
sequence contains the pool sequence exactly wins, as in fetch_alphafold.py.

Writes, under --output_dir:
    raw/AF-<accession>-F1-model_v<version>.cif.gz   one per matched model
    fetch_manifest_local.csv                        fetch_alphafold.py columns

Status per row: ok, no_match (a local model exists but does not contain the
sequence), or not_local (no local model for any accession; the AFDB API may
still have one). The tar is read once as a stream; members are copied
byte-for-byte, so the files are identical to the ones in the tar.

Example:
    python scripts/structures/match_afdb_local.py \\
        --input data/processed/protein_pool.csv \\
        --tar /scratch/.../afdb_bulk/swissprot_cif_v6.tar \\
        --tar_list /scratch/.../afdb_bulk/swissprot_cif_v6.list \\
        --rest_dir /scratch/.../afdb_bulk/rest \\
        --output_dir /scratch/.../structures_prod
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import os
from pathlib import Path
import re
import shutil
import tarfile

import fetch_alphafold

_SEQ_KEY = "_entity_poly.pdbx_seq_one_letter_code_can"
_NAME_RE = re.compile(r"^AF-(?P<acc>.+)-F1-model_v(?P<version>\d+)\.cif\.gz$")
_HEAD_BYTES = 1 << 16  # the sequence sits in the first few kB of an AFDB CIF


def model_sequence(cif_text: str) -> str:
    """Return the canonical one-letter sequence of an AFDB mmCIF.

    Reads _entity_poly.pdbx_seq_one_letter_code_can, either inline on the
    key line or as a ';'-delimited multi-line block. Raises ValueError if the
    key is absent or the block is not closed.
    """
    lines = cif_text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(_SEQ_KEY):
            continue
        inline = line[len(_SEQ_KEY):].strip()
        if inline:
            return inline.strip("'\"")
        if i + 1 >= len(lines) or not lines[i + 1].startswith(";"):
            raise ValueError(f"{_SEQ_KEY}: value is neither inline nor a ';' block")
        parts = [lines[i + 1][1:].strip()]
        for block_line in lines[i + 2:]:
            if block_line.startswith(";"):
                return "".join(parts)
            parts.append(block_line.strip())
        raise ValueError(f"{_SEQ_KEY}: ';' block is not closed")
    raise ValueError(f"{_SEQ_KEY} not found")


def _sequence_from_gz(data: bytes) -> str:
    """Decompress the head of a gzipped CIF and parse its sequence.

    Falls back to the whole file when the head does not contain the full
    sequence block.
    """
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as fh:
        head = fh.read(_HEAD_BYTES)
    try:
        return model_sequence(head)
    except ValueError:
        return model_sequence(gzip.decompress(data).decode("utf-8"))


def _write_atomic(dest: Path, data: bytes) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def _link_or_copy(src: Path, dest: Path) -> None:
    """Hard-link src to dest (same filesystem), copying if linking fails."""
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _pool_rows(input_path: Path) -> list[tuple[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as fh:
        return [
            (row["sequence_id"], row["protein_sequence"])
            for row in csv.DictReader(fh)
        ]


def _matches_any(model_seq: str, pool_seqs: list[str]) -> bool:
    return any(fetch_alphafold.find_match(s, model_seq) is not None for s in pool_seqs)


def resolve_rows(
    rows: list[tuple[str, str]],
    models: dict[str, tuple[str, int, str]],
) -> list[dict]:
    """Build one manifest row per pool row.

    models maps accession -> (model sequence, version, cif_file relative to
    the output directory). Accessions are tried in the order of the ID.
    """
    out = []
    for seq_id, pool_seq in rows:
        accessions = fetch_alphafold.parse_accessions(seq_id)
        row = {
            "sequence_id": seq_id,
            "accession": accessions[0] if accessions else "",
            "version": "",
            "start": "",
            "end": "",
            "status": "not_local",
            "cif_file": "",
        }
        for acc in accessions:
            if acc not in models:
                continue
            row["status"] = "no_match"
            model_seq, version, cif_file = models[acc]
            match = fetch_alphafold.find_match(pool_seq, model_seq)
            if match is None:
                continue
            row |= {
                "accession": acc,
                "version": version,
                "start": match[0],
                "end": match[1],
                "status": "ok",
                "cif_file": cif_file,
            }
            break
        out.append(row)
    return out


def run(
    input_path: Path,
    tar_path: Path,
    tar_list: Path,
    rest_dir: Path,
    output_dir: Path,
) -> None:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = _pool_rows(input_path)
    seqs_by_acc: dict[str, list[str]] = defaultdict(list)
    for seq_id, pool_seq in rows:
        for acc in fetch_alphafold.parse_accessions(seq_id):
            seqs_by_acc[acc].append(pool_seq)

    in_tar: set[str] = set()
    with tar_list.open(encoding="utf-8") as fh:
        for line in fh:
            name_match = _NAME_RE.match(line.strip())
            if name_match and name_match["acc"] in seqs_by_acc:
                in_tar.add(line.strip())
    print(f"pool rows {len(rows)}, accessions {len(seqs_by_acc)}, in tar {len(in_tar)}")

    models: dict[str, tuple[str, int, str]] = {}
    n_read = 0
    with tarfile.open(tar_path, mode="r|") as tar:
        for member in tar:
            if member.name not in in_tar:
                continue
            acc = _NAME_RE.match(member.name)["acc"]
            data = tar.extractfile(member).read()
            model_seq = _sequence_from_gz(data)
            n_read += 1
            if n_read % 20000 == 0:
                print(f"  tar members read: {n_read}", flush=True)
            if not _matches_any(model_seq, seqs_by_acc[acc]):
                models[acc] = (model_seq, 0, "")
                continue
            dest = raw_dir / member.name
            if not dest.exists():
                _write_atomic(dest, data)
            version = int(_NAME_RE.match(member.name)["version"])
            models[acc] = (model_seq, version, f"raw/{member.name}")
    print(f"tar members read: {n_read}")

    # List the directory once; one glob per accession is slow on Lustre.
    rest_files: dict[str, Path] = {}
    for path in rest_dir.iterdir():
        name_match = _NAME_RE.match(path.name)
        if not name_match:
            continue
        prev = rest_files.get(name_match["acc"])
        version = int(name_match["version"])
        if prev is None or version > int(_NAME_RE.match(prev.name)["version"]):
            rest_files[name_match["acc"]] = path

    n_rest = 0
    for acc in seqs_by_acc:
        if acc in models or acc not in rest_files:
            continue
        src = rest_files[acc]
        model_seq = _sequence_from_gz(src.read_bytes())
        n_rest += 1
        if not _matches_any(model_seq, seqs_by_acc[acc]):
            models[acc] = (model_seq, 0, "")
            continue
        dest = raw_dir / src.name
        if not dest.exists():
            _link_or_copy(src, dest)
        version = int(_NAME_RE.match(src.name)["version"])
        models[acc] = (model_seq, version, f"raw/{src.name}")
    print(f"rest files read: {n_rest}")

    manifest_rows = resolve_rows(rows, models)
    manifest_path = output_dir / "fetch_manifest_local.csv"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fetch_alphafold._MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    os.replace(tmp, manifest_path)

    counts = Counter(r["status"] for r in manifest_rows)
    print(f"wrote {manifest_path}: {dict(counts)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match pool sequences to local AlphaFold DB files."
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CSV with sequence_id and protein_sequence columns.")
    parser.add_argument("--tar", required=True, type=str,
                        help="Path to swissprot_cif_v6.tar.")
    parser.add_argument("--tar_list", required=True, type=str,
                        help="Member list of the tar (output of tar -tf).")
    parser.add_argument("--rest_dir", required=True, type=str,
                        help="Directory of AF-<acc>-F1-model_v<n>.cif.gz files.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Root output directory for raw/ and the manifest.")
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(
        input_path=Path(_args.input),
        tar_path=Path(_args.tar),
        tar_list=Path(_args.tar_list),
        rest_dir=Path(_args.rest_dir),
        output_dir=Path(_args.output_dir),
    )
