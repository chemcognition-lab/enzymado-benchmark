"""Offline tests for match_afdb_local: CIF sequence parsing and row matching."""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path
import random
import sys
import tarfile

import pytest

_STRUCTURES_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "structures")
if _STRUCTURES_DIR not in sys.path:
    sys.path.insert(0, _STRUCTURES_DIR)

import match_afdb_local  # noqa: E402


def _cif(seq: str, width: int = 80) -> str:
    """Minimal AFDB-like CIF with the sequence as a ';' block."""
    block = "\n".join(seq[i:i + width] for i in range(0, len(seq), width))
    return (
        "data_test\n#\n"
        "_entity_poly.pdbx_seq_one_letter_code \n;" + block + "\n;\n"
        "_entity_poly.pdbx_seq_one_letter_code_can \n;" + block + "\n;\n"
        "_entity_poly.type polypeptide(L)\n#\n"
    )


def test_model_sequence_multiline() -> None:
    seq = "M" + "ACDEFGHIKL" * 20
    assert match_afdb_local.model_sequence(_cif(seq)) == seq


def test_model_sequence_inline() -> None:
    text = "_entity_poly.pdbx_seq_one_letter_code_can   MKV\n"
    assert match_afdb_local.model_sequence(text) == "MKV"


def test_model_sequence_missing_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        match_afdb_local.model_sequence("data_x\n#\n")


def test_model_sequence_unclosed_raises() -> None:
    text = "_entity_poly.pdbx_seq_one_letter_code_can \n;MKV\nACD\n"
    with pytest.raises(ValueError, match="not closed"):
        match_afdb_local.model_sequence(text)


def test_resolve_rows_order_and_status() -> None:
    models = {
        "P1": ("MAAAKKK", 6, "raw/AF-P1-F1-model_v6.cif.gz"),
        "P2": ("MCCCDDD", 6, "raw/AF-P2-F1-model_v6.cif.gz"),
        "P3": ("MEEE", 0, ""),
    }
    rows = [
        ("P1", "MAAAKKK"),          # full match
        ("P1__bi", "AAKK"),         # fragment, offset 2
        ("P3", "MQQQ"),             # model exists, no match
        ("P9", "MAAA"),             # no local model
        ("P3,P2", "CCDD"),          # first accession fails, second matches
    ]
    out = {r["sequence_id"]: r for r in match_afdb_local.resolve_rows(rows, models)}
    assert out["P1"]["status"] == "ok"
    assert (out["P1"]["start"], out["P1"]["end"]) == (0, 7)
    assert out["P1__bi"]["status"] == "ok"
    assert (out["P1__bi"]["start"], out["P1__bi"]["end"]) == (2, 6)
    assert out["P3"]["status"] == "no_match" and out["P3"]["cif_file"] == ""
    assert out["P9"]["status"] == "not_local"
    assert out["P3,P2"]["status"] == "ok" and out["P3,P2"]["accession"] == "P2"
    assert out["P3,P2"]["cif_file"] == "raw/AF-P2-F1-model_v6.cif.gz"


def test_run_end_to_end(tmp_path: Path) -> None:
    """Tar and rest sources: matched members are copied byte-for-byte."""
    rng = random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seq_tar = "M" + "".join(rng.choice(aa) for _ in range(120))  # not periodic
    seq_rest = "M" + "".join(rng.choice(aa) for _ in range(120))
    seq_unused = "M" + "G" * 120

    members = {
        "AF-A11111-F1-model_v6.cif.gz": gzip.compress(_cif(seq_tar).encode()),
        "AF-B22222-F1-model_v6.cif.gz": gzip.compress(_cif(seq_unused).encode()),
        "AF-Z99999-F1-model_v6.cif.gz": gzip.compress(_cif(seq_unused).encode()),
    }
    tar_path = tmp_path / "sp.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_list = tmp_path / "sp.list"
    tar_list.write_text("\n".join(members) + "\n")

    rest_dir = tmp_path / "rest"
    rest_dir.mkdir()
    rest_bytes = gzip.compress(_cif(seq_rest).encode())
    (rest_dir / "AF-C33333-F1-model_v6.cif.gz").write_bytes(rest_bytes)
    (rest_dir / "D44444.404").write_text("")

    pool = tmp_path / "pool.csv"
    with pool.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence_id", "protein_sequence"])
        writer.writerow(["A11111", seq_tar])
        writer.writerow(["A11111__as", seq_tar[10:50]])
        writer.writerow(["B22222", "MNOTTHERE"])
        writer.writerow(["C33333", seq_rest])
        writer.writerow(["D44444", "MKKK"])
        writer.writerow(["seq_no_accession", "MKKK"])

    out_dir = tmp_path / "out"
    match_afdb_local.run(pool, tar_path, tar_list, rest_dir, out_dir)

    with (out_dir / "fetch_manifest_local.csv").open() as fh:
        manifest = {r["sequence_id"]: r for r in csv.DictReader(fh)}
    assert manifest["A11111"]["status"] == "ok"
    assert manifest["A11111__as"]["start"] == "10"
    assert manifest["A11111__as"]["end"] == "50"
    assert manifest["B22222"]["status"] == "no_match"
    assert manifest["C33333"]["status"] == "ok"
    assert manifest["C33333"]["cif_file"] == "raw/AF-C33333-F1-model_v6.cif.gz"
    assert manifest["D44444"]["status"] == "not_local"
    assert manifest["seq_no_accession"]["status"] == "not_local"

    raw = out_dir / "raw"
    assert sorted(p.name for p in raw.iterdir()) == [
        "AF-A11111-F1-model_v6.cif.gz",
        "AF-C33333-F1-model_v6.cif.gz",
    ]
    assert (raw / "AF-A11111-F1-model_v6.cif.gz").read_bytes() == members[
        "AF-A11111-F1-model_v6.cif.gz"
    ]
    assert (raw / "AF-C33333-F1-model_v6.cif.gz").read_bytes() == rest_bytes


def _row(seq_id: str, status: str, cif: str = "") -> dict:
    return {"sequence_id": seq_id, "accession": seq_id.split(",")[0],
            "version": "6" if cif else "", "start": "0" if cif else "",
            "end": "3" if cif else "", "status": status, "cif_file": cif}


def test_merge_local_wins_and_api_fills_gaps() -> None:
    import merge_fetch_manifests  # noqa: PLC0415

    local = [
        _row("A", "ok", "raw/AF-A-F1-model_v6.cif.gz"),
        _row("B", "not_local"),
        _row("C,D", "ok", "raw/AF-C-F1-model_v6.cif.gz"),
        _row("E", "no_match"),
        _row("F", "not_local"),
        _row("G", "not_local"),
    ]
    api = [
        _row("B", "ok", "raw/AF-0000000365803487-model_v1.cif.gz"),
        _row("C,D", "ok", "raw/AF-D-F1-model_v6.cif.gz"),  # local already ok
        _row("E", "no_afdb"),
        _row("F", "no_afdb"),
    ]
    merged, new_api = merge_fetch_manifests.merge(local, api)
    by_id = {r["sequence_id"]: r for r in merged}
    assert [r["sequence_id"] for r in merged] == ["A", "B", "C,D", "E", "F", "G"]
    assert by_id["C,D"]["cif_file"] == "raw/AF-C-F1-model_v6.cif.gz"
    assert by_id["B"]["status"] == "ok"
    assert by_id["E"]["status"] == "no_match"
    assert by_id["F"]["status"] == "no_afdb"
    assert by_id["G"]["status"] == "not_local"
    assert [r["sequence_id"] for r in new_api] == ["B"]


def test_merge_rejects_error_rows() -> None:
    import merge_fetch_manifests  # noqa: PLC0415

    with pytest.raises(ValueError, match="error"):
        merge_fetch_manifests.merge([_row("B", "not_local")], [_row("B", "error")])
