"""Offline tests for make_embedding_slices."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest

_STRUCTURES_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "structures")
if _STRUCTURES_DIR not in sys.path:
    sys.path.insert(0, _STRUCTURES_DIR)

import make_embedding_slices  # noqa: E402

_HEADER = "sequence_id,n_atoms,n_residues,charge,status,seconds\n"


def test_cut_slices_covers_all_sorted_and_balanced() -> None:
    rng = random.Random(0)
    rows = [(f"p{i}", rng.randint(500, 40000)) for i in range(5000)]
    slices = make_embedding_slices.cut_slices(rows, n_slices=20, overhead_atoms=2000)
    flat = [r for piece in slices for r in piece]
    assert sorted(flat) == sorted(rows)
    sizes = [n for _, n in flat]
    assert sizes == sorted(sizes)  # contiguous in atom order
    costs = [sum(n + 2000 for _, n in piece) for piece in slices]
    assert max(costs) / min(costs) < 1.1


def test_load_ok_rows_retry_and_duplicates(tmp_path: Path) -> None:
    (tmp_path / "local_0000.csv").write_text(
        _HEADER
        + "a,100,5,0,ok,1\nb,0,0,0,error,1\nb,120,6,0,ok,1\nc,0,0,0,mismatch,1\n"
    )
    (tmp_path / "local_0001.csv").write_text(_HEADER + "d,300,9,1,ok,1\n")
    rows = dict(make_embedding_slices.load_ok_rows(tmp_path))
    assert rows == {"a": 100, "b": 120, "d": 300}

    (tmp_path / "api_0000.csv").write_text(_HEADER + "d,300,9,1,ok,1\n")
    with pytest.raises(ValueError, match="is ok in"):
        make_embedding_slices.load_ok_rows(tmp_path)


def test_run_only_ids_keeps_listed_ok_rows(tmp_path: Path) -> None:
    manifests = tmp_path / "m"
    manifests.mkdir()
    (manifests / "api_0000.csv").write_text(
        _HEADER + "a,100,5,0,ok,1\nb,0,0,0,error,1\nb,120,6,0,ok,1\nc,200,7,0,ok,1\n"
    )
    wanted = tmp_path / "ids.txt"
    wanted.write_text("b\nz\n")
    out = tmp_path / "s"
    make_embedding_slices.run(manifests, 1, 2000, out, "api_*.csv", wanted)
    assert (out / "slice_0000.txt").read_text() == "b\n"
