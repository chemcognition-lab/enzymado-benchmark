"""Pure-logic tests for fetch_alphafold and prepare_structures helpers."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import sys
from unittest import mock
import urllib.error

import numpy as np
import pytest

# fetch_alphafold lives under scripts/structures/, which is not a package.
_STRUCTURES_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "structures")
if _STRUCTURES_DIR not in sys.path:
    sys.path.insert(0, _STRUCTURES_DIR)

import fetch_alphafold  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers shared by retry/error tests
# ---------------------------------------------------------------------------

def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    """Build a urllib.error.HTTPError with an optional Retry-After header."""
    hdrs = mock.MagicMock()
    hdrs.get.side_effect = lambda key, default=None: (
        retry_after if key == "Retry-After" else default
    )
    return urllib.error.HTTPError("http://test", code, "err", hdrs, None)


def _success_ctx(payload: bytes) -> mock.MagicMock:
    """Return a context-manager mock that yields a readable response."""
    resp = mock.MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = mock.MagicMock(return_value=False)
    resp.read.return_value = payload
    return resp


_CIF = b"data_AF-TEST\n#\n_entity_poly.pdbx_seq_one_letter_code_can MKV\n"


@pytest.mark.parametrize("payload", [_CIF, gzip.compress(_CIF)])
def test_write_cif_gz_compresses_exactly_once(tmp_path: Path, payload: bytes) -> None:
    """Plain and already-gzipped server bytes both end up gzipped once."""
    dest = tmp_path / "AF-TEST-model_v1.cif.gz"
    fetch_alphafold._write_cif_gz(payload, dest)
    with gzip.open(dest, "rb") as gz:
        assert gz.read() == _CIF
    assert not (tmp_path / "AF-TEST-model_v1.cif.gz.tmp").exists()


@pytest.mark.parametrize("payload", [b"<html>503</html>", gzip.compress(_CIF)[:20]])
def test_write_cif_gz_rejects_bad_bytes(tmp_path: Path, payload: bytes) -> None:
    """Non-CIF or truncated downloads raise RuntimeError and leave no file."""
    dest = tmp_path / "AF-TEST-model_v1.cif.gz"
    with pytest.raises(RuntimeError):
        fetch_alphafold._write_cif_gz(payload, dest)
    assert list(tmp_path.iterdir()) == []


def test_parse_accessions_single_with_suffix() -> None:
    """'A0A009IHW8__bi' extracts one accession."""
    result = fetch_alphafold.parse_accessions("A0A009IHW8__bi")
    assert result == ["A0A009IHW8"]

def test_parse_accessions_comma_separated() -> None:
    """'A0A097J792,Q9WTK1' extracts two accessions."""
    result = fetch_alphafold.parse_accessions("A0A097J792,Q9WTK1")
    assert result == ["A0A097J792", "Q9WTK1"]

def test_parse_accessions_comma_and_suffix() -> None:
    """'A0A097J792,Q9WTK1__bi' still splits on comma, then strips suffix."""
    # sequence_id.split("__")[0] = "A0A097J792,Q9WTK1"
    result = fetch_alphafold.parse_accessions("A0A097J792,Q9WTK1__bi")
    assert result == ["A0A097J792", "Q9WTK1"]

def test_parse_accessions_plain_id() -> None:
    """'P00533' with no suffix or comma returns a single-element list."""
    result = fetch_alphafold.parse_accessions("P00533")
    assert result == ["P00533"]

def test_find_match_exact_full_sequence() -> None:
    """When pool_seq equals afdb_seq the offset is (0, len)."""
    seq = "ACDEFGHIKLM"
    start, end = fetch_alphafold.find_match(seq, seq)  # type: ignore[misc]
    assert start == 0
    assert end == len(seq)

def test_find_match_subsequence_offset() -> None:
    """Pool sequence found in the middle returns the correct offset."""
    afdb_seq = "AAACDEFGHIKLMVVV"
    pool_seq = "CDEFGHIKLM"
    result = fetch_alphafold.find_match(pool_seq, afdb_seq)
    assert result is not None
    start, end = result
    assert afdb_seq[start:end] == pool_seq
    assert start == 3
    assert end == 13

def test_find_match_no_match_returns_none() -> None:
    """Returns None when pool_seq is not a substring of afdb_seq."""
    result = fetch_alphafold.find_match("ACDE", "GHIKLM")
    assert result is None

def test_find_match_mutation_gives_no_match() -> None:
    """A single amino-acid substitution breaks the match."""
    afdb_seq = "ACDEFGHIKLM"
    mutant = "ACDEFGHIKLX"  # X is not in afdb_seq
    result = fetch_alphafold.find_match(mutant, afdb_seq)
    assert result is None

def test_find_match_offset_correctness() -> None:
    """start and end are base-0 residue indices with end exclusive."""
    afdb_seq = "MAST" + "TARGETSEQ" + "TAIL"
    pool_seq = "TARGETSEQ"
    result = fetch_alphafold.find_match(pool_seq, afdb_seq)
    assert result is not None
    start, end = result
    assert start == 4
    assert end == 13
    assert afdb_seq[start:end] == pool_seq

# Minimal _atom_site loop with three columns in a compact CIF snippet.
# Two residues: residue 1 (pLDDT 72.5) and residue 2 (pLDDT 68.0).
# Non-CA atoms (N, C) must not contribute to the result.
_MINI_CIF = """\
loop_
_atom_site.label_atom_id
_atom_site.label_seq_id
_atom_site.B_iso_or_equiv
N 1 70.00
CA 1 72.50
C 1 71.00
N 2 65.00
CA 2 68.00
C 2 66.00
#
"""

def test_read_plddt_extracts_ca_bfactors() -> None:
    """read_plddt returns one value per residue from CA B_iso_or_equiv."""
    plddt = fetch_alphafold.read_plddt(_MINI_CIF)
    assert len(plddt) == 2
    assert float(plddt[0]) == pytest.approx(72.5)
    assert float(plddt[1]) == pytest.approx(68.0)

def test_read_plddt_dtype_is_float32() -> None:
    """Output dtype is float32 for compact storage."""
    plddt = fetch_alphafold.read_plddt(_MINI_CIF)
    assert plddt.dtype == np.float32

def test_read_plddt_ignores_non_ca() -> None:
    """N and C atoms do not affect the residue pLDDT values."""
    plddt = fetch_alphafold.read_plddt(_MINI_CIF)
    # Residue 1 has N=70.0, CA=72.5, C=71.0; only CA should be used.
    assert float(plddt[0]) == pytest.approx(72.5)

def test_read_plddt_missing_column_raises() -> None:
    """Raises ValueError naming the missing column when columns are absent."""
    bad_cif = "loop_\n_atom_site.id\n_atom_site.type_symbol\n1 N\n"
    with pytest.raises(ValueError, match="_atom_site"):
        fetch_alphafold.read_plddt(bad_cif)

def test_topology_sequence_skipped_without_openmm() -> None:
    """topology_sequence exists but requires openmm; skip gracefully."""
    pytest.importorskip("openmm")
    import prepare_structures  # noqa: F401, PLC0415

    # If openmm is available, verify the AA map contains all 20 standard AAs.
    aas = set("ACDEFGHIKLMNPQRSTVWY")
    assert aas == set(prepare_structures._AA3TO1.values())

# Minimal mmCIF for a 3-residue ALA-ALA-ALA tripeptide (backbone + CB).
# Heavy atoms only; PDBFixer will add hydrogens via addMissingHydrogens.
# Columns required by OpenMM PDBxFile: id, type_symbol, label_atom_id,
# label_alt_id, label_comp_id, label_asym_id, label_seq_id, Cartn_x/y/z.
# B_iso_or_equiv provides pLDDT values per residue for read_plddt tests.

_TINY_CIF = """\
data_TEST
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
1 N N . ALA A 1 0.000 2.000 0.000 70.00
2 C CA . ALA A 1 1.459 2.000 0.000 70.00
3 C C . ALA A 1 2.009 3.421 0.000 70.00
4 O O . ALA A 1 1.270 4.388 0.000 70.00
5 C CB . ALA A 1 1.915 1.224 -1.238 70.00
6 N N . ALA A 2 3.325 3.595 0.000 68.00
7 C CA . ALA A 2 3.909 4.939 0.000 68.00
8 C C . ALA A 2 5.403 4.924 0.000 68.00
9 O O . ALA A 2 6.025 3.858 0.000 68.00
10 C CB . ALA A 2 3.473 5.699 -1.238 68.00
11 N N . ALA A 3 6.090 5.996 0.000 65.00
12 C CA . ALA A 3 7.519 6.057 0.000 65.00
13 C C . ALA A 3 8.085 7.457 0.000 65.00
14 O O . ALA A 3 7.350 8.397 0.000 65.00
15 C CB . ALA A 3 7.966 5.254 -1.212 65.00
#
"""

def test_fix_structure_crop_keeps_expected_residues() -> None:
    """_fix_structure([start=1, end=3)) keeps exactly 2 of the 3 residues."""
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    import prepare_structures  # noqa: PLC0415

    fixer = prepare_structures._fix_structure(_TINY_CIF, 1, 3, 7.0, seed=0)
    n_res = sum(1 for _ in fixer.topology.residues())
    assert n_res == 2, (
        f"Expected 2 residues after crop [1, 3) but got {n_res}"
    )

def test_minimize_h_heavy_atoms_do_not_move() -> None:
    """After _minimize_h, heavy atom positions are unchanged within 1e-3 A.

    This test fails if masses are set after the Simulation is constructed
    (heavy atoms gain mobility) or if _minimize_h is not called at all
    (the positions_ang would match fixer2.positions, which is true, but
    the C-H bond test below would catch that).
    """
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    import openmm.app as _app
    import openmm.unit as _unit
    import prepare_structures  # noqa: PLC0415

    fixer = prepare_structures._fix_structure(_TINY_CIF, 0, 3, 7.0, seed=0)
    ff = _app.ForceField("amber14-all.xml")
    system = ff.createSystem(fixer.topology, nonbondedMethod=_app.NoCutoff)

    # Record heavy atom positions from fixer (before minimization)
    atom_list = list(fixer.topology.atoms())
    heavy_idx = [
        i for i, a in enumerate(atom_list)
        if a.element is not None and a.element.symbol != "H"
    ]
    heavy_before = np.array(
        [list(fixer.positions[i].value_in_unit(_unit.angstrom)) for i in heavy_idx],
        dtype=np.float64,
    )

    positions_ang = prepare_structures._minimize_h(
        fixer.topology, system, fixer.positions
    )

    heavy_after = positions_ang[heavy_idx]
    max_move = float(np.abs(heavy_before - heavy_after).max())
    assert max_move < 1e-3, (
        f"Heavy atoms moved {max_move:.4f} A; "
        "masses must be set to 0 before Simulation is created"
    )

def test_minimize_h_ch_bonds_near_1p09() -> None:
    """After _minimize_h, median C-H bond length is within 0.03 A of 1.09.

    PDBFixer places X-H bonds near 1.19 A before minimization; after
    minimization with AMBER14 the C-H equilibrium is 1.09 A. This test
    fails if minimization is skipped (median stays near 1.19).
    """
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    import openmm.app as _app
    import prepare_structures  # noqa: PLC0415

    fixer = prepare_structures._fix_structure(_TINY_CIF, 0, 3, 7.0, seed=0)
    ff = _app.ForceField("amber14-all.xml")
    system = ff.createSystem(fixer.topology, nonbondedMethod=_app.NoCutoff)
    positions_ang = prepare_structures._minimize_h(
        fixer.topology, system, fixer.positions
    )

    ch_dists: list[float] = []
    for bond in fixer.topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if a1.element is None or a2.element is None:
            continue
        syms = {a1.element.symbol, a2.element.symbol}
        if syms == {"C", "H"}:
            diff = positions_ang[a1.index] - positions_ang[a2.index]
            ch_dists.append(float(np.linalg.norm(diff)))

    assert ch_dists, "No C-H bonds found; did addMissingHydrogens run?"
    median_ch = float(np.median(ch_dists))
    assert abs(median_ch - 1.09) < 0.03, (
        f"Median C-H bond {median_ch:.3f} A; expected near 1.09 A. "
        "Minimization may have been skipped."
    )


def test_prepare_one_end_to_end_write_pdb(tmp_path: Path) -> None:
    """_prepare_one writes matching .npz and .pdb.gz from a gzipped CIF.

    Asserts status ok, both output files exist, pLDDT matches the CIF
    B-factors, and PDB coordinates agree with .npz positions within 0.01 A.
    This test fails if write_pdb writes unminimized positions (W1 bug).
    """
    import gzip

    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    import openmm.app as _app
    import openmm.unit as _unit
    import prepare_structures  # noqa: PLC0415

    cif_gz = tmp_path / "AF-TEST_v4.cif.gz"
    with gzip.open(cif_gz, "wt") as fh:
        fh.write(_TINY_CIF)

    result = prepare_structures._prepare_one(
        seq_id="TEST",
        cif_path=cif_gz,
        start=0,
        end=3,
        expected_seq="AAA",
        output_dir=tmp_path / "output",
        ph=7.0,
        write_pdb=True,
        seed=0,
    )

    assert result["status"] == "ok", f"Unexpected status {result['status']!r}"

    prepared_dir = tmp_path / "output" / "prepared"
    npz_path = prepared_dir / "TEST.npz"
    pdb_gz_path = prepared_dir / "TEST.pdb.gz"
    assert npz_path.exists(), ".npz not written"
    assert pdb_gz_path.exists(), ".pdb.gz not written"

    data = np.load(npz_path, allow_pickle=False)

    expected_plddt = np.array([70.0, 68.0, 65.0], dtype=np.float32)
    np.testing.assert_allclose(
        data["plddt"], expected_plddt, atol=0.1,
        err_msg="pLDDT values must come from CIF B_iso_or_equiv"
    )

    with gzip.open(pdb_gz_path, "rt") as fh:
        pdb_obj = _app.PDBFile(fh)
    pdb_pos = np.array(
        [list(p.value_in_unit(_unit.angstrom)) for p in pdb_obj.getPositions()],
        dtype=np.float32,
    )
    max_diff = float(np.abs(pdb_pos - data["positions"]).max())
    assert max_diff < 0.01, (
        f"PDB and NPZ positions differ by {max_diff:.4f} A; "
        "write_pdb must use minimized positions, not fixer2.positions"
    )


def test_fix_structure_reproducible_with_seed() -> None:
    """Two _fix_structure + _minimize_h runs with seed=0 give identical positions.

    Measured: without seeding, two runs can differ by more than 0.1 A.
    With seed=0 and one OpenMM thread the max difference is 0.0. With several
    CPU threads it is not deterministic: 2 of 3 runs failed on 2026-11-10 with
    a 0.0023 A difference. OpenMM reads OPENMM_CPU_THREADS when it is imported,
    so tests/conftest.py sets it to 1 before any test imports openmm.
    """
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    import openmm.app as _app
    import prepare_structures  # noqa: PLC0415

    def run_once() -> np.ndarray:
        fixer = prepare_structures._fix_structure(_TINY_CIF, 0, 3, 7.0, seed=0)
        ff = _app.ForceField("amber14-all.xml")
        system = ff.createSystem(fixer.topology, nonbondedMethod=_app.NoCutoff)
        return prepare_structures._minimize_h(fixer.topology, system, fixer.positions)

    pos1 = run_once()
    pos2 = run_once()
    max_diff = float(np.abs(pos1 - pos2).max())
    assert max_diff < 1e-3, (
        f"Two runs with seed=0 differed by {max_diff:.4f} A; "
        "random.seed must be called before addMissingHydrogens"
    )


# ---------------------------------------------------------------------------
# Bug 1: HTTP error handling
# ---------------------------------------------------------------------------

def test_fetch_api_404_returns_empty_list() -> None:
    """A genuine 404 from the API returns [] (accession not in AFDB)."""
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(404)):
        result = fetch_alphafold._fetch_api("NOENTRY", {}, 0.0)
    assert result == []


def test_fetch_api_404_recorded_as_no_afdb(tmp_path: Path) -> None:
    """run() writes no_afdb when the API returns 404 for all accessions."""
    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text("sequence_id,protein_sequence\nNOENTRY,ACDEF\n")

    with mock.patch("urllib.request.urlopen", side_effect=_http_error(404)):
        fetch_alphafold.run(pool_csv, tmp_path, delay=0.0)

    import csv as _csv

    with (tmp_path / "fetch_manifest.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["status"] == "no_afdb"
    assert rows[0]["cif_file"] == ""


def test_request_bytes_429_retries_with_retry_after() -> None:
    """_request_bytes retries on 429 and uses Retry-After sleep duration."""
    err = _http_error(429, retry_after="7")
    success = _success_ctx(b"hello")

    with mock.patch("urllib.request.urlopen", side_effect=[err, success]) as _mock:
        with mock.patch("time.sleep") as mock_sleep:
            result = fetch_alphafold._request_bytes("http://test", timeout=10)

    assert result == b"hello"
    mock_sleep.assert_called_once_with(7.0)  # used Retry-After, not _BASE_BACKOFF


def test_request_bytes_429_no_retry_after_uses_backoff() -> None:
    """_request_bytes uses _BASE_BACKOFF when Retry-After is absent."""
    err = _http_error(429, retry_after=None)
    success = _success_ctx(b"data")

    with mock.patch("urllib.request.urlopen", side_effect=[err, success]):
        with mock.patch("time.sleep") as mock_sleep:
            result = fetch_alphafold._request_bytes("http://test", timeout=10)

    assert result == b"data"
    mock_sleep.assert_called_once_with(fetch_alphafold._BASE_BACKOFF)


def test_request_bytes_503_retries() -> None:
    """_request_bytes retries on 503 Service Unavailable."""
    err = _http_error(503, retry_after=None)
    success = _success_ctx(b"ok")

    with mock.patch("urllib.request.urlopen", side_effect=[err, err, success]):
        with mock.patch("time.sleep"):
            result = fetch_alphafold._request_bytes("http://test", timeout=10)

    assert result == b"ok"


def test_persistent_429_raises_not_no_afdb(tmp_path: Path) -> None:
    """run() writes status='error' (not 'no_afdb') when retries are exhausted."""
    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text("sequence_id,protein_sequence\nRATELIM,ACDEF\n")

    # Always raise 429, more times than _MAX_RETRIES + 1 attempts
    n_errors = fetch_alphafold._MAX_RETRIES + 10
    side_effects = [_http_error(429, retry_after="0")] * n_errors

    with mock.patch("urllib.request.urlopen", side_effect=side_effects):
        with mock.patch("time.sleep"):
            fetch_alphafold.run(pool_csv, tmp_path, delay=0.0)

    import csv as _csv

    with (tmp_path / "fetch_manifest.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["status"] == "error", (
        f"Expected 'error', got '{rows[0]['status']}'. "
        "Persistent 429 must NOT silently become no_afdb."
    )


def test_error_row_retried_on_resume(tmp_path: Path) -> None:
    """An 'error' row in the manifest is retried; a finalized row is skipped."""
    import csv as _csv

    manifest = tmp_path / "fetch_manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = _csv.DictWriter(
            fh,
            fieldnames=fetch_alphafold._MANIFEST_FIELDS,
            lineterminator="\n",
        )
        w.writeheader()
        w.writerow(
            {
                "sequence_id": "PREV_OK",
                "accession": "PREV_OK",
                "version": "1",
                "start": "0",
                "end": "5",
                "status": "ok",
                "cif_file": "raw/dummy.cif.gz",
            }
        )
        w.writerow(
            {
                "sequence_id": "PREV_ERR",
                "accession": "PREV_ERR",
                "version": "",
                "start": "",
                "end": "",
                "status": "error",
                "cif_file": "",
            }
        )

    done = fetch_alphafold._load_done(manifest)
    assert "PREV_OK" in done
    assert "PREV_ERR" not in done  # error rows are not considered done


# ---------------------------------------------------------------------------
# Bug 2: polyprotein distinct filenames
# ---------------------------------------------------------------------------

_POLY_ENTRY_A = {
    "sequence": "AAACCC",
    "latestVersion": 1,
    "modelEntityId": "AF-POLY-MODEL1",
    "cifUrl": "http://fake/AF-POLY-MODEL1-model_v1.cif",
}
_POLY_ENTRY_B = {
    "sequence": "DDDGGG",
    "latestVersion": 1,
    "modelEntityId": "AF-POLY-MODEL2",
    "cifUrl": "http://fake/AF-POLY-MODEL2-model_v1.cif",
}


def test_resolve_match_distinct_files_for_same_version(tmp_path: Path) -> None:
    """Two sequences from the same accession+version get different cif_file names."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session: dict = {"POLY": [_POLY_ENTRY_A, _POLY_ENTRY_B]}

    # Mock download so no network needed; write a minimal gzip payload.
    def fake_download(url: str, dest: Path) -> None:
        with gzip.open(dest, "wb") as gz:
            gz.write(b"FAKE")

    with mock.patch.object(fetch_alphafold, "_download_cif", side_effect=fake_download):
        row_a = fetch_alphafold._resolve_match(
            "AAACCC", ["POLY"], session, 0.0, raw_dir
        )
        row_b = fetch_alphafold._resolve_match(
            "DDDGGG", ["POLY"], session, 0.0, raw_dir
        )

    assert row_a is not None and row_b is not None
    assert row_a["cif_file"] != row_b["cif_file"], (
        "Two models with the same accession and version must get distinct cif_file."
    )
    assert "AF-POLY-MODEL1" in row_a["cif_file"]
    assert "AF-POLY-MODEL2" in row_b["cif_file"]


def test_resolve_match_each_sequence_paired_with_own_cif(tmp_path: Path) -> None:
    """Each sequence is paired with the CIF of the model whose sequence matched."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session: dict = {"POLY": [_POLY_ENTRY_A, _POLY_ENTRY_B]}

    downloaded: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        downloaded.append(str(dest.name))
        with gzip.open(dest, "wb") as gz:
            gz.write(b"FAKE")

    with mock.patch.object(fetch_alphafold, "_download_cif", side_effect=fake_download):
        row_a = fetch_alphafold._resolve_match(
            "AAACCC", ["POLY"], session, 0.0, raw_dir
        )
        row_b = fetch_alphafold._resolve_match(
            "DDDGGG", ["POLY"], session, 0.0, raw_dir
        )

    assert row_a is not None
    assert "AF-POLY-MODEL1" in row_a["cif_file"]
    assert row_b is not None
    assert "AF-POLY-MODEL2" in row_b["cif_file"]
    # Both downloads happened
    assert len(downloaded) == 2


def test_run_two_chains_write_distinct_cif_files(tmp_path: Path) -> None:
    """run() writes distinct cif_file entries for two chains of a polyprotein."""
    import csv as _csv

    pool_csv = tmp_path / "pool.csv"
    pool_csv.write_text(
        "sequence_id,protein_sequence\n"
        "POLY__chain1,AAACCC\n"
        "POLY__chain2,DDDGGG\n"
    )

    api_payload = json.dumps([_POLY_ENTRY_A, _POLY_ENTRY_B]).encode()

    def fake_download(url: str, dest: Path) -> None:
        with gzip.open(dest, "wb") as gz:
            gz.write(b"FAKE")

    with mock.patch("urllib.request.urlopen", return_value=_success_ctx(api_payload)):
        with mock.patch.object(
            fetch_alphafold, "_download_cif", side_effect=fake_download
        ):
            fetch_alphafold.run(pool_csv, tmp_path, delay=0.0)

    with (tmp_path / "fetch_manifest.csv").open() as fh:
        rows = list(_csv.DictReader(fh))

    assert len(rows) == 2
    cif_files = [r["cif_file"] for r in rows]
    assert cif_files[0] != cif_files[1], (
        "Two polyprotein chains must map to distinct CIF files."
    )
    assert all(r["status"] == "ok" for r in rows)


def test_cif_path_from_row_uses_cif_file_column(tmp_path: Path) -> None:
    """_cif_path_from_row uses cif_file column when present."""
    pytest.importorskip("openmm")
    import prepare_structures  # noqa: PLC0415

    raw_dir = tmp_path / "raw"
    row = {
        "accession": "P12345",
        "version": "4",
        "cif_file": "raw/AF-P12345-F1-model_v4.cif.gz",
    }
    path = prepare_structures._cif_path_from_row(row, raw_dir)
    assert path == tmp_path / "raw" / "AF-P12345-F1-model_v4.cif.gz"


def test_cif_path_from_row_fallback_for_old_manifest(tmp_path: Path) -> None:
    """_cif_path_from_row falls back to old naming when cif_file is absent."""
    pytest.importorskip("openmm")
    import prepare_structures  # noqa: PLC0415

    raw_dir = tmp_path / "raw"
    row = {"accession": "P12345", "version": "4"}  # no cif_file key
    path = prepare_structures._cif_path_from_row(row, raw_dir)
    assert path == raw_dir / "P12345_v4.cif.gz"


def test_cif_path_from_row_fallback_empty_cif_file(tmp_path: Path) -> None:
    """_cif_path_from_row falls back to old naming when cif_file is empty string."""
    pytest.importorskip("openmm")
    import prepare_structures  # noqa: PLC0415

    raw_dir = tmp_path / "raw"
    row = {"accession": "P12345", "version": "4", "cif_file": ""}
    path = prepare_structures._cif_path_from_row(row, raw_dir)
    assert path == raw_dir / "P12345_v4.cif.gz"


def test_prepare_error_is_recorded_and_retried(tmp_path: Path) -> None:
    """A structure PDBFixer cannot handle becomes an 'error' row, not a crash.

    The other row in the slice is still processed, the manifest goes to the
    path given, and the error row is retried on the next run.
    """
    pytest.importorskip("openmm")
    import prepare_structures  # noqa: PLC0415

    pool = tmp_path / "pool.csv"
    pool.write_text("sequence_id,protein_sequence\nbad,MKV\ngood,MKV\n")
    fetch = tmp_path / "fetch_manifest.csv"
    fetch.write_text(
        "sequence_id,accession,version,start,end,status,cif_file\n"
        "bad,P1,6,0,3,ok,raw/a.cif.gz\n"
        "good,P2,6,0,3,ok,raw/b.cif.gz\n"
    )
    calls: list[str] = []

    def fake_prepare(seq_id: str, **_: object) -> dict:
        calls.append(seq_id)
        if seq_id == "bad":
            raise ValueError("No template found for residue 1 (XYZ)")
        return {"sequence_id": seq_id, "n_atoms": 10, "n_residues": 3,
                "charge": 0, "status": "ok", "seconds": 0.1}

    manifest = tmp_path / "manifests" / "task_000.csv"
    manifest.parent.mkdir()
    kwargs = dict(fetch_manifest=fetch, input_path=pool, output_dir=tmp_path,
                  ph=7.0, write_pdb=False, start_idx=0, stop_idx=0,
                  prepare_manifest=manifest)
    patch = mock.patch.object(
        prepare_structures, "_prepare_one", side_effect=fake_prepare
    )
    with patch:
        prepare_structures.run(**kwargs)
        assert calls == ["bad", "good"]
        rows = list(csv.DictReader(manifest.open()))
        assert [(r["sequence_id"], r["status"]) for r in rows] == [
            ("bad", "error"), ("good", "ok")]
        assert not (tmp_path / "prepare_manifest.csv").exists()

        calls.clear()
        prepare_structures.run(**kwargs)
        assert calls == ["bad"]  # error retried, ok skipped
