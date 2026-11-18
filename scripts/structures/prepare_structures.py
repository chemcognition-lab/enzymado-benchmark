"""Prepare AlphaFold structures: trim, add hydrogens, minimize, save NPZ.

Reads fetch_manifest.csv rows with status='ok', trims each CIF to the
sequence range, adds missing atoms and hydrogens with amber14-all.xml, and
always minimizes hydrogen positions with heavy atoms fixed. Minimization is
mandatory: PDBFixer places every X-H bond near 1.19 A regardless of element
(measured median C-H 1.193, N-H 1.187, O-H 1.184, S-H 1.177); after
minimization these become C-H 1.090, N-H 1.012, O-H 0.967, S-H 1.340 with
heavy atoms moving exactly 0.0. Skipping minimization shifts embeddings by
more than any real difference between proteins. With the seed and one OpenMM
thread, two runs give identical positions: 0.0 A on 6 pilot proteins of 1,526
to 4,901 atoms (2026-11-10). With several threads they do not.

Example:
    python scripts/structures/prepare_structures.py \
        --fetch_manifest /scratch/.../data/structures/fetch_manifest.csv \
        --input data/processed/protein_pool.csv \
        --output_dir /scratch/.../data/structures
"""

import argparse
import csv
import gzip
import io
import os
from pathlib import Path
import random
import time

# OpenMM reads OPENMM_CPU_THREADS once, when it is imported, and
# Modeller.addHydrogens minimizes in its own Context. With several threads the
# hydrogens are not repeatable (0.0023 A between two seeded runs of a 3-residue
# chain, 2026-11-10; 0 A with one thread), so one thread is the default. Export
# the variable before running to override it.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import fetch_alphafold  # noqa: E402  same directory; on sys.path when run as script
import numpy as np  # noqa: E402
import openmm  # noqa: E402
import openmm.app as app  # noqa: E402
import openmm.unit as unit  # noqa: E402
import pandas as pd  # noqa: E402
import pdbfixer  # noqa: E402

_AA3TO1: dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

def topology_sequence(topology: app.Topology) -> str:
    """Return one-letter sequence from an OpenMM topology.

    Non-standard residues map to 'X' so the length always equals the number
    of residues (no residues are silently dropped).
    """
    return "".join(_AA3TO1.get(res.name, "X") for res in topology.residues())

def _load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def _load_pool_sequences(input_path: Path) -> dict[str, str]:
    pool = pd.read_csv(input_path)
    ids = pool["sequence_id"].astype(str)
    seqs = pool["protein_sequence"].astype(str)
    return dict(zip(ids, seqs))

def _cif_path_from_row(fetch_row: dict, raw_dir: Path) -> Path:
    """Return the CIF path for a fetch_manifest row.

    Uses the cif_file column (relative to output_dir) when present;
    falls back to raw/<accession>_v<version>.cif.gz for manifests written
    before the cif_file column was added (e.g. the pilot manifest).
    """
    cif_file = fetch_row.get("cif_file", "")
    if cif_file:
        return raw_dir.parent / cif_file
    return raw_dir / f"{fetch_row['accession']}_v{fetch_row['version']}.cif.gz"


def _load_done(manifest_path: Path) -> set[str]:
    """Return IDs already written, except status 'error', which is retried."""
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as fh:
        return {
            row["sequence_id"]
            for row in csv.DictReader(fh)
            if row.get("status", "") != "error"
        }

def _append_prepare_row(manifest_path: Path, row: dict) -> None:
    fieldnames = [
        "sequence_id",
        "n_atoms",
        "n_residues",
        "charge",
        "status",
        "seconds",
    ]
    new_file = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def _load_cif_text(cif_path: Path) -> str:
    """Read a gzip-compressed CIF file and return its text content."""
    with gzip.open(cif_path, "rt", encoding="utf-8") as gz:
        return gz.read()

def _fix_structure(
    cif_text: str, start: int, end: int, ph: float, *, seed: int = 0
) -> pdbfixer.PDBFixer:
    """Trim CIF to residues [start, end) and add missing atoms and hydrogens.

    Uses PDBFixer(pdbxfile=...) to avoid a temp file for the initial load.
    After trimming, bounces through PDB round-trip so PDBFixer can find
    missing atoms in the trimmed chain. seed is applied to addMissingAtoms
    and to Python's random before addMissingHydrogens (which calls
    random.random() for initial H placement).
    """
    fixer = pdbfixer.PDBFixer(pdbxfile=io.StringIO(cif_text))
    modeller = app.Modeller(fixer.topology, fixer.positions)
    atoms_to_delete = [
        atom
        for atom in modeller.topology.atoms()
        if atom.residue.index < start or atom.residue.index >= end
    ]
    modeller.delete(atoms_to_delete)

    pdb_io = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, pdb_io)
    pdb_io.seek(0)
    fixer2 = pdbfixer.PDBFixer(pdbfile=pdb_io)
    fixer2.findMissingResidues()
    fixer2.findMissingAtoms()
    fixer2.addMissingAtoms(seed=seed)
    random.seed(seed)  # addMissingHydrogens uses random.random() internally
    fixer2.addMissingHydrogens(ph)
    return fixer2

def _charge_from_system(system: openmm.System) -> int:
    """Sum partial charges from the NonbondedForce and round to integer."""
    nbforce = next(
        f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    charge_sum = sum(
        nbforce.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(system.getNumParticles())
    )
    return int(round(charge_sum))

def _minimize_h(
    topology: app.Topology,
    system: openmm.System,
    positions,
) -> np.ndarray:
    """Minimize hydrogen positions with heavy atoms fixed (mass=0).

    Masses must be set to 0 on the System BEFORE constructing the Simulation,
    because the Context caches masses at construction time and will not see
    later calls to setParticleMass. Build atom_list once to avoid the O(N^2)
    list(topology.atoms())[i] antipattern.
    """
    atom_list = list(topology.atoms())  # build once
    for i, atom in enumerate(atom_list):
        if atom.element is not None and atom.element.symbol != "H":
            system.setParticleMass(i, 0.0)

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    return state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

def _make_arrays(
    topology: app.Topology,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (numbers, residue_index, n_residues) from an OpenMM topology."""
    numbers = np.array(
        [
            atom.element.atomic_number if atom.element is not None else 0
            for atom in topology.atoms()
        ],
        dtype=np.int32,
    )
    residue_index = np.array(
        [atom.residue.index for atom in topology.atoms()],
        dtype=np.int32,
    )
    n_residues = sum(1 for _ in topology.residues())
    return numbers, residue_index, n_residues

def _prepare_one(
    seq_id: str,
    cif_path: Path,
    start: int,
    end: int,
    expected_seq: str,
    output_dir: Path,
    *,
    ph: float,
    write_pdb: bool,
    seed: int = 0,
) -> dict:
    t0 = time.time()
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    npz_path = prepared_dir / f"{seq_id}.npz"

    cif_text = _load_cif_text(cif_path)
    fixer2 = _fix_structure(cif_text, start, end, ph, seed=seed)

    structure_seq = topology_sequence(fixer2.topology)
    if structure_seq != expected_seq:
        return {
            "sequence_id": seq_id,
            "n_atoms": 0,
            "n_residues": 0,
            "charge": 0,
            "status": "mismatch",
            "seconds": round(time.time() - t0, 2),
        }

    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(fixer2.topology, nonbondedMethod=app.NoCutoff)
    total_charge = _charge_from_system(system)
    positions_ang = _minimize_h(fixer2.topology, system, fixer2.positions)

    numbers, residue_index, n_residues = _make_arrays(fixer2.topology)
    n_atoms = len(numbers)

    plddt = fetch_alphafold.read_plddt(cif_text)[start:end]
    if len(plddt) != n_residues:
        raise ValueError(
            f"{seq_id}: read_plddt returned {len(plddt)} values for"
            f" residues [{start}, {end}), expected {n_residues}"
        )

    np.savez_compressed(
        npz_path,
        numbers=numbers,
        positions=positions_ang.astype(np.float32),
        residue_index=residue_index,
        plddt=plddt.astype(np.float32),
        charge=np.int32(total_charge),
        sequence=np.str_(expected_seq),
    )

    if write_pdb:
        pdb_gz_path = prepared_dir / f"{seq_id}.pdb.gz"
        pdb_out = io.StringIO()
        app.PDBFile.writeFile(
            fixer2.topology, positions_ang * unit.angstrom, pdb_out
        )
        with gzip.open(pdb_gz_path, "wt") as gz_out:
            gz_out.write(pdb_out.getvalue())

    return {
        "sequence_id": seq_id,
        "n_atoms": n_atoms,
        "n_residues": n_residues,
        "charge": total_charge,
        "status": "ok",
        "seconds": round(time.time() - t0, 2),
    }

def run(
    fetch_manifest: Path,
    input_path: Path,
    output_dir: Path,
    *,
    ph: float,
    write_pdb: bool,
    start_idx: int,
    stop_idx: int,
    seed: int = 0,
    prepare_manifest: Path | None = None,
) -> None:
    """Prepare every ok row in the slice and append one manifest row each.

    A ValueError, KeyError, IndexError, StopIteration or OpenMMException
    inside one protein (what PDBFixer and OpenMM raise on a structure they
    cannot handle) is recorded as status 'error' with the message printed,
    and the run continues. Any other exception stops it. Error rows are
    retried on the next run.
    """
    raw_dir = output_dir / "raw"
    if prepare_manifest is None:
        prepare_manifest = output_dir / "prepare_manifest.csv"
    done = _load_done(prepare_manifest)
    sequences = _load_pool_sequences(input_path)
    rows = _load_manifest(fetch_manifest)
    ok_rows = [r for r in rows if r["status"] == "ok"]

    if stop_idx > 0:
        ok_rows = ok_rows[start_idx:stop_idx]
    elif start_idx > 0:
        ok_rows = ok_rows[start_idx:]

    for fetch_row in ok_rows:
        seq_id = fetch_row["sequence_id"]
        if seq_id in done:
            continue
        cif_path = _cif_path_from_row(fetch_row, raw_dir)
        expected_seq = sequences[seq_id]  # KeyError signals a bad manifest row
        t0 = time.time()
        try:
            result = _prepare_one(
                seq_id=seq_id,
                cif_path=cif_path,
                start=int(fetch_row["start"]),
                end=int(fetch_row["end"]),
                expected_seq=expected_seq,
                output_dir=output_dir,
                ph=ph,
                write_pdb=write_pdb,
                seed=seed,
            )
        except (
            ValueError,
            KeyError,
            IndexError,
            StopIteration,
            openmm.OpenMMException,
        ) as exc:  # one bad structure must not stop the slice
            print(f"{seq_id}: ERROR {type(exc).__name__}: {exc}")
            result = {
                "sequence_id": seq_id,
                "n_atoms": 0,
                "n_residues": 0,
                "charge": 0,
                "status": "error",
                "seconds": round(time.time() - t0, 2),
            }
        _append_prepare_row(prepare_manifest, result)
        print(
            f"{seq_id}: {result['status']}"
            f"  atoms={result['n_atoms']}"
            f"  charge={result['charge']}"
            f"  {result['seconds']}s"
        )

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trim, protonate, minimize hydrogens, and prepare AlphaFold CIF structures."
        )
    )
    parser.add_argument(
        "--fetch_manifest",
        required=True,
        type=str,
        help="Path to fetch_manifest.csv produced by fetch_alphafold.py.",
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
        help="Root output directory. prepared/ and prepare_manifest.csv go here.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for adding hydrogens (default 7.0).",
    )
    parser.add_argument(
        "--write_pdb",
        action="store_true",
        default=False,
        help="Also write prepared/<sequence_id>.pdb.gz for manual inspection.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First row index (0-based) of ok rows to process (default 0).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=0,
        help="One-past-last row index; 0 means process all (default 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for hydrogen placement (default 0).",
    )
    parser.add_argument(
        "--prepare_manifest",
        type=str,
        default=None,
        help=(
            "Manifest to append to (default <output_dir>/prepare_manifest.csv). "
            "Give each parallel job its own file."
        ),
    )
    return parser

if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(
        fetch_manifest=Path(_args.fetch_manifest),
        input_path=Path(_args.input),
        output_dir=Path(_args.output_dir),
        ph=_args.ph,
        write_pdb=_args.write_pdb,
        start_idx=_args.start,
        stop_idx=_args.stop,
        seed=_args.seed,
        prepare_manifest=(
            Path(_args.prepare_manifest) if _args.prepare_manifest else None
        ),
    )
