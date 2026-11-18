"""Integration tests for _descriptors_no_force in mace_embedding_script.

Skipped automatically when mace is not installed or MACE_TEST_MODEL is unset.
Imports mace before anything that could pull e3nn, since importing e3nn first
causes a weights-only load failure under torch 2.7.

Set MACE_TEST_MODEL to a local path or recognised size string ('small',
'medium', 'large') to run against a real checkpoint:

    MACE_TEST_MODEL=/path/to/model.pt pytest tests/test_mace_embedding.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

# mace must be imported before any library that could pull e3nn.
# pytest.importorskip skips this module at collection time if mace is absent.
pytest.importorskip("mace")

import numpy as np  # noqa: E402

_EMBEDDING_SCRIPTS = str(Path(__file__).resolve().parents[1] / "embedding_scripts")
if _EMBEDDING_SCRIPTS not in sys.path:
    sys.path.insert(0, _EMBEDDING_SCRIPTS)

import mace_embedding_script  # noqa: E402


@pytest.fixture(scope="module")
def calc():
    """Load a MACE calculator from MACE_TEST_MODEL; skip if unset."""
    model_path = os.environ.get("MACE_TEST_MODEL", "")
    if not model_path:
        pytest.skip("MACE_TEST_MODEL env var not set; skipping real-weight tests")
    from mace.calculators.foundations_models import mace_off

    return mace_off(model=model_path, device="cpu", default_dtype="float32")

@pytest.fixture(scope="module")
def methane_atoms():
    """Return an ASE Atoms object for a tetrahedral CH4 molecule."""
    from ase import Atoms

    # Tetrahedral CH4 with C at origin and H at standard bond-angle positions
    positions = np.array(
        [
            [0.000, 0.000, 0.000],    # C
            [0.629, 0.629, 0.629],    # H
            [-0.629, -0.629, 0.629],  # H
            [-0.629, 0.629, -0.629],  # H
            [0.629, -0.629, -0.629],  # H
        ],
        dtype=np.float64,
    )
    return Atoms(numbers=[6, 1, 1, 1, 1], positions=positions, pbc=False)

def test_no_force_route_matches_get_descriptors(calc, methane_atoms) -> None:
    """_descriptors_no_force agrees with calc.get_descriptors within 1e-4.

    The reference is calc.get_descriptors(atoms, invariants_only=True).
    The pilot measurement on MACE-OFF medium showed max abs diff 2.4e-6.
    """
    ref = calc.get_descriptors(methane_atoms, invariants_only=True)
    result = mace_embedding_script._descriptors_no_force(calc, methane_atoms)
    assert result.shape == ref.shape
    max_diff = float(np.abs(result - ref).max())
    assert max_diff < 1e-4, f"Max abs diff {max_diff:.2e} exceeds 1e-4 threshold"

def test_descriptors_rotation_invariant(calc, methane_atoms) -> None:
    """Descriptor norms are unchanged under a rigid rotation within 1e-3.

    Uses a 90-degree rotation around the z-axis. Invariant descriptors
    (l=0 components only) must be identical regardless of orientation.
    """
    from ase import Atoms

    # 90-degree rotation around z: (x, y, z) -> (-y, x, z)
    pos = methane_atoms.get_positions()
    pos_rot = np.column_stack([-pos[:, 1], pos[:, 0], pos[:, 2]])
    rotated = Atoms(
        numbers=methane_atoms.get_atomic_numbers(),
        positions=pos_rot,
        pbc=False,
    )

    desc_orig = mace_embedding_script._descriptors_no_force(calc, methane_atoms)
    desc_rot = mace_embedding_script._descriptors_no_force(calc, rotated)

    max_diff = float(np.abs(desc_orig - desc_rot).max())
    assert max_diff < 1e-3, (
        f"Rotation changed descriptors by {max_diff:.2e}; "
        "invariants_only=True should give rotation-invariant features"
    )

def test_descriptor_shape_mace_off_medium(calc, methane_atoms) -> None:
    """MACE-OFF medium produces [N x 256] descriptors for N atoms."""
    result = mace_embedding_script._descriptors_no_force(calc, methane_atoms)
    n_atoms = len(methane_atoms)
    assert result.shape == (n_atoms, 256), (
        f"Expected ({n_atoms}, 256) but got {result.shape}. "
        "Check that MACE_TEST_MODEL points to a medium MACE-OFF checkpoint."
    )


@pytest.fixture(scope="module")
def protein():
    """A prepared protein NPZ from MACE_TEST_NPZ; skip if unset."""
    npz_path = os.environ.get("MACE_TEST_NPZ", "")
    if not npz_path:
        pytest.skip("MACE_TEST_NPZ env var not set; skipping protein tests")
    import mlip_common  # noqa: PLC0415

    return mlip_common.load_structure(Path(npz_path))


def test_chunked_matches_full(calc, protein) -> None:
    """Spatial chunks with a full receptive-field halo reproduce one pass.

    Negative control: a halo of one cutoff (half the receptive field of a
    two-layer model) must change the result, so the test can fail.
    """
    numbers, positions = protein.numbers, protein.positions
    full = mace_embedding_script._atom_features(calc, numbers, positions)
    halo = mace_embedding_script.receptive_field(calc)
    max_atoms = len(numbers) // 2
    chunked = mace_embedding_script.chunked_atom_features(
        calc, numbers, positions, max_atoms, halo
    )
    import mlip_common  # noqa: PLC0415

    n_chunks = len(mlip_common.spatial_chunks(positions, max_atoms, halo))
    assert n_chunks >= 2
    scale = float(np.abs(full).max())
    assert float(np.abs(chunked - full).max()) < 1e-5 * max(scale, 1.0)

    short = mace_embedding_script.chunked_atom_features(
        calc, numbers, positions, max_atoms, halo / 2
    )
    assert float(np.abs(short - full).max()) > 1e-3 * scale
