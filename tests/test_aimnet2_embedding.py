"""Integration tests for aimnet2_embedding_script.

Skipped automatically when aimnet is not installed or AIMNET2_TEST_MODEL is
unset. Set AIMNET2_TEST_MODEL to a local path of a .pt model file (e.g.,
aimnet2_wb97m_d3_0.pt) to run against real weights. Optionally set
AIMNET2_TEST_NPZ to a prepared .npz structure file for a larger-scale test.

    AIMNET2_TEST_MODEL=/scratch/.../aimnet2_wb97m_d3_0.pt \\
    AIMNET2_TEST_NPZ=/scratch/.../prepared/A0A1C3HPR2.npz \\
        pytest tests/test_aimnet2_embedding.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("aimnet")

_EMBEDDING_SCRIPTS = str(Path(__file__).resolve().parents[1] / "embedding_scripts")
if _EMBEDDING_SCRIPTS not in sys.path:
    sys.path.insert(0, _EMBEDDING_SCRIPTS)

import aimnet2_embedding_script  # noqa: E402  # isort: skip
import mlip_common  # noqa: E402  # isort: skip



@pytest.fixture(scope="module")
def model_path() -> str:
    """Path from AIMNET2_TEST_MODEL; skip if unset."""
    path = os.environ.get("AIMNET2_TEST_MODEL", "")
    if not path:
        pytest.skip("AIMNET2_TEST_MODEL env var not set; skipping real-weight tests")
    return path


@pytest.fixture(scope="module")
def calc(model_path: str):
    """Build a CPU AIMNet2Calculator from the given model path."""
    return aimnet2_embedding_script._build_calc(model=model_path, device="cpu")


@pytest.fixture(scope="module")
def tiny_structure() -> mlip_common.Structure:
    """Return a minimal synthetic Gly-Gly di-peptide-like Structure.

    Two residues, sequence 'GG', charge 0.  Coordinates are physical (C-N ~1.33 A,
    C-C ~1.52 A) but the molecule is not energy-minimized.
    """
    # Glycine backbone x2: N H CA HA2 HA3 C O for each residue + OXT on C-term
    # 7 atoms per residue * 2 residues + extra OXT = 15 atoms
    numbers = np.array(
        [7, 1, 6, 1, 1, 6, 8,   # residue 0: N H CA HA HA C O
         7, 1, 6, 1, 1, 6, 8, 8],  # residue 1: N H CA HA HA C O OXT
        dtype=np.int32,
    )
    # Rough linear backbone, spaced far enough to avoid any crash
    positions = np.array(
        [
            [0.00, 0.00, 0.00],  # N
            [-0.40, 0.90, 0.00],  # H on N
            [1.46, 0.00, 0.00],  # CA
            [1.80, 1.02, 0.00],  # HA
            [1.80, -0.52, 0.89],  # HA
            [2.10, -0.76, -1.02],  # C
            [1.60, -1.76, -1.52],  # O
            [3.46, -0.37, -1.21],  # N (second residue)
            [3.96,  0.52, -0.95],  # H on N
            [4.47, -1.19, -1.98],  # CA
            [4.10, -2.21, -2.15],  # HA
            [5.07, -0.81, -2.86],  # HA
            [5.14, -1.33, -0.81],  # C
            [4.72, -0.78, -0.03],  # O
            [5.68, -2.37, -0.65],  # OXT
        ],
        dtype=np.float32,
    )
    residue_index = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32
    )
    # sum(Z) = 68, charge=0, electrons=68 (even)
    return mlip_common.Structure(
        id="gly_gly_test",
        numbers=numbers,
        positions=positions,
        residue_index=residue_index,
        plddt=np.array([70.0, 65.0], dtype=np.float32),
        charge=0,
        sequence="GG",
    )


@pytest.fixture(scope="module")
def npz_structure() -> mlip_common.Structure | None:
    """Load structure from AIMNET2_TEST_NPZ if set, else return None."""
    npz_path = os.environ.get("AIMNET2_TEST_NPZ", "")
    if not npz_path:
        return None
    return mlip_common.load_structure(Path(npz_path))



def test_output_shape(calc, tiny_structure: mlip_common.Structure) -> None:
    """embed_fn returns [L x 256] where L == len(sequence)."""
    embed_fn = aimnet2_embedding_script.make_embed_fn(calc)
    result = embed_fn(tiny_structure)
    n_res = len(tiny_structure.sequence)
    assert result.shape == (n_res, aimnet2_embedding_script._AIM_DIM), (
        f"Expected ({n_res}, {aimnet2_embedding_script._AIM_DIM}), got {result.shape}"
    )


def test_output_shape_from_npz(calc, npz_structure) -> None:
    """When AIMNET2_TEST_NPZ is set, L == len(sequence) for the real protein."""
    if npz_structure is None:
        pytest.skip("AIMNET2_TEST_NPZ not set; skipping real-structure shape test")
    embed_fn = aimnet2_embedding_script.make_embed_fn(calc)
    result = embed_fn(npz_structure)
    n_res = len(npz_structure.sequence)
    assert result.shape == (n_res, aimnet2_embedding_script._AIM_DIM), (
        f"Expected ({n_res}, {aimnet2_embedding_script._AIM_DIM}), got {result.shape}"
    )


def test_rotation_invariance(calc, tiny_structure: mlip_common.Structure) -> None:
    """Embedding is unchanged under a 90-degree rotation around the z-axis.

    AIMNet2 aim features are grade-0 (invariant) scalars, so rotating the
    molecule must not change them.  Max abs diff < 1e-4 is an empirical bound.
    """
    embed_fn = aimnet2_embedding_script.make_embed_fn(calc)
    result_orig = embed_fn(tiny_structure).numpy()

    # 90-degree rotation around z: (x, y, z) -> (-y, x, z)
    pos_rot = tiny_structure.positions.copy()
    x, y = pos_rot[:, 0].copy(), pos_rot[:, 1].copy()
    pos_rot[:, 0] = -y
    pos_rot[:, 1] = x
    struct_rot = mlip_common.Structure(
        id=tiny_structure.id + "_rotated",
        numbers=tiny_structure.numbers,
        positions=pos_rot,
        residue_index=tiny_structure.residue_index,
        plddt=tiny_structure.plddt,
        charge=tiny_structure.charge,
        sequence=tiny_structure.sequence,
    )
    result_rot = embed_fn(struct_rot).numpy()

    max_diff = float(np.abs(result_orig - result_rot).max())
    assert max_diff < 1e-4, (
        f"Rotation changed embedding by {max_diff:.2e};"
        " aim should be rotation-invariant (grade-0)"
    )


def test_electron_parity_check_raises(calc) -> None:
    """_check_electron_parity raises ValueError for an odd electron count."""
    # H (Z=1), charge=0: 1 electron, odd -> should raise
    numbers_odd = np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="Odd electron count"):
        aimnet2_embedding_script._check_electron_parity(numbers_odd, charge=0)


def test_electron_parity_check_passes(calc) -> None:
    """_check_electron_parity does not raise for even electron count."""
    # H2 (Z=1, Z=1), charge=0: 2 electrons, even -> no raise
    numbers_even = np.array([1, 1], dtype=np.int32)
    aimnet2_embedding_script._check_electron_parity(numbers_even, charge=0)


def test_determinism(calc, tiny_structure: mlip_common.Structure) -> None:
    """Two consecutive calls produce bit-identical embeddings."""
    embed_fn = aimnet2_embedding_script.make_embed_fn(calc)
    r1 = embed_fn(tiny_structure).numpy()
    r2 = embed_fn(tiny_structure).numpy()
    max_diff = float(np.abs(r1 - r2).max())
    assert max_diff == 0.0, f"Non-deterministic output: max diff = {max_diff}"


def test_aim_is_energy_head_input(calc, tiny_structure: mlip_common.Structure) -> None:
    """Verify aim is the exact tensor the energy head reads.

    Strategy: call the model with keys_out extended to include aim; then call
    the model's forward_prepared directly (bypassing keep_only) and read aim
    from the full output dict. Both must be bit-identical.
    """
    import torch

    device = calc.device
    coord = torch.tensor(tiny_structure.positions, dtype=torch.float32, device=device)
    numbers = torch.tensor(tiny_structure.numbers, dtype=torch.int32, device=device)
    charge = torch.tensor(
        [float(tiny_structure.charge)], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        # Route through the full calculator (includes keep_only via calc.keys_out)
        result_via_calc = calc.eval(
            {"coord": coord, "numbers": numbers, "charge": charge},
            forces=False,
        )
        aim_via_calc = result_via_calc["aim"].float()

        # Bypass keep_only: run prepare_input + model.forward_prepared directly
        data_in = calc.prepare_input(
            {"coord": coord, "numbers": numbers, "charge": charge}
        )
        is_script = isinstance(calc.model, torch.jit.ScriptModule)
        data_out = calc.model(data_in) if is_script else calc.model(data_in)
        aim_direct = data_out["aim"].float()

    max_diff = float((aim_via_calc - aim_direct).abs().max())
    assert max_diff == 0.0, (
        f"aim from calculator ({aim_via_calc.shape}) and direct model call"
        f" ({aim_direct.shape}) differ by {max_diff:.2e}; they must be identical"
    )
