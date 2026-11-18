"""Integration tests for SO3LR Tauri embedding.

Skipped automatically unless SO3LR_TEST_MODEL is set. The variable may be:
  - A size key: 'S', 'M', or 'L'  (uses the bundled versioned checkpoint)
  - A filesystem path to a custom checkpoint directory

Set the env var to run these tests on a GPU compute node:

    SO3LR_TEST_MODEL=M pytest tests/test_so3lr_embedding.py -v

All tests use a small synthetic peptide (5 residues, ~41 atoms including
hydrogens) to avoid JAX compilation times dominating the test run.

The rotation-invariance test uses x/y/z permutation (a 90 degree rotation
in the x-z plane) rather than a random rotation to avoid floating-point
accumulation issues with general rotation matrices.

test_padding_invariance proves that the bucketed padding approach is harmless:
padded vs unpadded x values agree within float32 rounding, and two different
bucket sizes produce the same output for the same protein.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

_SO3LR_MODEL = os.environ.get("SO3LR_TEST_MODEL", "")
if not _SO3LR_MODEL:
    pytest.skip(
        "SO3LR_TEST_MODEL env var not set; skipping SO3LR real-weight tests",
        allow_module_level=True,
    )

pytest.importorskip("so3lr")

_EMBEDDING_SCRIPTS = str(Path(__file__).resolve().parents[1] / "embedding_scripts")
if _EMBEDDING_SCRIPTS not in sys.path:
    sys.path.insert(0, _EMBEDDING_SCRIPTS)

import mlip_common  # noqa: E402
import so3lr_embedding_script  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_size() -> str:
    """The size key ('S', 'M', or 'L') if SO3LR_TEST_MODEL is a valid key,
    else 'M' as the display-only label for a custom path."""
    keys = list(so3lr_embedding_script._VERSIONED_NAMES)
    return _SO3LR_MODEL if _SO3LR_MODEL in keys else "M"


@pytest.fixture(scope="module")
def versioned_name(model_size) -> str:
    if _SO3LR_MODEL in so3lr_embedding_script._VERSIONED_NAMES:
        return so3lr_embedding_script._VERSIONED_NAMES[_SO3LR_MODEL]
    return _SO3LR_MODEL  # treat as a path or already-versioned name


@pytest.fixture(scope="module")
def model_assets(versioned_name):
    """Load model weights and build jit function once per test session."""
    net, params, cutoff = so3lr_embedding_script._load_repr_model(versioned_name)
    jit_fn = so3lr_embedding_script._jit_obs_fn(net)
    return net, params, cutoff, jit_fn


@pytest.fixture(scope="module")
def embed_fn(model_assets):
    """Return a fully assembled embed_fn."""
    net, params, cutoff, jit_fn = model_assets
    return so3lr_embedding_script.make_embed_fn(net, params, cutoff, jit_fn)


@pytest.fixture(scope="module")
def alanine_structure() -> mlip_common.Structure:
    """Synthetic 5-residue Ala peptide-like structure.

    Numbers: H=1, C=6, N=7, O=8. Residue indices match a real backbone
    pattern so to_residues does not raise. Total charge = 0 (neutral).
    sum(Z) = 1*5 + 6*5 + 7*5 + 8*5 = 130, charge = 0, 130 is even.
    """
    rng = np.random.default_rng(42)
    n_residues = 5
    atoms_per_res = 8  # simplified: 1N + 2C + 1O + 4H

    numbers = []
    residue_index = []
    for r in range(n_residues):
        numbers.extend([7, 6, 6, 8, 1, 1, 1, 1])  # N, Ca, C, O, H*4
        residue_index.extend([r] * atoms_per_res)

    n_atoms = len(numbers)
    # Place atoms in a compact cluster so pairs exist within cutoff.
    positions = rng.uniform(-5.0, 5.0, (n_atoms, 3)).astype(np.float32)
    residue_index = np.array(residue_index, dtype=np.int32)
    numbers = np.array(numbers, dtype=np.int32)

    # sum = 7*5 + 6*5 + 6*5 + 8*5 + 1*20 = 155, which is odd.
    # Add one more H to make it even.
    numbers = np.append(numbers, 1).astype(np.int32)
    residue_index = np.append(residue_index, n_residues - 1).astype(np.int32)
    positions = np.vstack([positions, rng.uniform(-5.0, 5.0, (1, 3))]).astype(
        np.float32
    )

    return mlip_common.Structure(
        id="ala5_test",
        numbers=numbers,
        positions=positions,
        residue_index=residue_index,
        plddt=np.ones(n_residues, dtype=np.float32),
        charge=0,
        sequence="AAAAA",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(embed_fn, alanine_structure) -> None:
    """embed_fn returns [L x d] with L equal to the sequence length."""
    structure = alanine_structure
    n_res = len(structure.sequence)
    result = embed_fn(structure)
    assert result.ndim == 2, f"Expected 2D tensor, got shape {result.shape}"
    assert result.shape[0] == n_res, (
        f"Expected {n_res} residues, got {result.shape[0]}"
    )
    assert result.dtype == pytest.approx or result.dtype in (
        __import__("torch").float32,
    ), f"Expected float32 tensor, got {result.dtype}"
    assert result.shape[0] == n_res


def test_embedding_dim(model_size, embed_fn, alanine_structure) -> None:
    """Embedding dimension matches the expected size (128 for S/M, 256 for L)."""
    expected_d = so3lr_embedding_script._EMBED_DIM.get(model_size)
    if expected_d is None:
        pytest.skip(f"No expected dim for custom model path {_SO3LR_MODEL!r}")
    result = embed_fn(alanine_structure)
    assert result.shape[1] == expected_d, (
        f"Expected d={expected_d} for size {model_size}, got {result.shape[1]}"
    )


def test_rotation_invariance(model_assets, alanine_structure) -> None:
    """Atomic representations are invariant under a 90 degree x->z->-x rotation.

    Rotated positions: (x, y, z) -> (z, y, -x). This is an orthogonal
    transformation. Grade-0 (invariant) outputs must be identical.
    """
    net, params, cutoff, jit_fn = model_assets
    structure = alanine_structure

    fn = so3lr_embedding_script.make_embed_fn(net, params, cutoff, jit_fn)

    # Rotate: (x, y, z) -> (z, y, -x)
    pos_orig = structure.positions
    pos_rot = np.column_stack([pos_orig[:, 2], pos_orig[:, 1], -pos_orig[:, 0]])
    rotated = mlip_common.Structure(
        id=structure.id + "_rot",
        numbers=structure.numbers,
        positions=pos_rot.astype(np.float32),
        residue_index=structure.residue_index,
        plddt=structure.plddt,
        charge=structure.charge,
        sequence=structure.sequence,
    )

    emb_orig = fn(structure).numpy()
    emb_rot = fn(rotated).numpy()

    max_diff = float(np.abs(emb_orig - emb_rot).max())
    assert max_diff < 1e-3, (
        f"Rotation changed embedding by {max_diff:.2e}; "
        "atomic_representations should be rotation-invariant (grade-0 only)"
    )


def test_odd_electron_count_raises(embed_fn) -> None:
    """embed_fn raises ValueError for open-shell structures.

    sum(Z) - charge is odd when the system has an unpaired electron.
    Example: single H atom (Z=1), charge=0 -> 1-0=1 (odd).
    """
    rng = np.random.default_rng(7)
    odd_structure = mlip_common.Structure(
        id="odd_electrons",
        numbers=np.array([1], dtype=np.int32),  # single H
        positions=rng.uniform(-2.0, 2.0, (1, 3)).astype(np.float32),
        residue_index=np.array([0], dtype=np.int32),
        plddt=np.array([1.0], dtype=np.float32),
        charge=0,  # 1 - 0 = 1, odd
        sequence="A",
    )
    with pytest.raises(ValueError, match="odd"):
        embed_fn(odd_structure)


def test_determinism(embed_fn, alanine_structure) -> None:
    """Two calls with the same structure give bit-identical embeddings.

    The script sets --xla_gpu_deterministic_ops=true before JAX starts.
    Without it, GPU scatter-adds differed by up to 4e-3 between two calls
    on L40S (2026-11-11).
    """
    r1 = embed_fn(alanine_structure).numpy()
    r2 = embed_fn(alanine_structure).numpy()
    max_diff = float(np.abs(r1 - r2).max())
    assert max_diff == 0.0, f"Embedding differs between two calls by {max_diff:.2e}"


def test_local_pairs_directed(model_assets) -> None:
    """_local_pairs includes both directions for every pair within cutoff."""
    _, _, cutoff, _ = model_assets
    # Simple 3-atom system where all pairs are within cutoff.
    pos = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    idx_i, idx_j = so3lr_embedding_script._local_pairs(pos, cutoff=5.0)
    # Each undirected pair (a, b) should appear as (a, b) and (b, a).
    pairs_set = set(zip(idx_i.tolist(), idx_j.tolist()))
    for a in range(3):
        for b in range(3):
            if a != b and np.linalg.norm(pos[a] - pos[b]) < 5.0:
                assert (a, b) in pairs_set, (
                    f"Missing directed edge ({a}, {b}) from local pairs"
                )


def test_charge_spin_passed(model_assets, alanine_structure) -> None:
    """Charged version produces a different embedding than neutral version.

    If charge were silently dropped, adding a charge would give the same
    result. This test verifies the model actually uses total_charge.
    """
    net, params, cutoff, jit_fn = model_assets
    fn_neutral = so3lr_embedding_script.make_embed_fn(net, params, cutoff, jit_fn)

    # Build a charged version: charge=-1, sum(Z) must still be even.
    # sum(Z)=156 (even), charge=0 -> electrons=156. Change charge to -2 -> 158, even.
    charged = mlip_common.Structure(
        id=alanine_structure.id + "_charged",
        numbers=alanine_structure.numbers,
        positions=alanine_structure.positions,
        residue_index=alanine_structure.residue_index,
        plddt=alanine_structure.plddt,
        charge=-2,
        sequence=alanine_structure.sequence,
    )

    emb_neutral = fn_neutral(alanine_structure).numpy()
    emb_charged = fn_neutral(charged).numpy()

    max_diff = float(np.abs(emb_neutral - emb_charged).max())
    assert max_diff > 1e-6, (
        f"Charged and neutral embeddings are identical (max_diff={max_diff:.2e}); "
        "total_charge may not be reaching the model"
    )


def test_padding_invariance(model_assets, alanine_structure) -> None:
    """Padded and unpadded runs agree; two bucket sizes give the same residue tensor.

    Proves the jraph padding convention is harmless: padding atoms (node_mask=False)
    and their self-loop edges do not leak into real atoms' representations.
    Also proves that forcing a larger bucket (more padding) gives the same result,
    showing the amount of padding is irrelevant.
    """
    import jax
    import jax.numpy as jnp
    from so3lr.mlff.nn.stacknet.observable_function_sparse import (
        get_observable_fn_sparse,
    )

    net, params, cutoff, _jit_fn = model_assets
    jit_fn = jax.jit(get_observable_fn_sparse(net))
    st = alanine_structure

    idx_i, idx_j = so3lr_embedding_script._local_pairs(st.positions, cutoff)
    n_real = len(st.numbers)
    p_real = len(idx_i)

    # Run unpadded (single graph, no masks).
    charge_1 = jnp.array([float(st.charge)])
    spin_1 = jnp.zeros(1)
    result_unpad = jit_fn(
        params,
        jnp.array(st.positions), jnp.array(st.numbers, dtype=jnp.int32),
        jnp.array(idx_i, dtype=jnp.int32), jnp.array(idx_j, dtype=jnp.int32),
        total_charge=charge_1, num_unpaired_electrons=spin_1,
    )
    x_unpad = np.array(result_unpad["atomic_representations"])  # (N_real, d)

    # Run with tight bucket (smallest bucket that fits).
    n_bkt_tight = so3lr_embedding_script._next_bucket(
        n_real, so3lr_embedding_script._NODE_BUCKETS
    )
    edge_buckets = [b * 40 for b in so3lr_embedding_script._NODE_BUCKETS]
    p_bkt_tight = so3lr_embedding_script._next_bucket(p_real, edge_buckets)
    (pos_p, num_p, ii_p, jj_p, nm, bs, gm, ch2, sp2) = (
        so3lr_embedding_script._pad_to_buckets(
            st.numbers, st.positions, idx_i, idx_j, st.charge,
            n_bkt_tight, p_bkt_tight,
        )
    )
    result_tight = jit_fn(
        params,
        jnp.array(pos_p), jnp.array(num_p, dtype=jnp.int32),
        jnp.array(ii_p, dtype=jnp.int32), jnp.array(jj_p, dtype=jnp.int32),
        total_charge=jnp.array(ch2), num_unpaired_electrons=jnp.array(sp2),
        node_mask=jnp.array(nm), batch_segments=jnp.array(bs, dtype=jnp.int32),
        graph_mask=jnp.array(gm),
    )
    x_tight = np.array(result_tight["atomic_representations"][:n_real])

    # Run with the next-larger bucket (more padding = more self-loop edges).
    bkt_idx = so3lr_embedding_script._NODE_BUCKETS.index(n_bkt_tight)
    n_bkt_large = so3lr_embedding_script._NODE_BUCKETS[
        min(bkt_idx + 1, len(so3lr_embedding_script._NODE_BUCKETS) - 1)
    ]
    p_bkt_large = n_bkt_large * 40
    (pos_pl, num_pl, ii_pl, jj_pl, nml, bsl, gml, ch2l, sp2l) = (
        so3lr_embedding_script._pad_to_buckets(
            st.numbers, st.positions, idx_i, idx_j, st.charge,
            n_bkt_large, p_bkt_large,
        )
    )
    result_large = jit_fn(
        params,
        jnp.array(pos_pl), jnp.array(num_pl, dtype=jnp.int32),
        jnp.array(ii_pl, dtype=jnp.int32), jnp.array(jj_pl, dtype=jnp.int32),
        total_charge=jnp.array(ch2l), num_unpaired_electrons=jnp.array(sp2l),
        node_mask=jnp.array(nml), batch_segments=jnp.array(bsl, dtype=jnp.int32),
        graph_mask=jnp.array(gml),
    )
    x_large = np.array(result_large["atomic_representations"][:n_real])

    # Different shapes change the summation order. With full float32 matmuls
    # this moved x by at most 6e-6 on real proteins (2026-11-11); with the
    # default TF32 it was 5e-3, which is why this budget once failed.
    nondeterminism_budget = 1e-4
    diff_tight = float(np.abs(x_unpad - x_tight).max())
    diff_large = float(np.abs(x_tight - x_large).max())

    assert diff_tight < nondeterminism_budget, (
        f"Padded (tight) vs unpadded differ by {diff_tight:.2e}; "
        "padding is leaking into real atoms"
    )
    assert diff_large < nondeterminism_budget, (
        f"Two bucket sizes differ by {diff_large:.2e}; "
        "result should be independent of padding amount"
    )
