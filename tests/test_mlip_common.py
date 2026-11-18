"""Pure-logic tests for mlip_common: to_residues, resume, and OOM handling."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

# mlip_common and shard_writer live in embedding_scripts/, not in src/.
_EMBEDDING_SCRIPTS = str(Path(__file__).resolve().parents[1] / "embedding_scripts")
if _EMBEDDING_SCRIPTS not in sys.path:
    sys.path.insert(0, _EMBEDDING_SCRIPTS)

import mlip_common  # noqa: E402


def _make_atoms(residue_assignments: list[int], d: int = 4) -> tuple:
    """Return (x, residue_index, n_residues) for a simple atom layout."""
    ri = torch.tensor(residue_assignments, dtype=torch.long)
    n = len(residue_assignments)
    # Each atom feature is its residue index broadcast to d dimensions.
    x = ri.unsqueeze(1).expand(n, d).float()
    n_residues = int(ri.max().item()) + 1
    return x, ri, n_residues

def test_to_residues_correct_average() -> None:
    """Each residue row equals the mean of its atoms' features."""
    # Residue 0: atoms 0,1,2; Residue 1: atom 3; Residue 2: atom 4
    assignments = [0, 0, 0, 1, 2]
    d = 4
    ri = torch.tensor(assignments, dtype=torch.long)
    # Atom features: row i = i * ones(d)
    x = torch.arange(len(assignments), dtype=torch.float32).unsqueeze(1).expand(-1, d)
    result = mlip_common.to_residues(x, ri, n_residues=3)

    assert result.shape == (3, d)
    expected_r0 = (0.0 + 1.0 + 2.0) / 3.0
    expected_r1 = 3.0
    expected_r2 = 4.0
    assert torch.allclose(result[0], torch.full((d,), expected_r0))
    assert torch.allclose(result[1], torch.full((d,), expected_r1))
    assert torch.allclose(result[2], torch.full((d,), expected_r2))

def test_to_residues_shape() -> None:
    """Output has shape [n_residues x d]."""
    x, ri, n_res = _make_atoms([0, 0, 1, 1, 2], d=8)
    result = mlip_common.to_residues(x, ri, n_res)
    assert result.shape == (3, 8)

def test_to_residues_missing_residue_raises() -> None:
    """Raises ValueError when a residue has no atoms."""
    # 3 residues declared but residue 1 has no atom
    ri = torch.tensor([0, 0, 2], dtype=torch.long)
    x = torch.ones(3, 4)
    with pytest.raises(ValueError, match="Residues with no atoms"):
        mlip_common.to_residues(x, ri, n_residues=3)

def test_to_residues_single_atom_per_residue() -> None:
    """When each residue has exactly one atom, output equals input."""
    ri = torch.tensor([0, 1, 2], dtype=torch.long)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = mlip_common.to_residues(x, ri, n_residues=3)
    assert torch.allclose(result, x)

def test_protein_vector_is_mean_of_residues() -> None:
    """The pooled protein vector from embed_structures is mean over residues.

    Residue 0 has 3 atoms; residue 1 has 1 atom. The correct protein vector
    averages the two residue vectors (1/2 each), not the four atom vectors
    (which would give residue 0 three times the weight).
    """
    # Residue 0 atoms: feature 0.0 each. Residue 1 atom: feature 1.0.
    ri = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    x = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    residue_emb = mlip_common.to_residues(x, ri, n_residues=2)
    # Residue means: [0.0, 1.0]
    protein_vec = residue_emb.mean(dim=0)
    # Mean of residue vectors = (0.0 + 1.0) / 2 = 0.5, not 0.25
    assert float(protein_vec[0]) == pytest.approx(0.5)

def _make_structure(
    seq_id: str, n_atoms: int = 4, n_res: int = 2
) -> mlip_common.Structure:
    return mlip_common.Structure(
        id=seq_id,
        numbers=np.ones(n_atoms, dtype=np.int32),
        positions=np.zeros((n_atoms, 3), dtype=np.float32),
        residue_index=np.array([i % n_res for i in range(n_atoms)], dtype=np.int32),
        plddt=np.zeros(n_res, dtype=np.float32),
        charge=0,
        sequence="A" * n_res,
    )

def _save_structure_npz(structure_dir: Path, s: mlip_common.Structure) -> None:
    """Save a Structure to NPZ using the current production format (np.str_)."""
    np.savez_compressed(
        structure_dir / f"{s.id}.npz",
        numbers=s.numbers,
        positions=s.positions,
        residue_index=s.residue_index,
        plddt=s.plddt,
        charge=np.int32(s.charge),
        sequence=np.str_(s.sequence),
    )

def test_embed_structures_skips_existing(tmp_path: Path) -> None:
    """embed_structures does not call embed_fn for IDs already in the shard."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()

    for sid in ("prot_A", "prot_B"):
        _save_structure_npz(structure_dir, _make_structure(sid))

    output_dir = tmp_path / "out"
    call_log: list[str] = []

    def embed_fn(structure: mlip_common.Structure) -> torch.Tensor:
        call_log.append(structure.id)
        return torch.zeros(2, 4)

    # First run: embeds both.
    mlip_common.embed_structures(
        structure_dir=structure_dir,
        output=output_dir,
        mean_pooled_output=None,
        embed_fn=embed_fn,
    )
    assert set(call_log) == {"prot_A", "prot_B"}

    # Second run: should skip both.
    call_log.clear()
    mlip_common.embed_structures(
        structure_dir=structure_dir,
        output=output_dir,
        mean_pooled_output=None,
        embed_fn=embed_fn,
    )
    assert call_log == [], "embed_fn must not be called for already-embedded IDs"

def test_oom_is_counted_and_run_continues(tmp_path: Path) -> None:
    """RuntimeError with 'out of memory' message is caught; run continues."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()

    for sid in ("good", "oom_one", "also_good"):
        _save_structure_npz(structure_dir, _make_structure(sid))

    embedded: list[str] = []

    def embed_fn(structure: mlip_common.Structure) -> torch.Tensor:
        if structure.id == "oom_one":
            # TorchScript OOM arrives as plain RuntimeError, not the subclass.
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        embedded.append(structure.id)
        return torch.zeros(2, 4)

    mlip_common.embed_structures(
        structure_dir=structure_dir,
        output=tmp_path / "out",
        mean_pooled_output=None,
        embed_fn=embed_fn,
    )
    assert "good" in embedded
    assert "also_good" in embedded
    assert "oom_one" not in embedded

def test_non_oom_error_propagates(tmp_path: Path) -> None:
    """RuntimeError without 'out of memory' in the message stops the run."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()

    _save_structure_npz(structure_dir, _make_structure("only_one"))

    def embed_fn(_: mlip_common.Structure) -> torch.Tensor:
        raise RuntimeError("unexpected model failure")

    with pytest.raises(RuntimeError, match="unexpected model failure"):
        mlip_common.embed_structures(
            structure_dir=structure_dir,
            output=tmp_path / "out",
            mean_pooled_output=None,
            embed_fn=embed_fn,
        )

def test_ids_file_prefix_and_float32_pooled(tmp_path: Path) -> None:
    """Two slices share one shard directory; pooled vectors stay float32."""
    from safetensors import safe_open

    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    ids = ["p0", "p1", "p2", "p3"]
    for sid in ids:
        _save_structure_npz(structure_dir, _make_structure(sid))
    (tmp_path / "a.txt").write_text("p0\np2\nnot_prepared\n")
    (tmp_path / "b.txt").write_text("p1\np3\n")

    value = 1.0 + 2.0**-12  # representable in float32, rounds to 1.0 in bf16

    def embed_fn(structure: mlip_common.Structure) -> torch.Tensor:
        return torch.full((2, 4), value)

    shard_dir = tmp_path / "x_shards"
    for name in ("a", "b"):
        mlip_common.embed_structures(
            structure_dir=structure_dir,
            output=shard_dir,
            mean_pooled_output=tmp_path / f"pooled_{name}.safetensors",
            embed_fn=embed_fn,
            ids_file=tmp_path / f"{name}.txt",
            shard_prefix=f"s{name}_",
        )

    shard_names = sorted(p.name for p in shard_dir.iterdir())
    assert shard_names == ["sa_00000.safetensors", "sb_00000.safetensors"]
    keys = {}
    for name in shard_names:
        with safe_open(str(shard_dir / name), framework="pt") as fh:
            keys[name] = set(fh.keys())
            assert fh.get_tensor(next(iter(fh.keys()))).dtype == torch.bfloat16
    assert keys["sa_00000.safetensors"] == {"p0", "p2"}
    assert keys["sb_00000.safetensors"] == {"p1", "p3"}

    with safe_open(str(tmp_path / "pooled_a.safetensors"), framework="pt") as fh:
        assert set(fh.keys()) == {"p0", "p2"}
        pooled = fh.get_tensor("p0")
    assert pooled.dtype == torch.float32
    assert float(pooled[0]) == value


def test_shard_writer_ignores_other_prefixes_and_tmp(tmp_path: Path) -> None:
    """Resume reads only this prefix's shards; a leftover .tmp is never globbed."""
    import shard_writer

    with shard_writer.SafetensorsShardWriter(tmp_path, prefix="sa_") as w:
        w.add("k1", torch.zeros(2, 2))
    with shard_writer.SafetensorsShardWriter(tmp_path, prefix="sb_") as w:
        assert "k1" not in w
        w.add("k2", torch.zeros(2, 2))
    (tmp_path / "sa_00001.safetensors.tmp").write_bytes(b"truncated")

    resumed = shard_writer.SafetensorsShardWriter(tmp_path, prefix="sa_")
    assert "k1" in resumed and "k2" not in resumed
    assert resumed._next_index == 1
    assert sorted(p.name for p in tmp_path.glob("*.safetensors")) == [
        "sa_00000.safetensors",
        "sb_00000.safetensors",
    ]


def test_spatial_chunks_partition_and_halo() -> None:
    """Cores partition the atoms; each context has every atom within halo."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 60.0, (4000, 3)).astype(np.float32)
    halo = 6.0
    chunks = mlip_common.spatial_chunks(positions, max_atoms=900, halo=halo)
    assert len(chunks) > 4

    cores = np.concatenate([core for core, _ in chunks])
    assert np.array_equal(np.sort(cores), np.arange(len(positions)))
    for core, context in chunks:
        assert len(context) <= 900
        assert np.isin(core, context).all()
        diff = positions[core][:, None, :] - positions[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        needed = np.flatnonzero((dist <= halo).any(axis=0))
        assert np.isin(needed, context).all()


def test_spatial_chunks_single_chunk_when_small() -> None:
    positions = np.zeros((10, 3), dtype=np.float32)
    chunks = mlip_common.spatial_chunks(positions, max_atoms=10, halo=5.0)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0][0], np.arange(10))


def test_spatial_chunks_halo_too_large_raises() -> None:
    positions = np.zeros((50, 3), dtype=np.float32)  # all atoms in one point
    with pytest.raises(ValueError, match="context atoms"):
        mlip_common.spatial_chunks(positions, max_atoms=10, halo=1.0)


def test_merge_pooled_slices(tmp_path: Path) -> None:
    """Slices merge; duplicates and pooled/shard mismatches are refused."""
    import merge_pooled_slices
    from safetensors.torch import save_file
    import shard_writer

    slices = tmp_path / "pooled"
    slices.mkdir()
    save_file({"a": torch.ones(4), "b": torch.zeros(4)}, str(slices / "s0.safetensors"))
    save_file({"c": torch.ones(4)}, str(slices / "s1.safetensors"))
    shards = tmp_path / "x_shards"
    with shard_writer.SafetensorsShardWriter(shards, prefix="s0_") as w:
        for key in ("a", "b", "c"):
            w.add(key, torch.zeros(3, 4))

    out = tmp_path / "x.safetensors"
    merge_pooled_slices.run(slices, out, shards)
    assert set(merge_pooled_slices.merge_pooled([out])) == {"a", "b", "c"}

    with shard_writer.SafetensorsShardWriter(shards, prefix="s1_") as w:
        w.add("d", torch.zeros(3, 4))
    with pytest.raises(ValueError, match="differ"):
        merge_pooled_slices.run(slices, out, shards)

    save_file({"a": torch.ones(4)}, str(slices / "s2.safetensors"))
    with pytest.raises(ValueError, match="is in"):
        merge_pooled_slices.run(slices, out, None)
