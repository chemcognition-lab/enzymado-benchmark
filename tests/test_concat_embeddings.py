"""Concatenated embeddings (PLM + MLIP): a directory with concat.json whose parts are
ordinary embedding sources, joined along the feature axis."""

import json
import pickle

import pytest
import torch
from safetensors.torch import save_file

from enzymado.artifacts import reproducibility
from enzymado.data_loading import stores


def _spec(tmp_path, name, parts):
    folder = tmp_path / name
    folder.mkdir()
    (folder / stores.CONCAT_SPEC_NAME).write_text(
        json.dumps({"parts": [str(p) for p in parts]})
    )
    return folder


@pytest.fixture
def pooled(tmp_path):
    a = {"p1": torch.tensor([1.0, 2.0, 3.0]), "p2": torch.tensor([4.0, 5.0, 6.0]),
         "only_a": torch.tensor([0.0, 0.0, 0.0])}
    b = {"p1": torch.tensor([7.0, 8.0]), "p2": torch.tensor([9.0, 10.0]),
         "only_b": torch.tensor([0.0, 0.0])}
    save_file(a, tmp_path / "a.safetensors")
    save_file(b, tmp_path / "b.safetensors")
    return a, b, _spec(tmp_path, "a+b", [tmp_path / "a.safetensors", tmp_path / "b.safetensors"])


@pytest.fixture
def residue(tmp_path):
    torch.manual_seed(0)
    a = {"p1": torch.randn(4, 3).bfloat16(), "p2": torch.randn(5, 3).bfloat16(),
         "only_a": torch.randn(2, 3).bfloat16()}
    b = {"p1": torch.randn(4, 2).bfloat16(), "p2": torch.randn(6, 2).bfloat16()}
    for name, tensors in (("a_shards", a), ("b_shards", b)):
        (tmp_path / name).mkdir()
        save_file(tensors, tmp_path / name / "shard_000.safetensors")
    return a, b, _spec(tmp_path, "a+b_res", [tmp_path / "a_shards", tmp_path / "b_shards"])


def test_plain_source_is_not_a_concatenation(pooled):
    _, _, folder = pooled
    assert stores.concat_parts(folder.parent / "a.safetensors") is None
    assert len(stores.concat_parts(folder)) == 2


def test_spec_needs_two_parts(tmp_path):
    folder = _spec(tmp_path, "one", [tmp_path / "a.safetensors"])
    with pytest.raises(ValueError, match="at least two parts"):
        stores.concat_parts(folder)


def test_pooled_vectors_join_in_part_order(pooled):
    a, b, folder = pooled
    by_id, by_seq = stores.load_pooled_safetensors_subset(
        folder, {"p1", "p2", "only_a", "only_b"}, set()
    )
    assert by_seq == {}
    assert set(by_id) == {"p1", "p2"}  # a key missing from any part is dropped
    for key in ("p1", "p2"):
        assert torch.equal(by_id[key], torch.cat([a[key], b[key]]))


@pytest.mark.parametrize("loader", ["lazy", "materialized"])
def test_residue_lookup_joins_features(residue, loader):
    a, b, folder = residue
    load = {
        "lazy": stores.load_lazy_safetensors_embedding_store,
        "materialized": stores.load_safetensors_embedding_store_subset,
    }[loader]
    store = load(folder, {"p1", "p2", "only_a"}, set())
    assert isinstance(store, stores.ConcatResidueEmbeddingStore)
    assert store.lazy is (loader == "lazy")
    assert store._available_ids == {"p1", "p2"}
    assert store.has_id("p1") and not store.has_id("only_a")
    joined = store.lookup_id("p1")
    assert joined.shape == (4, 5) and joined.dtype == torch.bfloat16
    assert torch.equal(joined, torch.cat([a["p1"], b["p1"]], dim=1))
    assert store.lookup_id_length("p1") == 4
    row = {"uniprot_id": "p1", "protein_sequence": ""}
    assert torch.equal(store.lookup_row(row, id_columns=("uniprot_id",)), joined)


@pytest.mark.parametrize("loader", ["lazy", "materialized"])
def test_residue_count_mismatch_raises(residue, loader):
    _, _, folder = residue
    load = {
        "lazy": stores.load_lazy_safetensors_embedding_store,
        "materialized": stores.load_safetensors_embedding_store_subset,
    }[loader]
    store = load(folder, {"p2"}, set())
    with pytest.raises(ValueError, match="disagree on the residue count"):
        store.lookup_id("p2")
    with pytest.raises(ValueError, match="disagree on the residue count"):
        store.lookup_id_length("p2")


def test_mixed_dtypes_upcast(tmp_path):
    save_file({"p1": torch.ones(3, 2).bfloat16()}, tmp_path / "a.safetensors")
    save_file({"p1": torch.full((3, 1), 0.5)}, tmp_path / "b.safetensors")
    folder = _spec(tmp_path, "mix", [tmp_path / "a.safetensors", tmp_path / "b.safetensors"])
    joined = stores.load_lazy_safetensors_embedding_store(folder, {"p1"}, set()).lookup_id("p1")
    assert joined.dtype == torch.float32
    assert torch.equal(joined, torch.tensor([[1.0, 1.0, 0.5]] * 3))


def test_lazy_store_survives_pickling(residue):
    a, b, folder = residue
    store = stores.load_lazy_safetensors_embedding_store(folder, {"p1"}, set())
    store.lookup_id("p1")  # opens file handles, which must not be pickled
    clone = pickle.loads(pickle.dumps(store))
    assert torch.equal(clone.lookup_id("p1"), torch.cat([a["p1"], b["p1"]], dim=1))


def test_fingerprint_lists_every_part(residue):
    _, _, folder = residue
    fingerprint = reproducibility.fingerprint_embedding(folder)
    assert fingerprint["kind"] == "concatenation"
    assert len(fingerprint["parts"]) == 2
    assert all(part["kind"] == "tree_inventory_sha256" for part in fingerprint["parts"])
