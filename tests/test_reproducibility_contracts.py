"""Characterization tests for structure-preserving refactors."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from enzymado import paths_and_constants
import enzymado.artifacts.reproducibility
import enzymado.artifacts.results_layout
import enzymado.data_loading.data_registry
import enzymado.data_loading.stores
import enzymado.dataset_splitting
import enzymado.models.common
import enzymado.pipeline.run_config
import enzymado.prediction.predict
import enzymado.propertyprep.uniprot
import enzymado.training.model_selection

_P = paths_and_constants.Paths
_PROJECT_ROOT = _P.project_root
_SRC_ROOT = _P.src_dir


class FixedPredictor:
    """Return deterministic predictions without fitting a model."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (2.0 * x[:, 0] - x[:, 1]).astype(np.float32)


def _module_name(path: Path) -> str | None:
    """Return the production import name for one source module."""
    parts = path.relative_to(_SRC_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or None


def test_all_maintained_python_files_compile() -> None:
    """Catch syntax errors even in modules not reached by the normal tests."""
    roots = (
        _SRC_ROOT,
        _PROJECT_ROOT / "scripts",
        _PROJECT_ROOT / "embedding_scripts",
    )
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_all_library_modules_import() -> None:
    """Ensure every library module can be loaded in the project environment."""
    paths = sorted(_SRC_ROOT.rglob("*.py"))
    module_names = filter(None, map(_module_name, paths))
    for module_name in module_names:
        importlib.import_module(module_name)


def test_repository_roots_and_source_manifest_match_checkout() -> None:
    """Keep package paths and provenance anchored to the repository root."""
    assert _P.project_root == _PROJECT_ROOT
    assert enzymado.data_loading.data_registry.PROJECT_ROOT == _PROJECT_ROOT
    assert enzymado.data_loading.stores.PROJECT_ROOT == _PROJECT_ROOT
    assert enzymado.propertyprep.uniprot._DEFAULT_SWISSPROT_PATH == (
        _PROJECT_ROOT / "data/raw/swissprot/swissprot_ec.tsv"
    )

    expected_hash = enzymado.artifacts.reproducibility._source_tree_sha256(
        _PROJECT_ROOT
    )
    empty_hash = hashlib.sha256(b"").hexdigest()
    assert expected_hash != empty_hash

    build_static = enzymado.artifacts.reproducibility._build_static_environment_manifest
    build_static.cache_clear()
    try:
        environment = enzymado.artifacts.reproducibility.build_environment_manifest()
    finally:
        build_static.cache_clear()
    assert environment["source_tree_sha256"] == expected_hash


def test_deterministic_algorithms_are_explicitly_opt_in() -> None:
    """Keep legacy defaults while exposing a strict deterministic mode."""
    try:
        enzymado.models.common.seed_random_generators(
            17,
            deterministic_algorithms=True,
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert (
            enzymado.artifacts.reproducibility.build_environment_manifest()[
                "deterministic_algorithms"
            ]
            is True
        )

        enzymado.models.common.seed_random_generators(
            17,
            deterministic_algorithms=False,
        )
        assert not torch.are_deterministic_algorithms_enabled()
        assert (
            enzymado.artifacts.reproducibility.build_environment_manifest()[
                "deterministic_algorithms"
            ]
            is False
        )
    finally:
        torch.use_deterministic_algorithms(mode=False)


def test_run_manifest_fingerprints_inputs_without_modifying_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record data provenance without writing beside source artifacts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dataset_path = raw_dir / "dataset.csv"
    dataset_path.write_text("sample_id,split,target\nP1,train,1\n", encoding="utf-8")
    embedding_path = tmp_path / "embedding_shards"
    embedding_path.mkdir()
    (embedding_path / "shard_00001.safetensors").write_bytes(b"second")
    (embedding_path / "shard_00000.safetensors").write_bytes(b"first")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    hparams_path = run_dir / "hparams.json"
    hparams_path.write_text('{"epochs": 4}\n', encoding="utf-8")
    split_manifest = {
        "train": {"count": 1, "ordered_sha256": "a" * 64},
        "val": {"count": 0, "ordered_sha256": "b" * 64},
        "test": {"count": 1, "ordered_sha256": "c" * 64},
    }
    input_snapshot = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (dataset_path, *sorted(embedding_path.iterdir()))
    }
    monkeypatch.setattr(
        enzymado.artifacts.reproducibility,
        "build_environment_manifest",
        lambda: {"python": "test-python", "source_tree_sha256": "d" * 64},
    )

    manifest_path = enzymado.artifacts.reproducibility.write_run_manifest(
        run_dir,
        dataset_path=dataset_path,
        embedding_path=embedding_path,
        hparams_path=hparams_path,
        split_manifest=split_manifest,
        deterministic_algorithms=True,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["deterministic_algorithms"] is True
    assert payload["inputs"]["dataset"]["kind"] == "content_sha256"
    assert payload["inputs"]["embedding"]["kind"] == "tree_inventory_sha256"
    assert payload["inputs"]["hparams"]["kind"] == "content_sha256"
    assert payload["splits"] == split_manifest
    assert payload["environment"]["python"] == "test-python"
    assert len(payload["environment_sha256"]) == 64
    assert set(raw_dir.iterdir()) == {dataset_path}
    assert set(embedding_path.iterdir()) == {
        embedding_path / "shard_00000.safetensors",
        embedding_path / "shard_00001.safetensors",
    }
    for path, (content, modified_ns) in input_snapshot.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == modified_ns


def test_similarity_split_assignment_is_stable() -> None:
    """Lock the exact split assignment for a representative clustered dataset."""
    fieldnames = ["uniprot_id", "protein_sequence", "target"]
    rows = [
        {"uniprot_id": "P1", "protein_sequence": "AAAA", "target": "1"},
        {"uniprot_id": "P1", "protein_sequence": "AAAA", "target": "2"},
        {"uniprot_id": "P2", "protein_sequence": "BBBB", "target": "3"},
        {"uniprot_id": "P3", "protein_sequence": "CCCC", "target": "4"},
        {"uniprot_id": "P4", "protein_sequence": "DDDD", "target": "5"},
        {"uniprot_id": "P5", "protein_sequence": "EEEE", "target": "6"},
    ]
    clusters = {
        "P1": "cluster_a",
        "P2": "cluster_a",
        "P3": "cluster_b",
        "P4": "cluster_c",
        "P5": "cluster_d",
    }

    split_from_clusters = (
        enzymado.dataset_splitting.generate_similarity_split_from_clusters
    )
    _, split_rows, summary = split_from_clusters(
        rows,
        fieldnames,
        0.6,
        0.2,
        0.2,
        sequence_col="protein_sequence",
        group_clusters=clusters,
        group_col="uniprot_id",
        seed=3,
    )

    # Refrozen when the weight sort was removed from assign_items_to_splits.
    # With only five groups the greedy pass balances the row quotas less
    # tightly than the size-ordered pass did; on real datasets, which have
    # thousands of groups, the quotas stay on target and the bias the sort
    # introduced (val/test filling with the most-measured proteins) disappears.
    assignment = [(row["uniprot_id"], row["split"]) for row in split_rows]
    assert assignment == [
        ("P1", "val"),
        ("P1", "val"),
        ("P2", "val"),
        ("P3", "train"),
        ("P4", "test"),
        ("P5", "train"),
    ]
    assert summary == {
        "train": {"rows": 2, "proteins": 2},
        "val": {"rows": 3, "proteins": 2},
        "test": {"rows": 1, "proteins": 1},
    }


def test_array_prediction_plumbing_is_stable() -> None:
    """Lock split order, targets, masks, and deterministic prediction values."""
    x_train = np.asarray([[2.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    x_val = np.asarray([[5.0, 2.0]], dtype=np.float32)
    x_test = np.asarray([[7.0, 3.0], [11.0, 5.0]], dtype=np.float32)
    y_train = np.asarray([0.0, 1.0], dtype=np.float32)
    y_val = np.asarray([2.0], dtype=np.float32)
    y_test = np.asarray([3.0, 4.0], dtype=np.float32)
    mask_test = np.asarray([True, False])

    outputs = enzymado.prediction.predict.predict_from_arrays(
        FixedPredictor(),
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        use_validation=True,
        target_mask_test=mask_test,
    )

    np.testing.assert_array_equal(outputs.pred_train, [3.0, 2.0])
    np.testing.assert_array_equal(outputs.pred_val, [8.0])
    np.testing.assert_array_equal(outputs.pred_test, [11.0, 17.0])
    assert outputs.y_train is y_train
    assert outputs.y_val is y_val
    assert outputs.y_test is y_test
    assert outputs.target_mask_test is mask_test


def test_attention_run_configuration_is_stable() -> None:
    """Lock attention overrides and their copy behavior across file moves."""
    model_info: dict[str, object] = {
        "wrap": FixedPredictor,
        "params": {"epochs": 3},
        "loader_batch_size": 16,
    }
    seeded_info = enzymado.pipeline.run_config.apply_random_seed(model_info, 42)
    configured_info = enzymado.pipeline.run_config.apply_attention_overrides(
        seeded_info,
        {"lr": 0.0001, "gradient_clip_norm": 5.0},
    )

    assert configured_info["params"] == {
        "epochs": 3,
        "lr": 0.0001,
        "gradient_clip_norm": 5.0,
    }
    assert configured_info["loader_seed"] == 42
    assert model_info["params"] == {"epochs": 3}
    assert "loader_seed" not in model_info
    actual = enzymado.pipeline.run_config.apply_attention_overrides(model_info, None)
    assert actual is model_info


def test_model_prefix_selection_is_stable() -> None:
    """Lock standard and attention model-prefix behavior."""
    standard = enzymado.training.model_selection.select_models(
        "classification",
        "binary",
        "linear",
    )
    attention = enzymado.training.model_selection.select_attention_models(
        "classification",
        "binary",
        "attention_linear",
    )

    assert tuple(standard) == ("linear_classification",)
    assert tuple(attention) == ("attention_linear_classification",)


def test_embedding_mode_selection_is_stable() -> None:
    """Lock attention and no-attention filtering and input order."""
    model_map: dict[str, dict[str, object]] = {
        "mlp_classification": {},
        "linear_classification": {},
        "attention_mlp_classification": {},
        "attention_linear_classification": {},
    }

    no_attention = enzymado.training.model_selection.filter_by_embedding_modes(
        model_map,
        "binary_enzyme",
        {"no_attention"},
    )
    attention = enzymado.training.model_selection.filter_by_embedding_modes(
        model_map,
        "binary_enzyme",
        {"attention"},
    )

    assert tuple(no_attention) == (
        "mlp_classification",
        "linear_classification",
    )
    assert tuple(attention) == (
        "attention_mlp_classification",
        "attention_linear_classification",
    )
    assert (
        enzymado.training.model_selection.filter_by_embedding_modes(
            model_map,
            "binary_enzyme",
            set(),
        )
        is model_map
    )


def test_completed_model_selection_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lock completion checks to each model's canonical result directory."""
    model_map: dict[str, dict[str, object]] = {
        "mlp_classification": {},
        "linear_classification": {},
    }
    complete_dir = enzymado.artifacts.results_layout.build_results_run_dir(
        tmp_path,
        "similarity",
        "binary_enzyme",
        "ankh_base",
        "linear_classification",
        task_scope="properties",
    )

    def is_complete(run_dir: Path) -> bool:
        return run_dir == complete_dir

    monkeypatch.setattr(
        enzymado.artifacts.results_layout,
        "is_run_complete",
        is_complete,
    )

    completed = enzymado.training.model_selection.find_completed_models(
        model_map,
        tmp_path,
        "similarity",
        "binary_enzyme",
        "ankh_base",
        "properties",
    )

    assert completed == ["linear_classification"]
