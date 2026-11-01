"""Characterization tests for the benchmark command-line interface."""

from pathlib import Path

import pytest
import scripts.run_benchmark

import enzymado.data_loading.data_registry
import enzymado.pipeline.run_data
import enzymado.training.model_selection


def test_parser_defaults_are_stable() -> None:
    """Lock every default used when only the embeddings manifest is supplied."""
    args = scripts.run_benchmark._build_parser().parse_args(["embeddings.json"])

    assert vars(args) == {
        "embeddings_json": "embeddings.json",
        "output_dir": str(scripts.run_benchmark._DEFAULT_OUTPUT_DIR),
        "emb_name": None,
        "model_prefix": None,
        "task_name": [],
        "task_scope": [],
        "skip_label_kind": [],
        "allowed_ids": None,
        "residue_store_mode": "auto",
        "split_variant": None,
        "embedding_mode": [],
        "attach_probabilities": False,
        "overwrite_results": False,
        "deterministic": False,
        "attention_embeddings_json": None,
        "seed": 42,
        "attention_lr": None,
        "attention_gradient_clip_norm": None,
        "attention_batch_diagnostics": False,
        "diagnostic_loss_threshold": 100.0,
        "diagnostic_z_threshold": 20.0,
        "diagnostic_grad_norm_threshold": 1000.0,
    }


def test_completed_runs_are_successful_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A completed group succeeds without preparing data or fitting models."""
    config = enzymado.data_loading.data_registry.get_task_config("ph_opt")
    options = scripts.run_benchmark._BenchmarkOptions(
        output_dir=tmp_path,
        emb_name_filter=None,
        model_prefix_filter=None,
        task_name_filters=frozenset(),
        skip_label_kinds=frozenset(),
        allowed_ids=None,
        sequence_to_id={},
        residue_store_mode="auto",
        split_variant_filter="random",
        embedding_mode_filters=frozenset(),
        overwrite_results=False,
        random_seed=42,
        deterministic_algorithms=False,
        attach_probabilities=False,
    )
    model_map = {"linear_regressor": {}}

    def completed(*_args: object) -> list[str]:
        return list(model_map)

    def unexpected_prepare(**_kwargs: object) -> None:
        pytest.fail("Completed runs must not prepare data again.")

    monkeypatch.setattr(
        enzymado.training.model_selection, "find_completed_models", completed
    )
    monkeypatch.setattr(enzymado.pipeline.run_data, "prepare", unexpected_prepare)
    failures = scripts.run_benchmark._run_embedding_group(
        {"ankh_base": {"path": tmp_path / "unused", "dim": 2}},
        [("ph_opt", config)],
        options,
        heading="Completed test",
        model_map_fn=lambda *_args: model_map,
    )
    assert failures == []
