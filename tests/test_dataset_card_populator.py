"""Characterization tests for dataset-card statistics and output schemas."""

import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import safetensors.torch
import torch

from enzymado import paths_and_constants
import enzymado.dataset_card.clustering as dataset_clustering
import enzymado.dataset_card.dataset_card_populator as dataset_card_populator
import enzymado.dataset_card.statistics as dataset_statistics
import enzymado.dataset_card.umap_plots as umap_plots

_P = paths_and_constants.Paths
_C = paths_and_constants.Constants


def _property_frame() -> pd.DataFrame:
    """Return a small property table covering labels and sequence statistics."""
    return pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3", "P4"],
            "protein_sequence": ["AAAA", "ACDX", "ACDEFG", None],
            "phopt_median": [1.0, 1.0, 2.0, 3.0],
            "extra": [1.0, np.nan, 3.0, 4.0],
        }
    )


def test_distribution_and_split_statistics_are_stable(tmp_path) -> None:
    """Lock label summaries, split naming, and sequence statistics."""
    populator = dataset_card_populator.DatasetCardPopulator(tmp_path)
    base_df = _property_frame()
    split_path = tmp_path / "splits_random.csv"
    pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3", "P4"],
            "split": ["train", "train", "val", "test"],
        }
    ).to_csv(split_path, index=False)

    categorical = dataset_statistics.distribution(
        pd.Series(["active", "active", "inactive", None])
    )
    continuous = dataset_statistics.distribution(pd.Series(range(1, 16)))
    stats = populator.calc_stats(
        base_df,
        "ph_opt",
        {"splits_random": split_path},
    )

    assert categorical == {
        "total_valid_labels": 3,
        "classes": {
            "active": {"count": 2, "ratio": 0.6667},
            "inactive": {"count": 1, "ratio": 0.3333},
        },
    }
    assert continuous == {
        "total_valid_labels": 15,
        "mean": 8.0,
        "std": 4.4721,
        "min": 1.0,
        "max": 15.0,
        "median": 8.0,
    }
    assert stats["sequence_length_summary"] == {
        "min": 4,
        "max": 6,
        "mean": 4.67,
        "median": 4.0,
        "p95": 5.8,
        "count_greater_than_1024": 0,
        "contains_non_standard_aas": 1,
    }
    assert stats["split_sizes"] == {"splits_random": {"train": 2, "val": 1, "test": 1}}
    label_distribution = stats["label_distribution"]
    assert label_distribution["train"]["splits_random"]["classes"] == {
        "1.0": {"count": 2, "ratio": 1.0}
    }
    assert label_distribution["validation"]["splits_random"]["classes"] == {
        "2.0": {"count": 1, "ratio": 1.0}
    }
    assert label_distribution["test"]["splits_random"]["classes"] == {
        "3.0": {"count": 1, "ratio": 1.0}
    }


def test_run_writes_the_existing_dataset_card_schema(tmp_path, monkeypatch) -> None:
    """Lock the orchestrator's card fields without invoking MMseqs2 or UMAP."""
    property_dir = tmp_path / "active_sites"
    property_dir.mkdir()
    template = {
        "amount_of_removed_data": {},
        "statistics": {
            "nans_dups": {},
            "label_distribution": {},
        },
        "additional_important": {"quality_checks": {}},
    }
    (property_dir / "dataset_card_template.json").write_text(
        json.dumps(template),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2"],
            "protein_sequence": ["AAAA", "ACDE"],
            "active_site": [0, 1],
        }
    ).to_csv(property_dir / "active_sites.csv", index=False)

    monkeypatch.setattr(
        dataset_card_populator.DatasetCardPopulator,
        "run_clustering",
        lambda _self, _base_df, _prop_name: None,
    )
    # This test exercises the card schema, not clustering or UMAPs, and the
    # test environment has neither mmseqs nor umap-learn. Saying so explicitly
    # is the point: a card built without those sections is now a declared
    # choice rather than a silent omission.
    dataset_card_populator.DatasetCardPopulator(
        tmp_path,
        allow_incomplete=True,
        figures_dir=tmp_path / "figures",
    ).run()

    card = json.loads((property_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["property_name"] == "active_sites"
    assert card["dataset_id"] == "active_sites"
    assert card["amount_of_removed_data"] == {"datapoints_after_cleaning": 2}
    assert card["statistics"]["nans_dups"]["duplicate_rows_full"] == 0
    assert card["additional_important"]["quality_checks"] == {
        "schema_validated": True,
        "split_leakage_checked": {},
        "label_format_validated": True,
    }


def test_shared_paths_and_constants_keep_the_current_layout() -> None:
    """Lock path resolution and public column constants."""
    paths = _P()

    assert paths.data_dir == paths.project_root / "data"
    assert paths.data_processed_properties == (
        paths.project_root / "data" / "processed" / "properties"
    )
    assert paths.results_figures_dir == paths.project_root / "results" / "figures"
    assert _C.dataset_id == "dataset_id"
    assert _C.protein_sequence == "protein_sequence"


def test_dataset_card_cli_matches_recorded_identity_threshold(tmp_path) -> None:
    """Keep both programmatic and CLI defaults aligned with existing cards."""
    parser = dataset_card_populator.build_argument_parser()
    defaults = parser.parse_args([])
    selected = parser.parse_args(
        [
            "--base_dir",
            "processed",
            "--emb_model",
            "esm2_650M",
            "--embeddings_json",
            "registry.json",
            "--seq_identity",
            "0.4",
        ]
    )

    assert vars(defaults) == {
        "base_dir": "data/processed/properties",
        "emb_model": None,
        "embeddings_json": "embeddings_models_output.json",
        "seq_identity": 0.5,
        "figures_dir": "results/figures",
        "allow_incomplete": False,
    }
    assert vars(selected) == {
        "base_dir": "processed",
        "emb_model": "esm2_650M",
        "embeddings_json": "registry.json",
        "seq_identity": 0.4,
        "figures_dir": "results/figures",
        "allow_incomplete": False,
    }
    assert dataset_card_populator.DatasetCardPopulator(tmp_path).seq_identity == 0.5


def test_embedding_loading_keeps_key_order_pooling_and_missing_rows(tmp_path) -> None:
    """Lock safetensors lookup order, mean pooling, and retained row indices."""
    embedding_path = tmp_path / "embeddings.safetensors"
    tensors = {
        f"P{index}": torch.tensor(
            [[index, index + 1], [index + 2, index + 3]],
            dtype=torch.float32,
        )
        for index in range(5)
    }
    safetensors.torch.save_file(tensors, embedding_path)
    populator = dataset_card_populator.DatasetCardPopulator(tmp_path)

    embeddings, valid_indices = populator._load_embeddings(
        ["AAAA"] * 6,
        ["P0", "P1", "P2", "P3", "P4", "missing"],
        embedding_path,
    )

    np.testing.assert_array_equal(
        embeddings,
        np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            dtype=np.float32,
        ),
    )
    assert valid_indices == [0, 1, 2, 3, 4]


def test_umap_figures_keep_seed_parameters_and_output_names(
    tmp_path,
    monkeypatch,
) -> None:
    """Exercise both UMAP figures with a deterministic stand-in reducer."""
    reducer_parameters = {}

    class FixedReducer:
        def __init__(self, **kwargs) -> None:
            reducer_parameters.update(kwargs)

        def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
            indices = np.arange(len(embeddings), dtype=np.float32)
            return np.column_stack([indices, -indices])

    monkeypatch.setattr(
        umap_plots,
        "umap_lib",
        types.SimpleNamespace(UMAP=FixedReducer),
    )
    populator = dataset_card_populator.DatasetCardPopulator(tmp_path)
    monkeypatch.setattr(
        populator,
        "_load_embeddings",
        lambda _sequences, _ids, _path: (
            np.ones((5, 2), dtype=np.float32),
            list(range(5)),
        ),
    )
    base_df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3", "P4", "P5"],
            "protein_sequence": ["AAAA", "CCCC", "DDDD", "EEEE", "FFFF"],
            "active_site": [0, 1, 0, 1, 0],
        }
    )
    split_path = tmp_path / "splits_random.csv"
    pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3", "P4", "P5"],
            "split": ["train", "train", "val", "test", "test"],
        }
    ).to_csv(split_path, index=False)
    output_dir = tmp_path / "figures"

    result = populator.generate_umap(
        base_df,
        "active_sites",
        {"splits_random": split_path},
        tmp_path / "unused.safetensors",
        "fixed_embedding",
        output_dir,
    )

    assert reducer_parameters == {
        "n_components": 2,
        "random_state": 42,
        "verbose": False,
    }
    assert result == {
        "embedding_model": "fixed_embedding",
        "label_colored": str(output_dir / "active_sites_fixed_embedding_labels.png"),
        "split_colored": str(output_dir / "active_sites_fixed_embedding_splits.png"),
    }
    assert Path(result["label_colored"]).is_file()
    assert Path(result["split_colored"]).is_file()


def test_mmseqs_commands_and_cluster_summary_are_stable(
    tmp_path,
    monkeypatch,
) -> None:
    """Lock the MMseqs2 command sequence and derived cluster statistics."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs) -> types.SimpleNamespace:
        commands.append(command)
        if command[1] == "createtsv":
            Path(command[-1]).write_text(
                "seq0\tseq0\nseq0\tseq1\nseq2\tseq2\n",
                encoding="utf-8",
            )
        return types.SimpleNamespace(stderr="")

    monkeypatch.setattr(dataset_clustering.shutil, "which", lambda _name: "mmseqs")
    monkeypatch.setattr(dataset_clustering.subprocess, "run", fake_run)
    populator = dataset_card_populator.DatasetCardPopulator(
        tmp_path,
        seq_identity=0.3,
    )
    base_df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3"],
            "protein_sequence": ["AAAA", "CCCC", "DDDD"],
        }
    )

    stats = populator.run_clustering(base_df, "active_sites")

    assert [command[1] for command in commands] == [
        "createdb",
        "cluster",
        "createtsv",
    ]
    assert commands[1][-6:] == [
        "--min-seq-id",
        "0.3",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
    ]
    assert stats == {
        "cluster_method": "mmseqs2",
        "cluster_parameters": {
            "min_sequence_identity": 0.3,
            "coverage_mode": 0,
            "coverage_threshold": 0.8,
        },
        "total_sequences": 3,
        "number_of_clusters": 2,
        "singleton_clusters": 1,
        "largest_cluster_size": 2,
        "ratio_unique_variants": 0.6667,
        "_cluster_sizes": [2, 1],
    }


def test_cluster_figures_keep_the_existing_names(tmp_path) -> None:
    """Verify that deterministic cluster figures use the existing output paths."""
    populator = dataset_card_populator.DatasetCardPopulator(
        tmp_path,
        seq_identity=0.3,
    )
    cluster_stats = {"_cluster_sizes": [2, 1]}

    diversity_path = populator.generate_cluster_diversity_plots(
        cluster_stats,
        "active_sites",
        tmp_path,
    )

    assert diversity_path == str(tmp_path / "active_sites_cluster_diversity.png")
    assert Path(diversity_path).is_file()
