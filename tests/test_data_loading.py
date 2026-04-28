from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.data_loading.collators import ProteinResidueCollator, ProteinResidueConcatCollator
from src.data_loading.datasets import build_dataset_from_config
from src.data_loading.labels import FlatECLabelBuilder, StandardScaler1D
from src.data_loading.registry import TaskConfig
from src.data_loading.features import ResidueEmbeddingStore


class FakeH5Dataset:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype=np.float32)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return self.array.astype(dtype) if dtype is not None else self.array


class FakeH5File:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = {
            key: FakeH5Dataset(value)
            for key, value in mapping.items()
        }

    def __enter__(self) -> "FakeH5File":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def keys(self):
        return self.mapping.keys()

    def __getitem__(self, key: str) -> FakeH5Dataset:
        return self.mapping[key]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class DataLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_protein_global_dataset_uses_split_column_and_padding_mask(self) -> None:
        dataset_path = self.root / "solubility.csv"
        write_csv(
            dataset_path,
            ["dataset_id", "protein_sequence", "solubility", "split"],
            [
                {
                    "dataset_id": "sample_train",
                    "protein_sequence": "AAAA",
                    "solubility": "1",
                    "split": "train",
                },
                {
                    "dataset_id": "sample_val",
                    "protein_sequence": "AA",
                    "solubility": "0",
                    "split": "val",
                },
            ],
        )
        config = TaskConfig(
            name="solubility_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="solubility",
            sample_id_candidates=("dataset_id",),
        )
        store = ResidueEmbeddingStore(
            by_sequence={
                "AAAA": np.arange(8, dtype=np.float32).reshape(4, 2),
                "AA": np.arange(4, dtype=np.float32).reshape(2, 2),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        self.assertEqual(len(bundle.dataset), 1)
        sample = bundle.dataset[0]
        self.assertEqual(sample["sample_id"], "sample_train")
        np.testing.assert_array_equal(
            sample["protein_emb"],
            np.arange(8, dtype=np.float32).reshape(4, 2),
        )
        batch = bundle.collator([sample])
        self.assertEqual(batch.protein_emb.shape, (1, 4, 2))
        np.testing.assert_array_equal(batch.protein_mask, np.array([[True, True, True, True]]))
        np.testing.assert_array_equal(batch.target, np.array([1.0], dtype=np.float32))

    def test_residue_embedding_store_loads_h5_by_id_and_sequence(self) -> None:
        class FakeH5Module:
            def __init__(self, file_mapping: dict[str, dict[str, np.ndarray]]) -> None:
                self.file_mapping = file_mapping

            def File(self, path: str, mode: str) -> FakeH5File:
                return FakeH5File(self.file_mapping[path])

        file_mapping = {
            str(self.root / "id_embeddings.h5"): {
                "P1": np.ones((4, 2), dtype=np.float32),
            },
            str(self.root / "seq_embeddings.h5"): {
                "AAAA": np.full((4, 2), 3.0, dtype=np.float32),
            },
        }
        fake_h5py = types.SimpleNamespace(
            File=FakeH5Module(file_mapping).File,
        )

        with mock.patch.dict(sys.modules, {"h5py": fake_h5py}):
            id_store = ResidueEmbeddingStore.from_h5(self.root / "id_embeddings.h5", key_kind="id")
            seq_store = ResidueEmbeddingStore.from_h5(
                self.root / "seq_embeddings.h5",
                key_kind="sequence",
            )
            merged_store = ResidueEmbeddingStore.from_h5_files(
                id_h5_path=self.root / "id_embeddings.h5",
                sequence_h5_path=self.root / "seq_embeddings.h5",
            )

        np.testing.assert_array_equal(id_store.lookup_id("P1"), np.ones((4, 2), dtype=np.float32))
        np.testing.assert_array_equal(
            seq_store.lookup_sequence("AAAA"),
            np.full((4, 2), 3.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            merged_store.lookup_row({"uniprot_id": "P1", "protein_sequence": "AAAA"}),
            np.full((4, 2), 3.0, dtype=np.float32),
        )

    def test_regression_scaler_fits_only_on_train_rows(self) -> None:
        dataset_path = self.root / "phopt_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "phopt_median", "split"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "phopt_median": "4.0", "split": "train"},
                {"uniprot_id": "P2", "protein_sequence": "AAAAA", "phopt_median": "8.0", "split": "train"},
                {"uniprot_id": "P3", "protein_sequence": "AAA", "phopt_median": "10.0", "split": "val"},
            ],
        )
        config = TaskConfig(
            name="phopt_test",
            batch_kind="protein_global",
            label_kind="scalar_regression",
            dataset_paths={"random": dataset_path},
            target_column="phopt_median",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((5, 2), dtype=np.float32),
                "P3": np.ones((3, 2), dtype=np.float32),
            }
        )

        train_bundle = build_dataset_from_config(config, store, split_name="train")
        val_bundle = build_dataset_from_config(config, store, split_name="val")
        scaler = StandardScaler1D.from_rows(train_bundle.rows, "phopt_median")
        np.testing.assert_allclose(scaler.transform([4.0, 8.0]), np.array([-1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(scaler.transform([10.0]), np.array([2.0], dtype=np.float32))
        self.assertEqual(len(val_bundle.dataset), 1)

    def test_mutation_delta_subset_loader_resolves_sequence_ids_for_wildtype(self) -> None:
        dataset_path = self.root / "ddg.csv"
        sequence_index_path = self.root / "protein_sequences_unique.csv"
        embeddings_path = self.root / "embeddings.h5"

        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "wildtype_sequence", "mutation", "ddg"],
            [
                {
                    "uniprot_id": "row_specific_id",
                    "protein_sequence": "AAAA",
                    "wildtype_sequence": "BBBB",
                    "mutation": "A1B",
                    "ddg": "1.5",
                }
            ],
        )
        write_csv(
            sequence_index_path,
            ["sequence_id", "protein_sequence"],
            [
                {"sequence_id": "representative_mutant", "protein_sequence": "AAAA"},
                {
                    "sequence_id": "representative_wildtype__wildtype_sequence",
                    "protein_sequence": "BBBB",
                },
            ],
        )

        class FakeH5Module:
            def __init__(self, file_mapping: dict[str, dict[str, np.ndarray]]) -> None:
                self.file_mapping = file_mapping

            def File(self, path: str, mode: str) -> FakeH5File:
                return FakeH5File(self.file_mapping[path])

        fake_h5py = types.SimpleNamespace(
            File=FakeH5Module(
                {
                    str(embeddings_path): {
                        "representative_mutant": np.ones((4, 2), dtype=np.float32),
                        "representative_wildtype__wildtype_sequence": np.full((4, 2), 3.0, dtype=np.float32),
                    }
                }
            ).File,
        )

        with mock.patch.dict(sys.modules, {"h5py": fake_h5py}):
            store = ResidueEmbeddingStore.from_h5_subset(
                embeddings_path,
                needed_ids={"row_specific_id"},
                needed_sequences={"AAAA", "BBBB"},
                sequence_index_path=sequence_index_path,
            )

        config = TaskConfig(
            name="ddg_test",
            batch_kind="mutation_delta_global",
            label_kind="scalar_regression",
            dataset_paths={"random": dataset_path},
            target_column="ddg",
            sample_id_candidates=("uniprot_id",),
            wildtype_sequence_column="wildtype_sequence",
        )

        bundle = build_dataset_from_config(config, store)
        self.assertEqual(len(bundle.dataset), 1)
        sample = bundle.dataset[0]

        np.testing.assert_array_equal(
            sample["protein_delta_emb"],
            np.full((4, 2), -2.0, dtype=np.float32),
        )
        self.assertEqual(sample["sample_id"], "row_specific_id")

    def test_hierarchical_ec_uses_train_vocabulary_and_masks_incomplete_levels(self) -> None:
        dataset_path = self.root / "ec_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "ec_number", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "ec_number": "1.1.1.1; 2.2.-.-",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AAA",
                    "ec_number": "3.-.-.-; 9.9.9.9",
                    "split": "val",
                },
            ],
        )
        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_paths={"random": dataset_path},
            target_column="ec_number",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((3, 2), dtype=np.float32),
            }
        )

        train_bundle = build_dataset_from_config(config, store, split_name="train")
        self.assertEqual(
            train_bundle.label_builder.vocabulary_by_level,
            {
                "level_1": ("1", "2"),
                "level_2": ("1.1", "2.2"),
                "level_3": ("1.1.1",),
                "level_4": ("1.1.1.1",),
            },
        )
        with self.assertRaisesRegex(ValueError, "requires a vocabulary fit on train rows"):
            build_dataset_from_config(config, store, split_name="val")

        val_bundle = build_dataset_from_config(
            config,
            store,
            split_name="val",
            ec_vocabulary=train_bundle.label_builder.vocabulary_by_level,
        )
        sample = val_bundle.dataset[0]
        np.testing.assert_array_equal(sample["target"]["level_1"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(sample["target"]["level_2"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(sample["target"]["level_3"], np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(sample["target"]["level_4"], np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(sample["target_mask"]["level_1"], np.array(False))
        np.testing.assert_array_equal(sample["target_mask"]["level_2"], np.array(False))
        np.testing.assert_array_equal(sample["target_mask"]["level_3"], np.array(False))
        np.testing.assert_array_equal(sample["target_mask"]["level_4"], np.array(False))

        train_sample = train_bundle.dataset[0]
        np.testing.assert_array_equal(
            train_sample["target"]["level_1"],
            np.array([1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            train_sample["target"]["level_2"],
            np.array([1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            train_sample["target"]["level_3"],
            np.array([1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            train_sample["target"]["level_4"],
            np.array([1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(train_sample["target_mask"]["level_1"], np.array(True))
        np.testing.assert_array_equal(train_sample["target_mask"]["level_2"], np.array(True))
        np.testing.assert_array_equal(train_sample["target_mask"]["level_3"], np.array(True))
        np.testing.assert_array_equal(train_sample["target_mask"]["level_4"], np.array(True))

        batch = train_bundle.collator([train_sample])
        np.testing.assert_array_equal(
            batch.target["level_1"],
            np.array([[1.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            batch.target_mask["level_4"],
            np.array([True]),
        )

    def test_protein_ligand_dataset_builds_ligand_features(self) -> None:
        dataset_path = self.root / "binding_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "smiles", "binding", "split"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "smiles": "CCO", "binding": "1", "split": "train"},
                {"uniprot_id": "P2", "protein_sequence": "AA", "smiles": "N", "binding": "0", "split": "train"},
            ],
        )
        config = TaskConfig(
            name="binding_test",
            batch_kind="protein_ligand_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="binding",
            ligand_column="smiles",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 3), dtype=np.float32),
                "P2": np.ones((2, 3), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(
            config,
            store,
            split_name="train",
            ligand_featurizer=lambda smiles: np.array([len(smiles), smiles.count("C")], dtype=np.float32),
        )
        samples = [bundle.dataset[0], bundle.dataset[1]]
        batch = bundle.collator(samples)
        self.assertEqual(batch.ligand_feat.shape, (2, 2))
        np.testing.assert_array_equal(
            batch.ligand_feat,
            np.array([[3.0, 2.0], [1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(batch.protein_emb.shape, (2, 4, 3))
        np.testing.assert_array_equal(
            batch.protein_mask,
            np.array([[True, True, True, True], [True, True, False, False]]),
        )

    def test_flat_ec_label_builder_marks_unknown_descendants_with_nan(self) -> None:
        builder = FlatECLabelBuilder().fit(
            [
                {"ec_number": "1.2.3.4"},
                {"ec_number": "1.2.5.6"},
                {"ec_number": "2.1.-.-"},
            ]
        )

        self.assertEqual(
            builder.flat_vocabulary,
            ("1", "2", "1.2", "2.1", "1.2.3", "1.2.5", "1.2.3.4", "1.2.5.6"),
        )

        payload = builder.build({"ec_number": "1.2.-.-"})
        np.testing.assert_array_equal(
            payload["target_mask"],
            np.array([True, True, True, True, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            np.nan_to_num(payload["target"], nan=-1.0),
            np.array([1.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        )

    def test_flat_ec_label_builder_allows_complete_label_to_override_unknown_branch(self) -> None:
        builder = FlatECLabelBuilder().fit(
            [
                {"ec_number": "1.2.3.4"},
                {"ec_number": "1.2.5.6"},
            ]
        )

        payload = builder.build({"ec_number": "1.2.-.-;1.2.3.4"})
        np.testing.assert_array_equal(
            payload["target_mask"],
            np.array([True, True, True, False, True, False], dtype=bool),
        )
        np.testing.assert_array_equal(
            np.nan_to_num(payload["target"], nan=-1.0),
            np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        )

    def test_mutation_delta_dataset_builds_mut_minus_wt(self) -> None:
        dataset_path = self.root / "ddg_random_standardized.csv"
        write_csv(
            dataset_path,
            ["dataset_id", "protein_sequence", "wildtype_sequence", "ddg", "split"],
            [
                {
                    "dataset_id": "row1",
                    "protein_sequence": "ABCD",
                    "wildtype_sequence": "ABCE",
                    "ddg": "-1.5",
                    "split": "train",
                }
            ],
        )
        config = TaskConfig(
            name="ddg_test",
            batch_kind="mutation_delta_global",
            label_kind="scalar_regression",
            dataset_paths={"random": dataset_path},
            target_column="ddg",
            wildtype_sequence_column="wildtype_sequence",
            sample_id_candidates=("dataset_id",),
            split_id_candidates=("dataset_id",),
        )
        store = ResidueEmbeddingStore(
            by_sequence={
                "ABCD": np.array([[2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]], dtype=np.float32),
                "ABCE": np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        sample = bundle.dataset[0]
        np.testing.assert_array_equal(
            sample["protein_delta_emb"],
            np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(float(sample["target"]), -1.5)

    def test_residue_dataset_creates_binary_vectors_and_masks(self) -> None:
        dataset_path = self.root / "active_sites_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "active_site": "[1, 3]", "split": "train"},
                {"uniprot_id": "P2", "protein_sequence": "AA", "active_site": "[2]", "split": "train"},
            ],
        )
        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((2, 2), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        batch = bundle.collator([bundle.dataset[0], bundle.dataset[1]])
        np.testing.assert_array_equal(
            batch.target_per_residue,
            np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            batch.target_mask,
            np.array([[True, True, True, True], [True, True, False, False]]),
        )

    def test_protein_residue_concat_collator_concatenates_residue_batches(self) -> None:
        collator = ProteinResidueConcatCollator()

        batch = collator(
            [
                {
                    "sample_id": "P1",
                    "protein_emb": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "target_per_residue": np.asarray([1.0, 0.0], dtype=np.float32),
                    "target_mask": np.asarray([True, True], dtype=bool),
                },
                {
                    "sample_id": "P2",
                    "protein_emb": np.asarray([[2.0, 2.0]], dtype=np.float32),
                    "target_per_residue": np.asarray([0.0], dtype=np.float32),
                    "target_mask": np.asarray([True], dtype=bool),
                },
            ]
        )

        self.assertEqual(batch.sample_id, ["P1", "P2"])
        np.testing.assert_array_equal(batch.lengths, np.asarray([2, 1], dtype=np.int32))
        np.testing.assert_array_equal(
            batch.protein_emb,
            np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            batch.target_per_residue,
            np.asarray([1, 0, 0], dtype=np.int8),
        )
        self.assertEqual(batch.target_per_residue.dtype, np.int8)

    def test_protein_residue_concat_collator_preserves_bf16_embeddings(self) -> None:
        import torch

        collator = ProteinResidueConcatCollator()

        batch = collator(
            [
                {
                    "sample_id": "P1",
                    "protein_emb": torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0]],
                        dtype=torch.bfloat16,
                    ).view(torch.int16).numpy(),
                    "target_per_residue": np.asarray([1.0, 0.0], dtype=np.float32),
                    "target_mask": np.asarray([True, True], dtype=bool),
                },
            ]
        )

        self.assertTrue(torch.is_tensor(batch.protein_emb))
        self.assertEqual(batch.protein_emb.dtype, torch.bfloat16)
        self.assertEqual(batch.target_per_residue.dtype, np.int8)

    def test_missing_policy_skip_filters_rows_with_missing_embeddings(self) -> None:
        dataset_path = self.root / "solubility_skip.csv"
        write_csv(
            dataset_path,
            ["dataset_id", "protein_sequence", "solubility", "split"],
            [
                {
                    "dataset_id": "keep_me",
                    "protein_sequence": "AAAA",
                    "solubility": "1",
                    "split": "train",
                },
                {
                    "dataset_id": "drop_me",
                    "protein_sequence": "BBBB",
                    "solubility": "0",
                    "split": "train",
                },
            ],
        )
        config = TaskConfig(
            name="solubility_skip",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="solubility",
            sample_id_candidates=("dataset_id",),
        )
        store = ResidueEmbeddingStore(by_sequence={"AAAA": np.ones((4, 2), dtype=np.float32)})

        bundle = build_dataset_from_config(
            config,
            store,
            split_name="train",
            missing_policy="skip",
        )
        self.assertEqual(len(bundle.dataset), 1)
        self.assertEqual(bundle.dataset.skipped_rows, 1)

    def test_missing_policy_skip_filters_residue_rows_with_length_mismatch(self) -> None:
        dataset_path = self.root / "active_sites_skip_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {"uniprot_id": "keep_me", "protein_sequence": "AAAA", "active_site": "[1, 3]", "split": "train"},
                {"uniprot_id": "drop_me", "protein_sequence": "AAAAA", "active_site": "[2]", "split": "train"},
            ],
        )
        config = TaskConfig(
            name="active_sites_skip",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "keep_me": np.ones((4, 2), dtype=np.float32),
                "drop_me": np.ones((3, 2), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(
            config,
            store,
            split_name="train",
            missing_policy="skip",
        )
        self.assertEqual(len(bundle.dataset), 1)
        self.assertEqual(bundle.dataset.skipped_rows, 1)
        self.assertEqual(bundle.dataset[0]["sample_id"], "keep_me")

    def test_protein_residue_dataset_build_uses_length_lookup_before_materializing(self) -> None:
        dataset_path = self.root / "active_sites_length_lookup_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "active_site": "[1, 3]", "split": "train"},
                {"uniprot_id": "P2", "protein_sequence": "AA", "active_site": "[2]", "split": "train"},
            ],
        )
        config = TaskConfig(
            name="active_sites_length_lookup",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
            split_id_candidates=("uniprot_id",),
        )

        class TrackingResidueEmbeddingStore(ResidueEmbeddingStore):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.lookup_row_calls = 0
                self.lookup_row_length_calls = 0

            def lookup_row(self, row, id_columns=("uniprot_id", "dataset_id"), sequence_column="protein_sequence"):
                self.lookup_row_calls += 1
                return super().lookup_row(
                    row,
                    id_columns=id_columns,
                    sequence_column=sequence_column,
                )

            def lookup_row_length(
                self,
                row,
                id_columns=("uniprot_id", "dataset_id"),
                sequence_column="protein_sequence",
            ):
                self.lookup_row_length_calls += 1
                return super().lookup_row_length(
                    row,
                    id_columns=id_columns,
                    sequence_column=sequence_column,
                )

        store = TrackingResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((2, 2), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        self.assertEqual(store.lookup_row_calls, 0)
        self.assertEqual(store.lookup_row_length_calls, 2)

        sample = bundle.dataset[0]
        self.assertEqual(sample["sample_id"], "P1")
        self.assertEqual(store.lookup_row_calls, 1)

    def test_protein_residue_collator_raises_clear_error_on_length_mismatch(self) -> None:
        collator = ProteinResidueCollator()

        with self.assertRaisesRegex(
            ValueError,
            "Protein embedding and residue target length mismatch for sample 'bad_sample'",
        ):
            collator(
                [
                    {
                        "sample_id": "bad_sample",
                        "protein_emb": np.ones((2, 2), dtype=np.float32),
                        "target_per_residue": np.array([1.0, 0.0, 1.0], dtype=np.float32),
                        "target_mask": np.array([True, True, True], dtype=bool),
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
