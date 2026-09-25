from __future__ import annotations

import csv
import importlib.util
import lzma
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from enzymado.data_loading.collators import ProteinResidueConcatCollator
from enzymado.data_loading.data_registry import TaskConfig
from enzymado.data_loading.datasets import build_dataset_from_config
from enzymado.data_loading.features import ResidueEmbeddingStore

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for training tests")
class StructuredTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_training_seed_reproducibly_resets_all_rngs(self) -> None:
        import random

        import torch

        from enzymado.models.common import seed_random_generators

        seed_random_generators(42)
        first = (
            random.random(),
            np.random.random(3),
            torch.rand(3),
        )
        seed_random_generators(42)
        second = (
            random.random(),
            np.random.random(3),
            torch.rand(3),
        )

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)

    def test_model_seed_is_propagated_without_mutating_registry(self) -> None:
        import enzymado.pipeline.run_config

        class TorchWrapper:
            pass

        class XGBWrapper:
            pass

        torch_info = {"wrap": TorchWrapper, "params": {"epochs": 3}}
        xgb_info = {"wrap": XGBWrapper, "params": {"n_estimators": 10}}

        seeded_torch = enzymado.pipeline.run_config.apply_random_seed(torch_info, 42)
        seeded_xgb = enzymado.pipeline.run_config.apply_random_seed(xgb_info, 42)

        self.assertEqual(seeded_torch["loader_seed"], 42)
        self.assertNotIn("random_state", seeded_torch["params"])
        self.assertEqual(seeded_xgb["params"]["random_state"], 42)
        self.assertNotIn("loader_seed", torch_info)
        self.assertNotIn("random_state", xgb_info["params"])

    def test_early_stopping_selects_trained_epoch_over_better_initialization(
        self,
    ) -> None:
        import torch

        from enzymado.models.common import run_training_with_early_stopping

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        callback_epochs = []

        def train_epoch() -> None:
            with torch.no_grad():
                model.weight.add_(1.0)

        def val_loss() -> float:
            return float(model.weight.square().item())

        best_epoch, best_val_loss = run_training_with_early_stopping(
            model,
            epochs=3,
            train_epoch_fn=train_epoch,
            val_loss_fn=val_loss,
            patience=1,
            log_every=100,
            epoch_callback=lambda epoch, *_: callback_epochs.append(epoch),
        )

        self.assertEqual(best_epoch, 1)
        self.assertEqual(best_val_loss, 1.0)
        self.assertEqual(callback_epochs, [0, 1, 2])
        torch.testing.assert_close(
            model.weight,
            torch.ones_like(model.weight),
            rtol=0.0,
            atol=0.0,
        )

    def test_ec_multihead_training_and_prediction_serialization(self) -> None:
        from safetensors.torch import load as load_tensors

        from enzymado.artifacts.serialization import make_safetensors_predictions_file
        from enzymado.data_loading.stores import get_target_payload
        from enzymado.training.torch_structured_wrappers import (
            ECMultiHeadClassifierTorch,
        )

        dataset_path = self.root / "ec_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "ec_number", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "ec_number": "1.1.1.1",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AAA",
                    "ec_number": "2.2.-.-",
                    "split": "train",
                },
            ],
        )

        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_paths={"random": dataset_path},
            target_column="ec_number",
            index_columns=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 3), dtype=np.float32),
                "P2": np.full((3, 3), 2.0, dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        x = np.stack(
            [
                np.asarray(bundle.dataset[i]["protein_emb"], dtype=np.float32).mean(
                    axis=0
                )
                for i in range(len(bundle.dataset))
            ]
        )
        y_true, target_mask = get_target_payload(bundle)
        model = ECMultiHeadClassifierTorch(
            hidden_dims=(8,),
            lr=1e-2,
            epochs=3,
            early_stopping_patience=1,
        )
        model.fit(
            x, y_true, target_mask, x_val=x, y_val=y_true, target_mask_val=target_mask
        )
        preds = model.predict(x)

        self.assertEqual(set(preds), {"level_1", "level_2", "level_3", "level_4"})
        self.assertEqual(preds["level_1"].shape, y_true["level_1"].shape)
        self.assertEqual(preds["level_4"].shape, y_true["level_4"].shape)
        self.assertEqual(target_mask["level_2"].shape, (2, 2))
        self.assertIsNotNone(model.best_epoch_)

        run_dir = self.root / "ec_outputs"
        run_dir.mkdir()
        pred_path = make_safetensors_predictions_file(
            y_true, preds, run_dir, "train", target_mask
        )
        payload = load_tensors(lzma.open(pred_path, "rb").read())
        self.assertIn("y_true_level_1", payload)
        self.assertIn("predictions_level_4", payload)
        self.assertIn("target_mask_level_3", payload)

    def test_ec_multihead_pooled_training_and_prediction(self) -> None:
        from enzymado.training.torch_structured_wrappers import (
            ECMultiHeadClassifierTorch,
        )

        x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        y = {
            "level_1": np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
            "level_2": np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
            "level_3": np.asarray([[1], [0], [1], [0]], dtype=np.float32),
            "level_4": np.asarray([[1], [0], [1], [0]], dtype=np.float32),
        }
        target_mask = {
            level_name: np.ones_like(target, dtype=bool)
            for level_name, target in y.items()
        }
        target_mask["level_2"][1, 1] = False

        model = ECMultiHeadClassifierTorch(
            hidden_dims=(8,),
            lr=1e-2,
            epochs=3,
            batch_size=2,
            early_stopping_patience=1,
        )
        model.fit(
            x,
            y,
            target_mask,
            x_val=x,
            y_val=y,
            target_mask_val=target_mask,
        )
        preds = model.predict(x)

        self.assertEqual(set(preds), {"level_1", "level_2", "level_3", "level_4"})
        self.assertEqual(preds["level_1"].shape, y["level_1"].shape)
        self.assertEqual(preds["level_4"].shape, y["level_4"].shape)
        self.assertEqual(target_mask["level_2"].shape, y["level_2"].shape)
        self.assertIsNotNone(model.best_epoch_)

    def test_ec_pooled_route_uses_vocabulary_from_unfiltered_train(self) -> None:
        import torch

        from enzymado.data_loading.task_data import (
            _build_pooled_data,
            _load_and_filter_rows,
        )

        dataset_path = self.root / "ec_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "ec_number", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "ec_number": "1.1.1.1",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "BBBB",
                    "ec_number": "2.2.2.2",
                    "split": "train",
                },
                {
                    "uniprot_id": "P3",
                    "protein_sequence": "CCCC",
                    "ec_number": "1.1.1.1",
                    "split": "val",
                },
                {
                    "uniprot_id": "P4",
                    "protein_sequence": "DDDD",
                    "ec_number": "1.1.1.1",
                    "split": "test",
                },
            ],
        )
        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_paths={"random": dataset_path},
            target_column="ec_number",
            index_columns=("uniprot_id",),
        )
        rows = _load_and_filter_rows(
            config,
            split_variant="random",
            allowed_ids={"P1", "P3", "P4"},
            sequence_to_id={},
        )

        self.assertEqual(len(rows.train_rows), 1)
        self.assertEqual(
            rows.ec_vocabulary["level_4"],
            ("1.1.1.1", "2.2.2.2"),
        )

        embeddings = {
            protein_id: torch.ones(3, dtype=torch.float32)
            for protein_id in ("P1", "P3", "P4")
        }
        with mock.patch(
            "enzymado.data_loading.task_data.load_pooled_embedding_subset",
            return_value=(embeddings, {}),
        ):
            task_data = _build_pooled_data(
                config,
                "ec_test",
                "unused.safetensors",
                "random",
                rows,
                embedding_dim=3,
            )

        self.assertEqual(task_data.y_train["level_4"].shape, (1, 2))
        self.assertEqual(task_data.y_train_flat.shape, (1, 8))

    def test_ec_loader_route_rejects_missing_filtered_samples(self) -> None:
        from enzymado.data_loading.labels import HierarchicalECLabelBuilder
        from enzymado.data_loading.task_data import _build_loader_data, _TaskRows

        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="ec_number",
            index_columns=("uniprot_id",),
        )
        train_rows = [
            {
                "uniprot_id": "P1",
                "protein_sequence": "AAAA",
                "ec_number": "1.1.1.1",
            },
            {
                "uniprot_id": "P2",
                "protein_sequence": "BBBB",
                "ec_number": "2.2.2.2",
            },
        ]
        vocabulary = HierarchicalECLabelBuilder().fit(train_rows).vocabulary_by_level
        rows = _TaskRows(
            train_rows=train_rows,
            val_rows=[],
            test_rows=[train_rows[0]],
            needed_ids={"P1", "P2"},
            needed_sequences={"AAAA", "BBBB"},
            ec_vocabulary=vocabulary,
        )
        store = ResidueEmbeddingStore(by_id={"P1": np.ones((4, 3), dtype=np.float32)})
        store._available_ids = {"P1"}
        store._seq_to_key = {}

        with (
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store",
                return_value=store,
            ),
            mock.patch(
                "enzymado.data_loading.datasets.get_task_config",
                return_value=config,
            ),
            self.assertRaisesRegex(ValueError, "Missing residue embedding"),
        ):
            _build_loader_data(
                config,
                "ec_test",
                "unused.safetensors",
                "random",
                rows,
                residue_store_mode="auto",
                embedding_dim=3,
                sequence_to_id={},
            )

    def test_ec_loader_route_uses_supplied_rows_and_canonical_vocabulary(
        self,
    ) -> None:
        from enzymado.data_loading.task_data import _build_loader_data, _TaskRows

        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="ec_number",
            index_columns=("uniprot_id",),
        )
        kept_row = {
            "sample_id": "P1",
            "uniprot_id": "P1",
            "protein_sequence": "AAAA",
            "ec_number": "1.1.1.1",
        }
        vocabulary = {
            "level_1": ("1", "2"),
            "level_2": ("1.1", "2.2"),
            "level_3": ("1.1.1", "2.2.2"),
            "level_4": ("1.1.1.1", "2.2.2.2"),
        }
        rows = _TaskRows(
            train_rows=[kept_row],
            val_rows=[],
            test_rows=[kept_row],
            needed_ids={"P1"},
            needed_sequences={"AAAA"},
            ec_vocabulary=vocabulary,
        )
        store = ResidueEmbeddingStore(by_id={"P1": np.ones((4, 3), dtype=np.float32)})
        store._available_ids = {"P1"}
        store._seq_to_key = {}

        with (
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store",
                return_value=store,
            ),
            mock.patch(
                "enzymado.data_loading.datasets.get_task_config",
                return_value=config,
            ),
        ):
            task_data = _build_loader_data(
                config,
                "ec_test",
                "unused.safetensors",
                "random",
                rows,
                residue_store_mode="auto",
                embedding_dim=3,
                sequence_to_id={},
            )

        label_builder = task_data.train_bundle.label_builder
        self.assertEqual(len(task_data.train_bundle.dataset), 1)
        self.assertEqual(
            label_builder.vocabulary_by_level["level_4"],
            ("1.1.1.1", "2.2.2.2"),
        )

    def test_build_protein_global_xy_direct(self) -> None:
        from enzymado.data_loading.features import PooledEmbeddingStore
        from enzymado.data_loading.stores import build_xy_pooled

        dataset_path = self.root / "solubility_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "solubility", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "solubility": "1",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "BBBB",
                    "solubility": "0",
                    "split": "train",
                },
            ],
        )
        config = TaskConfig(
            name="solubility_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="solubility",
            index_columns=("uniprot_id",),
        )
        store = PooledEmbeddingStore(
            by_id={
                "P1": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "P2": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
            }
        )
        bundle = build_dataset_from_config(config, store, split_name="train")

        x, y = build_xy_pooled(bundle, config.batch_kind)

        np.testing.assert_array_equal(
            x.numpy(),
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            y.numpy(), np.asarray([1.0, 0.0], dtype=np.float32)
        )

    def test_build_mutation_delta_xy_direct(self) -> None:
        from enzymado.data_loading.features import PooledEmbeddingStore
        from enzymado.data_loading.stores import build_xy_pooled

        dataset_path = self.root / "dtm_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "wildtype_sequence", "dtm", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "wildtype_sequence": "BBBB",
                    "dtm": "1.5",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "CCCC",
                    "wildtype_sequence": "DDDD",
                    "dtm": "-0.5",
                    "split": "train",
                },
            ],
        )
        config = TaskConfig(
            name="dtm_test",
            batch_kind="mutation_delta_global",
            label_kind="scalar_regression",
            dataset_paths={"random": dataset_path},
            target_column="dtm",
            wildtype_sequence_column="wildtype_sequence",
            index_columns=("uniprot_id",),
        )
        # Both ends are keyed by sequence. A mutation row's uniprot_id names the
        # protein, not the variant — in FireProtDB 3774 rows share 69 of them —
        # so resolving the mutant by id would hand every variant of a protein
        # the same embedding.
        store = PooledEmbeddingStore(
            by_id={},
            by_sequence={
                "AAAA": np.asarray([3.0, 5.0, 7.0], dtype=np.float32),
                "CCCC": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                "BBBB": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "DDDD": np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
            },
        )
        bundle = build_dataset_from_config(config, store, split_name="train")

        x, y = build_xy_pooled(bundle, config.batch_kind)

        np.testing.assert_array_equal(
            x.numpy(),
            np.asarray([[2.0, 3.0, 4.0], [-0.5, 0.5, 1.5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            y.numpy(), np.asarray([1.5, -0.5], dtype=np.float32)
        )

    @staticmethod
    def _fake_task_rows(needed_ids: set[str], needed_sequences: set[str]):
        from enzymado.data_loading.task_data import _TaskRows

        return _TaskRows(
            train_rows=[],
            val_rows=[],
            test_rows=[],
            needed_ids=needed_ids,
            needed_sequences=needed_sequences,
        )

    def test_auto_mode_materializes_small_residue_stores(
        self,
    ) -> None:
        from enzymado.data_loading.task_data import _build_loader_data

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        materialized_store = types.SimpleNamespace(by_id={"P1": None}, by_sequence={})
        fake_bundle = types.SimpleNamespace(dataset=[1], label_builder=None)

        with (
            mock.patch(
                "enzymado.data_loading.task_data.load_safetensors_embedding_store_subset",
                return_value=materialized_store,
            ) as load_subset,
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store"
            ) as load_lazy,
            mock.patch(
                "enzymado.data_loading.task_data.build_task_dataset",
                return_value=fake_bundle,
            ),
        ):
            _build_loader_data(
                config,
                "active_sites_test",
                "fake_embeddings.safetensors",
                "random",
                self._fake_task_rows({"P1"}, {"AAAA"}),
                residue_store_mode="auto",
                embedding_dim=128,
                sequence_to_id={},
            )

        load_subset.assert_called_once_with(
            "fake_embeddings.safetensors", {"P1"}, {"AAAA"}
        )
        load_lazy.assert_not_called()

    def test_auto_mode_uses_lazy_for_large_residue_stores(
        self,
    ) -> None:
        from enzymado.data_loading.task_data import _build_loader_data

        config = TaskConfig(
            name="binding_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="binding_positions",
            index_columns=("uniprot_id",),
        )
        lazy_store = types.SimpleNamespace(_available_ids=["P1"], _seq_to_key={})
        fake_bundle = types.SimpleNamespace(dataset=[1], label_builder=None)
        huge_sequence = "A" * 10_000

        with (
            mock.patch(
                "enzymado.data_loading.task_data.load_safetensors_embedding_store_subset"
            ) as load_subset,
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store",
                return_value=lazy_store,
            ) as load_lazy,
            mock.patch(
                "enzymado.data_loading.task_data.build_task_dataset",
                return_value=fake_bundle,
            ),
        ):
            # embedding_dim=2_000_000 over a 10k-residue sequence estimates ~74 GiB,
            # comfortably over any real machine's available RAM -> forces "lazy".
            _build_loader_data(
                config,
                "binding_sites_test",
                "fake_embeddings.safetensors",
                "random",
                self._fake_task_rows({"P1"}, {huge_sequence}),
                residue_store_mode="auto",
                embedding_dim=2_000_000,
                sequence_to_id={},
            )

        load_lazy.assert_called_once_with(
            "fake_embeddings.safetensors", {"P1"}, {huge_sequence}
        )
        load_subset.assert_not_called()

    def test_load_task_embedding_store_keeps_lazy_strategy_for_non_residue_tasks(
        self,
    ) -> None:
        from enzymado.data_loading.task_data import _build_loader_data

        config = TaskConfig(
            name="solubility_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="solubility",
            index_columns=("uniprot_id",),
        )
        lazy_store = types.SimpleNamespace(_available_ids=["P1"], _seq_to_key={})
        fake_bundle = types.SimpleNamespace(dataset=[1], label_builder=None)

        with (
            mock.patch(
                "enzymado.data_loading.task_data.load_safetensors_embedding_store_subset"
            ) as load_subset,
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store",
                return_value=lazy_store,
            ) as load_lazy,
            mock.patch(
                "enzymado.data_loading.task_data.build_task_dataset",
                return_value=fake_bundle,
            ),
        ):
            _build_loader_data(
                config,
                "solubility_test",
                "fake_embeddings.safetensors",
                "random",
                self._fake_task_rows({"P1"}, {"AAAA"}),
                residue_store_mode="auto",
                embedding_dim=None,
                sequence_to_id={},
            )

        load_lazy.assert_called_once_with(
            "fake_embeddings.safetensors", {"P1"}, {"AAAA"}
        )
        load_subset.assert_not_called()

    def test_auto_store_mode_respects_slurm_cap_not_node_ram(self) -> None:
        """The store budget must honor the SLURM `--mem` cgroup cap, not the node.

        Regression for the OOM where a 512 GiB node let a `--mem=250G` job
        materialize a ~361 GiB-estimate store (phantom budget) that then stacked
        under the next task. With the real cap the task must fall back to lazy.
        """
        import os

        from enzymado.data_loading import task_data
        from enzymado.data_loading.task_data import _build_loader_data

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": self.root / "unused.csv"},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        lazy_store = types.SimpleNamespace(_available_ids=["P1"], _seq_to_key={})
        fake_bundle = types.SimpleNamespace(dataset=[1], label_builder=None)

        phantom_vm = types.SimpleNamespace(
            available=480 * 1024**3,  # a 512 GiB node reports ~480 GiB free
            total=512 * 1024**3,
        )
        small_proc = types.SimpleNamespace(
            memory_info=lambda: types.SimpleNamespace(rss=8 * 1024**3)
        )

        with (
            mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": str(250 * 1024)}),
            mock.patch.object(
                task_data.psutil, "virtual_memory", return_value=phantom_vm
            ),
            mock.patch.object(task_data.psutil, "Process", return_value=small_proc),
            mock.patch.object(task_data, "_cgroup_mem_limit_bytes", return_value=None),
            mock.patch.object(
                task_data, "_estimate_residue_store_gib", return_value=361.0
            ),
            mock.patch(
                "enzymado.data_loading.task_data.load_safetensors_embedding_store_subset"
            ) as load_subset,
            mock.patch(
                "enzymado.data_loading.task_data.load_lazy_safetensors_embedding_store",
                return_value=lazy_store,
            ) as load_lazy,
            mock.patch(
                "enzymado.data_loading.task_data.build_task_dataset",
                return_value=fake_bundle,
            ),
        ):
            # Sanity: the budget honors the 250G cap despite the 480 GiB "available".
            self.assertLessEqual(task_data._process_memory_budget_gib(), 250.0)
            _build_loader_data(
                config,
                "active_sites_test",
                "fake_embeddings.safetensors",
                "random",
                self._fake_task_rows({"P1"}, {"AAAA"}),
                residue_store_mode="auto",
                embedding_dim=2560,
                sequence_to_id={},
            )

        load_lazy.assert_called_once()
        load_subset.assert_not_called()

    def test_residue_xgboost_uses_external_memory_for_lazy_store(self) -> None:
        """A lazy (disk-streamed) store routes the residue XGBoost binned matrix
        to an on-disk cache (ExtMemQuantileDMatrix) that is cleaned up after fit."""
        import glob
        import os

        from safetensors.torch import save_file
        import torch
        from torch.utils.data import DataLoader
        import xgboost as xgb

        from enzymado.data_loading.collators import ProteinResidueConcatCollator
        from enzymado.data_loading.stores import load_lazy_safetensors_embedding_store
        from enzymado.training.xgb_wrappers import XGBResidueBinaryClassifier

        ids = [f"P{i}" for i in range(8)]
        rng = np.random.default_rng(0)
        embeddings = {
            pid: torch.tensor(rng.standard_normal((10, 4)), dtype=torch.bfloat16)
            for pid in ids
        }
        save_file(embeddings, str(self.root / "emb.safetensors"))

        dataset_path = self.root / "ext_active.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {
                    "uniprot_id": pid,
                    "protein_sequence": "A" * 10,
                    "active_site": str([1, 3, 5] if i % 2 == 0 else [2, 4]),
                    "split": "train",
                }
                for i, pid in enumerate(ids)
            ],
        )
        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        lazy_store = load_lazy_safetensors_embedding_store(
            self.root / "emb.safetensors", needed_ids=set(ids), needed_sequences=set()
        )
        bundle = build_dataset_from_config(config, lazy_store, split_name="train")

        def make_loader():
            return DataLoader(
                bundle.dataset,
                batch_size=3,
                collate_fn=ProteinResidueConcatCollator(),
                shuffle=False,
                num_workers=0,
            )

        cache_dir = self.root / "xgb_cache"
        cache_dir.mkdir()
        model = XGBResidueBinaryClassifier(
            n_estimators=8,
            max_depth=3,
            max_bin=64,
            n_jobs=2,
            device="cpu",
            external_memory="auto",
            cache_dir=str(cache_dir),
        )
        self.assertTrue(model._should_use_external_memory(make_loader()))

        dmatrix, _ = model._build_quantile_dmatrix(make_loader(), split_name="train")
        self.assertIsInstance(dmatrix, xgb.ExtMemQuantileDMatrix)
        self.assertTrue(glob.glob(os.path.join(str(cache_dir), "xgb_residue_cache_*")))

        model.fit_loader(make_loader(), "protein_residue", val_dataloader=make_loader())
        preds = model.predict_loader(make_loader(), "protein_residue")
        self.assertIsNotNone(model.booster)
        self.assertEqual(preds.shape, (8, 10))
        self.assertFalse(
            glob.glob(os.path.join(str(cache_dir), "xgb_residue_cache_*")),
            "external-memory cache must be removed after fit",
        )

    def test_residue_xgb_checkpoints_and_resumes_mid_boosting(self) -> None:
        """Residue XGBoost must checkpoint during boosting, resume a partial
        fit by continuing, and reject legacy training checkpoints."""
        import joblib
        import torch
        from torch.utils.data import DataLoader

        from enzymado.data_loading.collators import ProteinResidueConcatCollator
        from enzymado.data_loading.features import ResidueEmbeddingStore
        from enzymado.training.xgb_wrappers import XGBResidueBinaryClassifier

        ids = [f"P{i}" for i in range(8)]
        rng = np.random.default_rng(0)
        emb = {
            pid: torch.tensor(rng.standard_normal((10, 4)), dtype=torch.bfloat16)
            for pid in ids
        }
        dataset_path = self.root / "ckpt_active.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {
                    "uniprot_id": pid,
                    "protein_sequence": "A" * 10,
                    "active_site": str([1, 3, 5] if i % 2 == 0 else [2, 4]),
                    "split": "train",
                }
                for i, pid in enumerate(ids)
            ],
        )
        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        bundle = build_dataset_from_config(
            config, ResidueEmbeddingStore(by_id=emb), split_name="train"
        )

        def make_loader():
            return DataLoader(
                bundle.dataset,
                batch_size=3,
                collate_fn=ProteinResidueConcatCollator(),
                num_workers=0,
            )

        def new_model(n_estimators):
            # no val loader + in-RAM matrix -> deterministic full n_estimators rounds
            return XGBResidueBinaryClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                max_bin=64,
                n_jobs=2,
                device="cpu",
                external_memory="off",
                checkpoint_interval=1,
            )

        # Fresh fit writes a checkpoint and marks it complete.
        checkpoint = self.root / "xgb_ckpt.bin"
        fresh = new_model(10)
        fresh.fit_loader(make_loader(), "protein_residue", checkpoint_path=checkpoint)
        self.assertTrue(checkpoint.exists())
        self.assertTrue(fresh.training_complete_)
        self.assertEqual(fresh.booster.num_boosted_rounds(), 10)
        fresh_predictions = fresh.predict_loader(make_loader(), "protein_residue")

        # A completed checkpoint short-circuits instead of boosting again.
        done = new_model(10)
        done.fit_loader(make_loader(), "protein_residue", checkpoint_path=checkpoint)
        self.assertEqual(done.booster.num_boosted_rounds(), 10)

        # A partial checkpoint (killed at round 4 of 10) continues to 10.
        partial = new_model(4)
        partial.fit_loader(
            make_loader(), "protein_residue", checkpoint_path=self.root / "p.bin"
        )
        partial.n_estimators = (
            10  # resume reads hyperparameters back from the checkpoint
        )
        partial.training_contract_["params"]["n_estimators"] = 10
        partial.training_complete_ = False
        partial.rounds_completed_ = 4
        joblib.dump(partial, self.root / "partial.bin")

        resumed = new_model(10)
        resumed.fit_loader(
            make_loader(), "protein_residue", checkpoint_path=self.root / "partial.bin"
        )
        self.assertEqual(resumed.booster.num_boosted_rounds(), 10)
        self.assertTrue(resumed.training_complete_)
        resumed_predictions = resumed.predict_loader(make_loader(), "protein_residue")
        np.testing.assert_array_equal(resumed_predictions, fresh_predictions)

        # Legacy checkpoints predate the marker and must NOT be boosted a second time.
        legacy = joblib.load(checkpoint)
        del legacy.__dict__["training_complete_"]
        legacy.__dict__.pop("rounds_completed_", None)
        legacy.__dict__.pop("training_contract_", None)
        joblib.dump(legacy, self.root / "legacy.bin")
        legacy_model = new_model(10)
        with self.assertRaisesRegex(ValueError, "Incompatible XGBoost checkpoint"):
            legacy_model.fit_loader(
                make_loader(),
                "protein_residue",
                checkpoint_path=self.root / "legacy.bin",
            )

    def test_residue_xgb_cache_prefers_largest_disk(self) -> None:
        """The external-memory cache must avoid small node-local /tmp and pick the
        filesystem with the most free space (or an explicit override)."""
        import os
        import types
        from unittest import mock

        from enzymado.training import xgb_residue, xgb_support, xgb_wrappers

        model = xgb_wrappers.XGBResidueBinaryClassifier(external_memory="auto")
        self.assertIs(
            xgb_wrappers.XGBResidueBinaryClassifier,
            xgb_residue.XGBResidueBinaryClassifier,
        )
        free = {
            "/local_tmp": 219 * 1024**3,
            "/big_tmp": 1600 * 1024**3,
            "/scratch": 2400 * 1024**3,
        }

        def fake_usage(path):
            return types.SimpleNamespace(total=0, used=0, free=free.get(path, 0))

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SCRATCH": "/scratch",
                    "SLURM_TMPDIR": "/local_tmp",
                    "TMPDIR": "/big_tmp",
                },
                clear=False,
            ),
            mock.patch.object(xgb_residue.os, "makedirs"),
            mock.patch.object(
                xgb_residue.shutil,
                "disk_usage",
                side_effect=fake_usage,
            ),
        ):
            os.environ.pop("XGB_RESIDUE_CACHE_DIR", None)
            self.assertEqual(model._resolve_cache_base(), "/scratch")
            with mock.patch.dict(os.environ, {"XGB_RESIDUE_CACHE_DIR": "/forced"}):
                self.assertEqual(model._resolve_cache_base(), "/forced")

        self.assertTrue(
            xgb_support._is_disk_write_error(Exception("FileStream.Write incomplete"))
        )
        self.assertFalse(xgb_support._is_disk_write_error(Exception("out of memory")))

    def test_residue_binary_training_and_prediction_serialization(self) -> None:
        from safetensors.torch import load as load_tensors
        from torch.utils.data import DataLoader

        from enzymado.artifacts.serialization import make_safetensors_predictions_file
        from enzymado.data_loading.stores import get_target_payload
        from enzymado.training.torch_structured_wrappers import (
            ResidueBinaryClassifierTorch,
        )

        dataset_path = self.root / "active_sites_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "active_site": "[1, 3]",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AA",
                    "active_site": "[2]",
                    "split": "train",
                },
            ],
        )

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        import torch

        # Real embedding stores always hold native bfloat16 tensors (safetensors
        # decodes bf16 directly); no int16 bitcast reinterpretation ever happens.
        store = ResidueEmbeddingStore(
            by_id={
                "P1": torch.tensor(
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    dtype=torch.bfloat16,
                ),
                "P2": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.bfloat16),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        loader = DataLoader(
            bundle.dataset,
            batch_size=2,
            collate_fn=bundle.collator,
            shuffle=False,
            num_workers=0,
        )
        model = ResidueBinaryClassifierTorch(
            hidden_dims=(8,),
            lr=1e-2,
            epochs=3,
            early_stopping_patience=1,
        )
        model.fit_loader(loader, batch_kind="protein_residue", val_dataloader=loader)
        preds = model.predict_loader(loader, batch_kind="protein_residue")
        y_true, target_mask = get_target_payload(bundle)

        self.assertEqual(preds.shape, y_true.shape)
        self.assertEqual(preds.shape, target_mask.shape)
        self.assertEqual(preds.shape, (2, 4))
        self.assertIsNotNone(model.best_epoch_)

        run_dir = self.root / "residue_outputs"
        run_dir.mkdir()
        pred_path = make_safetensors_predictions_file(
            y_true, preds, run_dir, "train", target_mask
        )
        payload = load_tensors(lzma.open(pred_path, "rb").read())
        self.assertIn("y_true", payload)
        self.assertIn("predictions", payload)
        self.assertIn("target_mask", payload)

    def test_attention_ligand_extra_features_are_train_scaled(self) -> None:
        from torch.utils.data import DataLoader

        from enzymado.models.attention_blocks import batch_to_attention_tensors
        from enzymado.training.attention_wrappers import AttentionBinaryClassifierTorch

        dataset_path = self.root / "binding_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "smiles", "binding", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "smiles": "L1",
                    "binding": "1",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AAAA",
                    "smiles": "L2",
                    "binding": "0",
                    "split": "train",
                },
                {
                    "uniprot_id": "P3",
                    "protein_sequence": "AAAA",
                    "smiles": "L3",
                    "binding": "1",
                    "split": "train",
                },
                {
                    "uniprot_id": "P4",
                    "protein_sequence": "AAAA",
                    "smiles": "L4",
                    "binding": "0",
                    "split": "train",
                },
            ],
        )
        config = TaskConfig(
            name="binding_test",
            batch_kind="protein_ligand_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="binding",
            ligand_column="smiles",
            index_columns=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((4, 2), dtype=np.float32),
                "P3": np.ones((4, 2), dtype=np.float32),
                "P4": np.ones((4, 2), dtype=np.float32),
            }
        )
        ligand_values = {
            "L1": np.asarray([100.0, 10.0], dtype=np.float32),
            "L2": np.asarray([200.0, 20.0], dtype=np.float32),
            "L3": np.asarray([300.0, 30.0], dtype=np.float32),
            "L4": np.asarray([400.0, 40.0], dtype=np.float32),
        }
        bundle = build_dataset_from_config(
            config,
            store,
            split_name="train",
            ligand_featurizer=lambda smiles: ligand_values[smiles],
        )
        loader = DataLoader(
            bundle.dataset,
            batch_size=2,
            collate_fn=bundle.collator,
            shuffle=False,
            num_workers=0,
        )
        model = AttentionBinaryClassifierTorch(
            hidden_dims=(), lr=1e-3, epochs=1, optimizer_name="adamw"
        )
        model._fit_extra_scaler(loader, "protein_ligand_global")

        normalized_chunks = []
        for batch in loader:
            _, _, extra_t = batch_to_attention_tensors(
                batch, "protein_ligand_global", model.device
            )
            normalized_chunks.append(model._normalize_extra(extra_t).cpu().numpy())
        normalized = np.concatenate(normalized_chunks, axis=0)

        np.testing.assert_allclose(
            normalized.mean(axis=0), np.zeros(2, dtype=np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            normalized.std(axis=0), np.ones(2, dtype=np.float32), atol=1e-6
        )

    def test_attention_sequence_scaler_is_stable_and_neutralizes_constants(
        self,
    ) -> None:
        import torch

        from enzymado.training.attention_wrappers import AttentionRegressorTorch

        model = AttentionRegressorTorch(
            hidden_dims=(),
            lr=1e-3,
            epochs=1,
            optimizer_name="adamw",
            device="cpu",
        )
        batches = [
            (
                torch.tensor([[[3.0, 10_000.0], [3.0, 10_000.0]]]),
                torch.tensor([[True, True]]),
            ),
            (
                torch.tensor([[[3.0, 10_001.0], [3.0, 10_001.0]]]),
                torch.tensor([[True, True]]),
            ),
        ]

        model._fit_sequence_scaler_from_batches(
            batches,
            lambda batch, device: (batch[0].to(device), batch[1].to(device)),
        )

        torch.testing.assert_close(
            model.seq_mean_.flatten(),
            torch.tensor([3.0, 10_000.5]),
        )
        torch.testing.assert_close(
            model.seq_scale_.flatten(),
            torch.tensor([1.0, 0.5]),
        )
        normalized = model._normalize_sequence(torch.tensor([[[3.001, 10_000.5]]]))
        self.assertAlmostEqual(float(normalized[0, 0, 0]), 0.001, places=6)
        self.assertEqual(float(normalized[0, 0, 1]), 0.0)

    def test_attention_resume_skips_scaler_refit(self) -> None:
        """A resumed attention fit must not re-run the full-dataset scaler passes.

        The scaler is fit once on a fresh run and restored from the checkpoint on
        resume; re-fitting it on every TIMEOUT restart was pure wasted disk I/O.
        """
        from torch.utils.data import DataLoader

        from enzymado.training.attention_wrappers import (
            AttentionBinaryClassifierTorch,
            _PooledAttentionWrapper,
        )

        dataset_path = self.root / "resume_binary.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "binding", "split"],
            [
                {
                    "uniprot_id": f"P{i}",
                    "protein_sequence": "AAAA",
                    "binding": str(i % 2),
                    "split": "train",
                }
                for i in range(6)
            ],
        )
        config = TaskConfig(
            name="resume_binary_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_paths={"random": dataset_path},
            target_column="binding",
            index_columns=("uniprot_id",),
        )
        rng = np.random.default_rng(0)
        store = ResidueEmbeddingStore(
            by_id={
                f"P{i}": rng.standard_normal((4, 3)).astype(np.float32)
                for i in range(6)
            }
        )
        bundle = build_dataset_from_config(config, store, split_name="train")

        def make_loader():
            return DataLoader(
                bundle.dataset, batch_size=2, collate_fn=bundle.collator, num_workers=0
            )

        checkpoint = self.root / "training_checkpoint.bin"

        fresh = AttentionBinaryClassifierTorch(
            hidden_dims=(), lr=1e-3, epochs=2, optimizer_name="adamw"
        )
        with mock.patch.object(
            AttentionBinaryClassifierTorch,
            "_fit_sequence_scaler",
            autospec=True,
            side_effect=_PooledAttentionWrapper._fit_sequence_scaler,
        ) as fresh_spy:
            fresh.fit_loader(
                make_loader(), "protein_global", checkpoint_path=checkpoint
            )
        self.assertEqual(fresh_spy.call_count, 1)
        self.assertIsNotNone(fresh.seq_mean_)
        self.assertTrue(checkpoint.exists())

        resumed = AttentionBinaryClassifierTorch(
            hidden_dims=(), lr=1e-3, epochs=2, optimizer_name="adamw"
        )
        with mock.patch.object(
            AttentionBinaryClassifierTorch,
            "_fit_sequence_scaler",
            autospec=True,
            side_effect=_PooledAttentionWrapper._fit_sequence_scaler,
        ) as resume_spy:
            resumed.fit_loader(
                make_loader(), "protein_global", checkpoint_path=checkpoint
            )
        self.assertEqual(resume_spy.call_count, 0)
        self.assertIsNotNone(resumed.seq_mean_)

    def test_load_loader_task_data_honors_auto_store_mode(self) -> None:
        """The attention route forwards residue_store_mode (default 'auto') so
        per-residue tasks materialize into RAM instead of forcing a lazy store."""
        from enzymado.data_loading import task_data

        with (
            mock.patch.object(task_data, "get_task_config", return_value=object()),
            mock.patch.object(
                task_data, "_load_and_filter_rows", return_value=object()
            ),
            mock.patch.object(task_data, "_build_loader_data") as build,
        ):
            task_data.load_loader_task_data(
                "any_task",
                "fake.safetensors",
                "random",
                allowed_ids={"P1"},
                sequence_to_id={},
            )
        # build is called positionally; argument 5 is the embedding mode.
        self.assertEqual(build.call_args.args[5], "auto")

    def test_get_target_payload_for_residue_tasks_does_not_reload_embeddings(
        self,
    ) -> None:
        from enzymado.data_loading.stores import get_target_payload

        dataset_path = self.root / "active_sites_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "active_site": "[1, 3]",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AA",
                    "active_site": "[2]",
                    "split": "train",
                },
            ],
        )

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column="active_site",
            index_columns=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((2, 2), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        bundle.dataset.protein_builder.build = mock.Mock(
            side_effect=AssertionError("embedding reload")
        )

        y_true, target_mask = get_target_payload(bundle)

        self.assertEqual(y_true.shape, (2, 4))
        self.assertEqual(target_mask.shape, (2, 4))
        np.testing.assert_array_equal(
            y_true[0], np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            target_mask[1], np.asarray([1, 1, 0, 0], dtype=bool)
        )

    def test_xgb_residue_binary_training_and_prediction(self) -> None:
        from torch.utils.data import DataLoader

        from enzymado.data_loading.stores import get_target_payload
        from enzymado.training.xgb_wrappers import XGBResidueBinaryClassifier

        dataset_path = self.root / "binding_sites_random_standardized.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "binding_positions", "split"],
            [
                {
                    "uniprot_id": "P1",
                    "protein_sequence": "AAAA",
                    "binding_positions": "[1, 3]",
                    "split": "train",
                },
                {
                    "uniprot_id": "P2",
                    "protein_sequence": "AAA",
                    "binding_positions": "[2]",
                    "split": "train",
                },
            ],
        )

        config = TaskConfig(
            name="binding_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_paths={"random": dataset_path},
            target_column="binding_positions",
            index_columns=("uniprot_id",),
        )
        import torch

        # Real embedding stores always hold native bfloat16 tensors (safetensors
        # decodes bf16 directly); no int16 bitcast reinterpretation ever happens.
        store = ResidueEmbeddingStore(
            by_id={
                "P1": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.bfloat16,
                ),
                "P2": torch.tensor(
                    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16
                ),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        concat_collator = ProteinResidueConcatCollator()
        loader = DataLoader(
            bundle.dataset,
            batch_size=2,
            collate_fn=concat_collator,
            shuffle=False,
            num_workers=0,
        )
        model = XGBResidueBinaryClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            max_bin=64,
            n_jobs=1,
            device="cpu",
        )
        x_flat, y_flat, _ = model._flatten_batch(next(iter(loader)))
        self.assertEqual(x_flat.dtype, np.float32)
        self.assertEqual(y_flat.dtype, np.int8)
        model.fit_loader(loader, batch_kind="protein_residue", val_dataloader=loader)
        preds = model.predict_loader(loader, batch_kind="protein_residue")
        y_true, target_mask = get_target_payload(bundle)

        self.assertEqual(preds.shape, y_true.shape)
        self.assertEqual(preds.shape, target_mask.shape)
        self.assertEqual(preds.shape, (2, 4))

    def test_xgb_flat_ec_block_classifier_trains_blocks(self) -> None:
        import enzymado.training.xgb_support as xgb_support
        import enzymado.training.xgb_wrappers as wrappers

        fit_label_shapes: list[tuple[int, ...]] = []

        class FakeQuantileDMatrix:
            def __init__(self, x, label=None, ref=None, max_bin=None):
                self.x = np.asarray(x, dtype=np.float32)
                self.label = (
                    None if label is None else np.asarray(label, dtype=np.float32)
                )
                self.ref = ref
                self.max_bin = max_bin

        class FakeBooster:
            def __init__(self, out_dim: int):
                self.out_dim = out_dim

            def predict(self, dmatrix):
                return np.full(
                    (dmatrix.x.shape[0], self.out_dim), 0.75, dtype=np.float32
                )

        def fake_train(
            params,
            dtrain,
            num_boost_round,
            evals,
            verbose_eval,
            early_stopping_rounds=None,
        ):
            del params, num_boost_round, evals, verbose_eval, early_stopping_rounds
            fit_label_shapes.append(tuple(dtrain.label.shape))
            return FakeBooster(dtrain.label.shape[1])

        fake_xgb = types.SimpleNamespace(
            QuantileDMatrix=FakeQuantileDMatrix,
            train=fake_train,
        )

        x = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        y = np.asarray(
            [
                [1, 0, 1, 0, 1],
                [1, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        target_mask = np.asarray(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, False, True, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        )

        with mock.patch.object(xgb_support, "xgb", fake_xgb):
            model = wrappers.XGBFlatECBlockClassifier(
                n_estimators=5,
                max_depth=2,
                learning_rate=0.3,
                block_size=2,
                n_jobs=1,
                device="cpu",
            )
            model.fit(x, y, target_mask=target_mask)
            preds = model.predict(x)

        self.assertEqual(model.n_outputs_, 5)
        self.assertEqual(len(model.block_slices_), 3)
        self.assertEqual(fit_label_shapes, [(3, 2), (4, 2)])
        self.assertEqual(preds.shape, y.shape)
        np.testing.assert_array_equal(preds[:, :4], np.ones((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(preds[:, 4], np.ones(4, dtype=np.float32))

    def test_train_model_loader_passes_supported_extra_kwargs(self) -> None:
        from enzymado.training.model_registry import train_model_loader

        calls: dict[str, object] = {}

        class DummyLoaderModel:
            def __init__(self, alpha: int) -> None:
                self.alpha = alpha

            def fit_loader(
                self,
                train_loader,
                batch_kind: str,
                val_dataloader=None,
                progress_label: str | None = None,
            ) -> None:
                calls["alpha"] = self.alpha
                calls["train_loader"] = train_loader
                calls["batch_kind"] = batch_kind
                calls["val_dataloader"] = val_dataloader
                calls["progress_label"] = progress_label

        model = train_model_loader(
            {"wrap": DummyLoaderModel, "params": {"alpha": 7}},
            train_loader="train-loader",
            batch_kind="protein_residue",
            val_loader="val-loader",
            progress_label="demo/xgb",
            ignored_kw="ignored",
        )

        self.assertIsInstance(model, DummyLoaderModel)
        self.assertEqual(calls["alpha"], 7)
        self.assertEqual(calls["train_loader"], "train-loader")
        self.assertEqual(calls["batch_kind"], "protein_residue")
        self.assertEqual(calls["val_dataloader"], "val-loader")
        self.assertEqual(calls["progress_label"], "demo/xgb")


if __name__ == "__main__":
    unittest.main()
