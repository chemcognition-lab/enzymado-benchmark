from __future__ import annotations

import csv
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from src.data_loading.collators import ProteinResidueConcatCollator
from src.data_loading.datasets import build_dataset_from_config
from src.data_loading.features import ResidueEmbeddingStore
from src.data_loading.registry import TaskConfig


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

    def test_ec_multihead_training_and_prediction_serialization(self) -> None:
        from models.models import ECMultiHeadClassifierTorch
        from models.train import get_target_payload, make_predictions_file

        dataset_path = self.root / "ec.csv"
        split_path = self.root / "ec_split.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "ec_number"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "ec_number": "1.1.1.1"},
                {"uniprot_id": "P2", "protein_sequence": "AAA", "ec_number": "2.2.-.-"},
            ],
        )
        write_csv(
            split_path,
            ["uniprot_id", "split"],
            [
                {"uniprot_id": "P1", "split": "train"},
                {"uniprot_id": "P2", "split": "train"},
            ],
        )

        config = TaskConfig(
            name="ec_test",
            batch_kind="protein_global",
            label_kind="multilabel_ec",
            dataset_path=dataset_path,
            target_column="ec_number",
            sample_id_candidates=("uniprot_id",),
            split_files={"random": split_path},
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 3), dtype=np.float32),
                "P2": np.full((3, 3), 2.0, dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        X = np.stack(
            [
                np.asarray(bundle.dataset[i]["protein_emb"], dtype=np.float32).mean(axis=0)
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
        model.fit(X, y_true, target_mask, X_val=X, y_val=y_true, target_mask_val=target_mask)
        preds = model.predict(X)

        self.assertEqual(set(preds), {"level_1", "level_2", "level_3", "level_4"})
        self.assertEqual(preds["level_1"].shape, y_true["level_1"].shape)
        self.assertEqual(preds["level_4"].shape, y_true["level_4"].shape)
        self.assertEqual(target_mask["level_2"].shape, (2,))
        self.assertIsNotNone(model.best_epoch_)

        run_dir = self.root / "ec_outputs"
        run_dir.mkdir()
        make_predictions_file(y_true, preds, run_dir, "train", target_mask)
        payload = np.load(run_dir / "train_predictions.npz")
        self.assertIn("y_true_level_1", payload.files)
        self.assertIn("predictions_level_4", payload.files)
        self.assertIn("target_mask_level_3", payload.files)

    def test_ec_multihead_pooled_training_and_prediction(self) -> None:
        from models.models import ECMultiHeadClassifierTorch

        X = np.asarray(
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
            "level_1": np.asarray([True, True, True, True], dtype=bool),
            "level_2": np.asarray([True, True, True, True], dtype=bool),
            "level_3": np.asarray([True, True, True, True], dtype=bool),
            "level_4": np.asarray([True, True, True, True], dtype=bool),
        }

        model = ECMultiHeadClassifierTorch(
            hidden_dims=(8,),
            lr=1e-2,
            epochs=3,
            batch_size=2,
            early_stopping_patience=1,
        )
        model.fit(
            X,
            y,
            target_mask,
            X_val=X,
            y_val=y,
            target_mask_val=target_mask,
        )
        preds = model.predict(X)

        self.assertEqual(set(preds), {"level_1", "level_2", "level_3", "level_4"})
        self.assertEqual(preds["level_1"].shape, y["level_1"].shape)
        self.assertEqual(preds["level_4"].shape, y["level_4"].shape)
        self.assertIsNotNone(model.best_epoch_)

    def test_build_protein_global_Xy_direct(self) -> None:
        from models.train import build_protein_global_Xy_direct

        config = TaskConfig(
            name="solubility_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_path=self.root / "unused.csv",
            target_column="solubility",
            sample_id_candidates=("uniprot_id",),
        )
        rows = [
            {"uniprot_id": "P1", "protein_sequence": "AAAA", "solubility": "1"},
            {"uniprot_id": "P2", "protein_sequence": "BBBB", "solubility": "0"},
        ]
        by_id = {
            "P1": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "P2": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        }
        by_seq: dict[str, np.ndarray] = {}

        X, y = build_protein_global_Xy_direct(rows, by_id, by_seq, config)

        np.testing.assert_array_equal(
            X,
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(y, np.asarray([1.0, 0.0], dtype=np.float32))

    def test_build_mutation_delta_Xy_direct(self) -> None:
        from models.train import build_mutation_delta_Xy_direct

        config = TaskConfig(
            name="dtm_test",
            batch_kind="mutation_delta_global",
            label_kind="scalar_regression",
            dataset_path=self.root / "unused.csv",
            target_column="dtm",
            wildtype_sequence_column="wildtype_sequence",
            sample_id_candidates=("uniprot_id",),
        )
        rows = [
            {
                "uniprot_id": "P1",
                "protein_sequence": "AAAA",
                "wildtype_sequence": "BBBB",
                "dtm": "1.5",
            },
            {
                "uniprot_id": "P2",
                "protein_sequence": "CCCC",
                "wildtype_sequence": "DDDD",
                "dtm": "-0.5",
            },
        ]
        by_id = {
            "P1": np.asarray([3.0, 5.0, 7.0], dtype=np.float32),
            "P2": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        }
        by_seq = {
            "BBBB": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "DDDD": np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        }

        X, y = build_mutation_delta_Xy_direct(rows, by_id, by_seq, config)

        np.testing.assert_array_equal(
            X,
            np.asarray([[2.0, 3.0, 4.0], [-0.5, 0.5, 1.5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(y, np.asarray([1.5, -0.5], dtype=np.float32))

    def test_load_task_embedding_store_materializes_small_protein_residue_tasks_in_auto_mode(self) -> None:
        from models.train import load_task_embedding_store

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_path=self.root / "unused.csv",
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
        )
        materialized_store = object()

        with (
            mock.patch("models.train.load_embedding_store_subset", return_value=materialized_store) as load_subset,
            mock.patch("models.train.load_lazy_embedding_store") as load_lazy,
        ):
            store, mode = load_task_embedding_store(
                config,
                "fake_embeddings.h5",
                {"P1"},
                {"AAAA"},
                residue_store_mode="auto",
                embedding_dim=128,
            )

        self.assertIs(store, materialized_store)
        self.assertEqual(mode, "materialized")
        load_subset.assert_called_once_with("fake_embeddings.h5", {"P1"}, {"AAAA"})
        load_lazy.assert_not_called()

    def test_load_task_embedding_store_uses_lazy_for_large_protein_residue_tasks_in_auto_mode(self) -> None:
        from models.train import load_task_embedding_store

        config = TaskConfig(
            name="binding_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_path=self.root / "unused.csv",
            target_column=None,
            residue_positions_column="binding_positions",
            sample_id_candidates=("uniprot_id",),
        )
        lazy_store = object()

        with (
            mock.patch("models.train.load_embedding_store_subset") as load_subset,
            mock.patch("models.train.load_lazy_embedding_store", return_value=lazy_store) as load_lazy,
        ):
            store, mode = load_task_embedding_store(
                config,
                "fake_embeddings.h5",
                {"P1"},
                {"A" * 10_000},
                residue_store_mode="auto",
                embedding_dim=2_000_000,
            )

        self.assertIs(store, lazy_store)
        self.assertEqual(mode, "lazy")
        load_lazy.assert_called_once_with("fake_embeddings.h5", {"P1"}, {"A" * 10_000})
        load_subset.assert_not_called()

    def test_load_task_embedding_store_keeps_lazy_strategy_for_non_residue_tasks(self) -> None:
        from models.train import load_task_embedding_store

        config = TaskConfig(
            name="solubility_test",
            batch_kind="protein_global",
            label_kind="binary",
            dataset_path=self.root / "unused.csv",
            target_column="solubility",
            sample_id_candidates=("uniprot_id",),
        )
        lazy_store = object()

        with (
            mock.patch("models.train.load_embedding_store_subset") as load_subset,
            mock.patch("models.train.load_lazy_embedding_store", return_value=lazy_store) as load_lazy,
        ):
            store, mode = load_task_embedding_store(
                config,
                "fake_embeddings.h5",
                {"P1"},
                {"AAAA"},
            )

        self.assertIs(store, lazy_store)
        self.assertEqual(mode, "lazy")
        load_lazy.assert_called_once_with("fake_embeddings.h5", {"P1"}, {"AAAA"})
        load_subset.assert_not_called()

    def test_residue_binary_training_and_prediction_serialization(self) -> None:
        from torch.utils.data import DataLoader

        from models.models import ResidueBinaryClassifierTorch
        from models.train import get_target_payload, make_predictions_file

        dataset_path = self.root / "active_sites.csv"
        split_path = self.root / "active_sites_split.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "active_site": "[1, 3]"},
                {"uniprot_id": "P2", "protein_sequence": "AA", "active_site": "[2]"},
            ],
        )
        write_csv(
            split_path,
            ["uniprot_id", "split"],
            [
                {"uniprot_id": "P1", "split": "train"},
                {"uniprot_id": "P2", "split": "train"},
            ],
        )

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_path=dataset_path,
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
            split_files={"random": split_path},
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((2, 2), dtype=np.float32),
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
        make_predictions_file(y_true, preds, run_dir, "train", target_mask)
        payload = np.load(run_dir / "train_predictions.npz")
        self.assertIn("y_true", payload.files)
        self.assertIn("predictions", payload.files)
        self.assertIn("target_mask", payload.files)

    def test_get_target_payload_for_residue_tasks_does_not_reload_embeddings(self) -> None:
        from models.train import get_target_payload

        dataset_path = self.root / "active_sites.csv"
        split_path = self.root / "active_sites_split.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "active_site"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "active_site": "[1, 3]"},
                {"uniprot_id": "P2", "protein_sequence": "AA", "active_site": "[2]"},
            ],
        )
        write_csv(
            split_path,
            ["uniprot_id", "split"],
            [
                {"uniprot_id": "P1", "split": "train"},
                {"uniprot_id": "P2", "split": "train"},
            ],
        )

        config = TaskConfig(
            name="active_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_path=dataset_path,
            target_column=None,
            residue_positions_column="active_site",
            sample_id_candidates=("uniprot_id",),
            split_files={"random": split_path},
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.ones((4, 2), dtype=np.float32),
                "P2": np.ones((2, 2), dtype=np.float32),
            }
        )

        bundle = build_dataset_from_config(config, store, split_name="train")
        bundle.dataset.protein_builder.build = mock.Mock(side_effect=AssertionError("embedding reload"))

        y_true, target_mask = get_target_payload(bundle)

        self.assertEqual(y_true.shape, (2, 4))
        self.assertEqual(target_mask.shape, (2, 4))
        np.testing.assert_array_equal(y_true[0], np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(target_mask[1], np.asarray([1, 1, 0, 0], dtype=bool))

    def test_xgb_residue_binary_training_and_prediction(self) -> None:
        from torch.utils.data import DataLoader

        from models.models import XGBResidueBinaryClassifier
        from models.train import get_target_payload

        dataset_path = self.root / "binding_sites.csv"
        split_path = self.root / "binding_sites_split.csv"
        write_csv(
            dataset_path,
            ["uniprot_id", "protein_sequence", "binding_positions"],
            [
                {"uniprot_id": "P1", "protein_sequence": "AAAA", "binding_positions": "[1, 3]"},
                {"uniprot_id": "P2", "protein_sequence": "AAA", "binding_positions": "[2]"},
            ],
        )
        write_csv(
            split_path,
            ["uniprot_id", "split"],
            [
                {"uniprot_id": "P1", "split": "train"},
                {"uniprot_id": "P2", "split": "train"},
            ],
        )

        config = TaskConfig(
            name="binding_sites_test",
            batch_kind="protein_residue",
            label_kind="residue_binary_vector",
            dataset_path=dataset_path,
            target_column=None,
            residue_positions_column="binding_positions",
            sample_id_candidates=("uniprot_id",),
            split_files={"random": split_path},
            split_id_candidates=("uniprot_id",),
        )
        store = ResidueEmbeddingStore(
            by_id={
                "P1": np.asarray(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    dtype=np.float32,
                ),
                "P2": np.asarray(
                    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    dtype=np.float32,
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
        model.fit_loader(loader, batch_kind="protein_residue", val_dataloader=loader)
        preds = model.predict_loader(loader, batch_kind="protein_residue")
        y_true, target_mask = get_target_payload(bundle)

        self.assertEqual(preds.shape, y_true.shape)
        self.assertEqual(preds.shape, target_mask.shape)
        self.assertEqual(preds.shape, (2, 4))

    def test_xgb_flat_ec_block_classifier_trains_blocks(self) -> None:
        import models.xgb_wrappers as wrappers

        fit_label_shapes: list[tuple[int, ...]] = []

        class FakeQuantileDMatrix:
            def __init__(self, X, label=None, ref=None, max_bin=None):
                self.X = np.asarray(X, dtype=np.float32)
                self.label = None if label is None else np.asarray(label, dtype=np.float32)
                self.ref = ref
                self.max_bin = max_bin

        class FakeBooster:
            def __init__(self, out_dim: int):
                self.out_dim = out_dim

            def predict(self, dmatrix):
                return np.full((dmatrix.X.shape[0], self.out_dim), 0.75, dtype=np.float32)

        def fake_train(params, dtrain, num_boost_round, evals, verbose_eval, early_stopping_rounds=None):
            del params, num_boost_round, evals, verbose_eval, early_stopping_rounds
            fit_label_shapes.append(tuple(dtrain.label.shape))
            return FakeBooster(dtrain.label.shape[1])

        fake_xgb = types.SimpleNamespace(
            QuantileDMatrix=FakeQuantileDMatrix,
            train=fake_train,
        )

        X = np.asarray(
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

        with mock.patch.object(wrappers, "xgb", fake_xgb):
            model = wrappers.XGBFlatECBlockClassifier(
                n_estimators=5,
                max_depth=2,
                learning_rate=0.3,
                block_size=2,
                n_jobs=1,
                device="cpu",
            )
            model.fit(X, y, target_mask=target_mask)
            preds = model.predict(X)

        self.assertEqual(model.n_outputs_, 5)
        self.assertEqual(len(model.block_slices_), 3)
        self.assertEqual(fit_label_shapes, [(3, 2), (4, 2)])
        self.assertEqual(preds.shape, y.shape)
        np.testing.assert_array_equal(preds[:, :4], np.ones((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(preds[:, 4], np.ones(4, dtype=np.float32))

    def test_train_model_loader_passes_supported_extra_kwargs(self) -> None:
        from models.train_runtime import train_model_loader

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
