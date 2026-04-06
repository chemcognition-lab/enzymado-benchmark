from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
        from torch.utils.data import DataLoader

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
        loader = DataLoader(
            bundle.dataset,
            batch_size=2,
            collate_fn=bundle.collator,
            shuffle=False,
            num_workers=0,
        )
        model = ECMultiHeadClassifierTorch(
            hidden_dims=(8,),
            lr=1e-2,
            epochs=3,
            early_stopping_patience=1,
        )
        model.fit_loader(loader, batch_kind="protein_global", val_dataloader=loader)
        preds = model.predict_loader(loader, batch_kind="protein_global")
        y_true, target_mask = get_target_payload(bundle)

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


if __name__ == "__main__":
    unittest.main()
