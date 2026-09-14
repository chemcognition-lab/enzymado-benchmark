"""Exact reproducibility contracts for interrupted PyTorch training."""

from pathlib import Path
import random
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import enzymado.data_loading.collators
import enzymado.models.common
import enzymado.training.torch_standard_wrappers


class _InterruptedError(RuntimeError):
    """Simulate a scheduler stopping a job after a completed epoch."""


def _loader(seed: int) -> DataLoader:
    """Build a shuffled loader whose generator state matters across epochs."""
    samples = [
        {
            "sample_id": f"P{index}",
            "protein_emb": torch.tensor(
                [
                    [float(index), float(index % 3), -1.0],
                    [float(index + 1), float((index + 1) % 3), 1.0],
                ],
                dtype=torch.float32,
            ),
            "target": float(0.5 * index - 1.0),
        }
        for index in range(8)
    ]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        samples,
        batch_size=2,
        shuffle=True,
        generator=generator,
        collate_fn=enzymado.data_loading.collators.ProteinGlobalCollator(),
        num_workers=0,
    )


def _wrapper(
    device: str,
) -> enzymado.training.torch_standard_wrappers.MLPRegressorTorch:
    """Return the fixed stochastic model used by both training paths."""
    return enzymado.training.torch_standard_wrappers.MLPRegressorTorch(
        hidden_dims=(6,),
        lr=0.01,
        epochs=4,
        optimizer_name="adam",
        weight_decay=0.01,
        dropout=0.25,
        batch_size=2,
        device=device,
    )


def _rng_sample() -> tuple[
    float,
    np.ndarray,
    torch.Tensor,
    list[torch.Tensor],
]:
    """Read every global RNG after training without resetting it."""
    return (
        random.random(),
        np.random.random(3),
        torch.rand(3),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def test_early_stopping_resume_preserves_patience_progress() -> None:
    """Resume must count non-improving epochs completed before interruption."""
    continuous_model = torch.nn.Linear(1, 1)
    continuous_steps: list[int] = []
    continuous_losses = iter([1.0, 0.9, 0.91, 0.92, 0.93])
    continuous_best_epoch, continuous_best_loss = (
        enzymado.models.common.run_training_with_early_stopping(
            continuous_model,
            epochs=6,
            train_epoch_fn=lambda: continuous_steps.append(1),
            val_loss_fn=lambda: next(continuous_losses),
            patience=2,
        )
    )

    resumed_model = torch.nn.Linear(1, 1)
    resumed_steps: list[int] = []
    resumed_losses = iter([0.92, 0.93])
    resumed_best_epoch, resumed_best_loss = (
        enzymado.models.common.run_training_with_early_stopping(
            resumed_model,
            epochs=6,
            train_epoch_fn=lambda: resumed_steps.append(1),
            val_loss_fn=lambda: next(resumed_losses),
            patience=2,
            start_epoch=2,
            initial_best_state=resumed_model.state_dict(),
            initial_best_epoch=1,
            initial_best_val_loss=0.9,
        )
    )

    assert continuous_steps == [1, 1, 1]
    assert resumed_steps == [1]
    assert resumed_best_epoch == continuous_best_epoch == 1
    assert resumed_best_loss == continuous_best_loss == 0.9


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available on this node.",
            ),
        ),
    ],
)
def test_checkpoint_resume_matches_continuous_training_exactly(
    tmp_path: Path,
    device: str,
) -> None:
    """Resume must reproduce weights, predictions, logs, and RNG exactly."""
    continuous_path = tmp_path / "continuous.bin"
    enzymado.models.common.seed_random_generators(17)
    continuous = _wrapper(device).fit_loader(
        _loader(23),
        "protein_global",
        checkpoint_path=continuous_path,
    )
    continuous_pred = continuous.predict_loader(_loader(41), "protein_global")
    continuous_rng = _rng_sample()

    interrupted_path = tmp_path / "interrupted.bin"
    real_save = enzymado.models.common.save_torch_wrapper_checkpoint

    def save_then_interrupt(wrapper, path) -> None:
        real_save(wrapper, path)
        if wrapper.current_epoch_ == 2:
            raise _InterruptedError

    enzymado.models.common.seed_random_generators(17)
    with (
        mock.patch.object(
            enzymado.models.common,
            "save_torch_wrapper_checkpoint",
            side_effect=save_then_interrupt,
        ),
        pytest.raises(_InterruptedError),
    ):
        _wrapper(device).fit_loader(
            _loader(23),
            "protein_global",
            checkpoint_path=interrupted_path,
        )

    checkpoint = torch.load(
        interrupted_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint.checkpoint_schema_version_ == 3
    assert set(checkpoint.resume_state_) == {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "dataloader_generator_state",
    }
    assert checkpoint.resume_state_["optimizer_state_dict"]["state"]
    assert checkpoint.resume_state_["scheduler_state_dict"] is None
    assert checkpoint.resume_state_["dataloader_generator_state"] is not None
    assert (checkpoint.resume_state_["cuda_rng_state_all"] is not None) == (
        torch.cuda.is_available()
    )

    enzymado.models.common.seed_random_generators(999)
    resumed = _wrapper(device).fit_loader(
        _loader(997),
        "protein_global",
        checkpoint_path=interrupted_path,
    )
    resumed_pred = resumed.predict_loader(_loader(41), "protein_global")
    resumed_rng = _rng_sample()

    assert resumed.current_epoch_ == continuous.current_epoch_ == 4
    assert resumed.training_log_ == continuous.training_log_
    assert resumed.checkpoint_schema_version_ == 3
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
    np.testing.assert_array_equal(resumed_pred, continuous_pred)
    assert resumed_rng[0] == continuous_rng[0]
    np.testing.assert_array_equal(resumed_rng[1], continuous_rng[1])
    torch.testing.assert_close(
        resumed_rng[2],
        continuous_rng[2],
        rtol=0.0,
        atol=0.0,
    )
    assert len(resumed_rng[3]) == len(continuous_rng[3])
    for resumed_state, continuous_state in zip(
        resumed_rng[3],
        continuous_rng[3],
        strict=True,
    ):
        torch.testing.assert_close(
            resumed_state,
            continuous_state,
            rtol=0.0,
            atol=0.0,
        )
