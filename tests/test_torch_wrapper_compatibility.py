"""Compatibility contracts for PyTorch wrapper refactoring."""

from pathlib import Path

import numpy as np
import torch

from enzymado.models import common, nn_blocks
from enzymado.training import (
    attention_wrappers,
    torch_standard_wrappers,
    torch_structured_wrappers,
)


def test_public_wrapper_classes_use_canonical_module_paths() -> None:
    """Checkpoint classes must record their real package modules."""
    expected_modules = {
        torch_standard_wrappers.LinearRegressorTorch: (
            "enzymado.training.torch_standard_wrappers"
        ),
        torch_standard_wrappers.LinearClassifierTorch: (
            "enzymado.training.torch_standard_wrappers"
        ),
        torch_standard_wrappers.MLPRegressorTorch: (
            "enzymado.training.torch_standard_wrappers"
        ),
        torch_standard_wrappers.MLPClassifierTorch: (
            "enzymado.training.torch_standard_wrappers"
        ),
        torch_structured_wrappers.ECMultiHeadClassifierTorch: (
            "enzymado.training.torch_structured_wrappers"
        ),
        torch_structured_wrappers.ResidueBinaryClassifierTorch: (
            "enzymado.training.torch_structured_wrappers"
        ),
        attention_wrappers.AttentionRegressorTorch: (
            "enzymado.training.attention_wrappers"
        ),
        attention_wrappers.AttentionBinaryClassifierTorch: (
            "enzymado.training.attention_wrappers"
        ),
        attention_wrappers.AttentionECMultiHeadClassifierTorch: (
            "enzymado.training.attention_wrappers"
        ),
        attention_wrappers.AttentionResidueBinaryClassifierTorch: (
            "enzymado.training.attention_wrappers"
        ),
    }

    for wrapper_class, expected_module in expected_modules.items():
        assert wrapper_class.__module__ == expected_module


def test_training_checkpoint_roundtrip_preserves_resume_state(tmp_path: Path) -> None:
    """A versioned full-object checkpoint must restore its public wrapper."""
    checkpoint_path = tmp_path / "training_checkpoint.bin"
    trained = torch_standard_wrappers.LinearRegressorTorch(epochs=4, device="cpu")
    trained.model = nn_blocks.LinearModel(input_dim=2, output_dim=1)
    optimizer = torch.optim.Adam(trained.model.parameters(), lr=0.01)
    trained.x_mean_ = np.asarray([1.0, 2.0], dtype=np.float32)
    trained.x_scale_ = np.asarray([3.0, 4.0], dtype=np.float32)

    callback = common.make_torch_checkpoint_callback(
        trained,
        checkpoint_path,
        optimizer=optimizer,
    )
    assert callback is not None
    callback(
        2,
        1,
        0.25,
        {
            key: value.detach().clone()
            for key, value in trained.model.state_dict().items()
        },
    )

    resumed = torch_standard_wrappers.LinearRegressorTorch(epochs=8, device="cpu")
    resumed.model = nn_blocks.LinearModel(input_dim=2, output_dim=1)
    start_epoch = common.maybe_resume_torch_wrapper(resumed, checkpoint_path)

    assert start_epoch == 2
    assert type(resumed) is torch_standard_wrappers.LinearRegressorTorch
    assert resumed.current_epoch_ == 2
    assert resumed.best_epoch_ == 1
    assert resumed.best_val_loss_ == 0.25
    assert resumed.training_log_ == [
        {"epoch": 2, "best_epoch": 1, "best_val_loss": 0.25}
    ]
    np.testing.assert_array_equal(resumed.x_mean_, trained.x_mean_)
    np.testing.assert_array_equal(resumed.x_scale_, trained.x_scale_)
    for key, value in trained.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_legacy_v1_training_checkpoint_still_loads(tmp_path: Path) -> None:
    """A checkpoint created before schema v2 must retain legacy resume behavior."""
    checkpoint_path = tmp_path / "legacy_training_checkpoint.bin"
    legacy = torch_standard_wrappers.LinearRegressorTorch(epochs=4, device="cpu")
    legacy.model = nn_blocks.LinearModel(input_dim=2, output_dim=1)
    legacy.current_epoch_ = 2
    legacy.best_epoch_ = 1
    legacy.best_val_loss_ = 0.25
    torch.save(legacy, checkpoint_path)

    resumed = torch_standard_wrappers.LinearRegressorTorch(epochs=8, device="cpu")
    resumed.model = nn_blocks.LinearModel(input_dim=2, output_dim=1)
    start_epoch = common.maybe_resume_torch_wrapper(resumed, checkpoint_path)

    assert start_epoch == 2
    assert resumed.current_epoch_ == 2
    assert resumed.best_epoch_ == 1
    assert resumed.best_val_loss_ == 0.25
    assert not hasattr(resumed, "checkpoint_schema_version_")
    assert not hasattr(resumed, "resume_state_")
    for key, value in legacy.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
