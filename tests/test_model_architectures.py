"""Validate model parameter counts and output shapes with torchinfo summaries."""

# ``x`` matches the estimator notation used by the model blocks under test.
# ruff: noqa: N806

from __future__ import annotations

import pytest
import torch

import enzymado.models.nn_blocks as nn_blocks

torchinfo = pytest.importorskip("torchinfo")

BATCH = 4
INPUT_DIM = 8


def _summary(model, input_size):
    """Run a torchinfo summary without touching the global RNG stream.

    ``input_size=`` makes torchinfo build the dummy tensor with ``torch.rand``,
    which advances the global RNG; an explicit ``input_data=`` tensor does not.
    """
    model.eval()
    return torchinfo.summary(model, input_data=torch.zeros(input_size), verbose=0)


def _expected_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_linear_model_summary_matches_parameter_count() -> None:
    model = nn_blocks.LinearModel(INPUT_DIM, 3)
    info = _summary(model, (BATCH, INPUT_DIM))
    assert info.total_params == _expected_params(model) == 8 * 3 + 3


def test_mlp_model_summary_matches_parameter_count() -> None:
    model = nn_blocks.MLPModel(INPUT_DIM, [16, 4], 3, dropout=0.1)
    info = _summary(model, (BATCH, INPUT_DIM))
    expected = (8 * 16 + 16) + (16 * 4 + 4) + (4 * 3 + 3)
    assert info.total_params == _expected_params(model) == expected


def test_shared_backbone_summary_matches_parameter_count() -> None:
    model = nn_blocks.SharedMLPBackbone(INPUT_DIM, [16, 4])
    info = _summary(model, (BATCH, INPUT_DIM))
    assert info.total_params == _expected_params(model) == (8 * 16 + 16) + (16 * 4 + 4)
    assert model.output_dim == 4


def test_multi_head_model_summary_matches_parameter_count() -> None:
    output_dims = {"level_1": 3, "level_2": 2}
    model = nn_blocks.MultiHeadMLPModel(INPUT_DIM, [16], output_dims)
    info = _summary(model, (BATCH, INPUT_DIM))
    expected = (8 * 16 + 16) + (16 * 3 + 3) + (16 * 2 + 2)
    assert info.total_params == _expected_params(model) == expected


def test_empty_backbone_is_identity() -> None:
    model = nn_blocks.SharedMLPBackbone(INPUT_DIM, [])
    assert model.output_dim == INPUT_DIM
    assert _expected_params(model) == 0


def test_forward_shapes_match_declared_dims() -> None:
    x = torch.zeros(BATCH, INPUT_DIM)
    assert nn_blocks.LinearModel(INPUT_DIM, 3)(x).shape == (BATCH, 3)
    assert nn_blocks.MLPModel(INPUT_DIM, [16], 3)(x).shape == (BATCH, 3)

    heads = nn_blocks.MultiHeadMLPModel(INPUT_DIM, [16], {"a": 3, "b": 2})(x)
    assert heads["a"].shape == (BATCH, 3)
    assert heads["b"].shape == (BATCH, 2)


def test_summary_does_not_consume_the_global_rng() -> None:
    """A summary must not shift training reproducibility."""
    model = nn_blocks.MLPModel(INPUT_DIM, [16], 3, dropout=0.5)

    torch.manual_seed(0)
    baseline = torch.randn(3)

    torch.manual_seed(0)
    _summary(model, (BATCH, INPUT_DIM))
    after = torch.randn(3)

    assert torch.equal(baseline, after)
