"""Global attention pooling: the block, its training parity with mean pooling."""

from __future__ import annotations

import contextlib
import inspect
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from enzymado.models.attention_blocks import (  # noqa: E402
    AttentionPredictor,
    AttentionResidueMixer,
    MeanAggregation,
    ResidualAttentionAggregation,
    batch_to_attention_tensors,
)

DIM = 6


def _padded_batch():
    torch.manual_seed(0)
    x = torch.randn(2, 5, DIM)
    padding = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, False]]
    )
    return x, padding


def test_padding_does_not_leak_into_the_pooled_vector() -> None:
    torch.manual_seed(1)
    pooler = ResidualAttentionAggregation(DIM).eval()
    x, padding = _padded_batch()
    noisy = x.clone()
    noisy[0, 3:] = 1e3
    with torch.no_grad():
        clean_out, _, _ = pooler(x, padding)
        noisy_out, _, _ = pooler(noisy, padding)
    assert clean_out.shape == (2, DIM)
    torch.testing.assert_close(clean_out, noisy_out)


def test_silent_attention_leaves_exactly_the_mean() -> None:
    pooler = ResidualAttentionAggregation(DIM).eval()
    with torch.no_grad():
        pooler.cross_attn_layer.out_proj.weight.zero_()
        pooler.cross_attn_layer.out_proj.bias.zero_()
    x, padding = _padded_batch()
    with torch.no_grad():
        pooled, _, _ = pooler(x, padding)
    torch.testing.assert_close(pooled, MeanAggregation()(x, padding))


def test_global_block_shares_the_residue_block_layers() -> None:
    """Same LayerNorm and attention shapes; only the query differs (mean vs residues)."""
    pooled = ResidualAttentionAggregation(DIM)
    residue = AttentionResidueMixer(DIM)
    shapes = lambda module: {  # noqa: E731
        name.split(".", 1)[1]: tuple(p.shape) for name, p in module.named_parameters()
    }
    assert shapes(pooled) == shapes(residue)


def test_predictor_state_dict_holds_block_and_pooled_statistics() -> None:
    model = AttentionPredictor(DIM, (4,), output_dim=1)
    keys = {k for k in model.state_dict() if not k.startswith("head.")}
    assert keys == {
        "pooler.layer_norm.weight",
        "pooler.layer_norm.bias",
        "pooler.cross_attn_layer.in_proj_weight",
        "pooler.cross_attn_layer.in_proj_bias",
        "pooler.cross_attn_layer.out_proj.weight",
        "pooler.cross_attn_layer.out_proj.bias",
        "pooled_mean",
        "pooled_scale",
    }


def test_wrapper_rejects_negative_accumulation() -> None:
    from enzymado.training.attention_wrappers import AttentionRegressorTorch

    with pytest.raises(ValueError, match="gradient_accumulation_steps must be >= 0"):
        AttentionRegressorTorch(
            hidden_dims=(),
            lr=1e-3,
            epochs=1,
            optimizer_name="adamw",
            device="cpu",
            gradient_accumulation_steps=-1,
        )


# --- training parity with the mean-pool route -------------------------------


def _toy_loader(n_proteins: int, batch_size: int, tmp_path):
    import csv

    import numpy as np
    from torch.utils.data import DataLoader

    from enzymado.data_loading.data_registry import TaskConfig
    from enzymado.data_loading.datasets import build_dataset_from_config
    from enzymado.data_loading.features import ResidueEmbeddingStore

    path = tmp_path / "toy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["uniprot_id", "protein_sequence", "binding", "split"]
        )
        writer.writeheader()
        for i in range(n_proteins):
            writer.writerow(
                {
                    "uniprot_id": f"P{i}",
                    "protein_sequence": "A" * (3 + i % 3),
                    "binding": str(i % 2),
                    "split": "train",
                }
            )
    config = TaskConfig(
        name="toy_pooling",
        batch_kind="protein_global",
        label_kind="binary",
        dataset_paths={"random": path},
        target_column="binding",
        index_columns=("uniprot_id",),
    )
    rng = np.random.default_rng(0)
    store = ResidueEmbeddingStore(
        by_id={
            f"P{i}": rng.standard_normal((3 + i % 3, DIM)).astype(np.float32)
            for i in range(n_proteins)
        }
    )
    bundle = build_dataset_from_config(config, store, split_name="train")
    return DataLoader(
        bundle.dataset, batch_size=batch_size, collate_fn=bundle.collator, num_workers=0
    )


def _one_sgd_epoch(loader, accumulation: int):
    """One epoch with SGD: unlike Adam, its step scales with the gradient, so a
    wrongly scaled accumulated gradient would show up as a different update."""
    from enzymado.training.attention_wrappers import AttentionBinaryClassifierTorch

    torch.manual_seed(0)
    wrapper = AttentionBinaryClassifierTorch(
        hidden_dims=(),
        lr=0.5,
        epochs=1,
        optimizer_name="sgd",
        device="cpu",
        gradient_accumulation_steps=accumulation,
    )
    # bf16 autocast rounds differently with a different batch composition, which
    # would hide whether the accumulated step is exact; compare in float32.
    with mock.patch(
        "enzymado.training.attention_pooled.attention_autocast",
        lambda device: contextlib.nullcontext(),
    ):
        wrapper.fit_loader(loader, "protein_global")
    return wrapper


def _max_parameter_gap(a, b) -> float:
    return max(
        (p - q).abs().max().item()
        for (_, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters())
    )


@pytest.mark.parametrize(
    ("n", "micro", "accumulation"),
    [
        (8, 2, 4),  # full groups
        (6, 2, 5),  # one short final group
        (7, 2, 4),  # a short last micro-batch inside a short final group
        (7, 3, 0),  # whole epoch, unequal micro-batches (full-batch linear head)
    ],
)
def test_accumulated_step_matches_one_full_batch_step(tmp_path, n, micro, accumulation):
    full = _one_sgd_epoch(_toy_loader(n, n, tmp_path), accumulation=1)
    accumulated = _one_sgd_epoch(_toy_loader(n, micro, tmp_path), accumulation)
    for (name, a), (_, b) in zip(
        full.model.named_parameters(), accumulated.model.named_parameters()
    ):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)


def test_accumulation_changes_the_update_when_it_should(tmp_path) -> None:
    """Guard against a vacuous pass: four plain micro-batch steps must differ."""
    full = _one_sgd_epoch(_toy_loader(8, 8, tmp_path), accumulation=1)
    plain = _one_sgd_epoch(_toy_loader(8, 2, tmp_path), accumulation=1)
    assert _max_parameter_gap(full, plain) > 1e-2


def test_pooled_standardizer_matches_train_mean_pool_statistics(tmp_path) -> None:
    loader = _toy_loader(9, 3, tmp_path)
    wrapper = _one_sgd_epoch(loader, accumulation=1)
    pooled = []
    with torch.no_grad():
        for batch in loader:
            seq_t, mask_t, _ = batch_to_attention_tensors(
                batch, "protein_global", torch.device("cpu")
            )
            pooled.append(
                MeanAggregation()(wrapper._normalize_sequence(seq_t), ~mask_t)
            )
    pooled = torch.cat(pooled).double()
    torch.testing.assert_close(
        wrapper.model.pooled_mean, pooled.mean(0).float(), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        wrapper.model.pooled_scale,
        pooled.std(0, unbiased=False).float(),
        rtol=1e-4,
        atol=1e-5,
    )
    standardized = (pooled.float() - wrapper.model.pooled_mean) / (
        wrapper.model.pooled_scale
    )
    torch.testing.assert_close(
        standardized.mean(0), torch.zeros(DIM), rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        standardized.std(0, unbiased=False), torch.ones(DIM), rtol=0, atol=1e-4
    )


def _ec_loader(n_proteins: int, batch_size: int):
    """Synthetic EC loader batches with full label masks (equal valid counts)."""
    import types

    import numpy as np
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n_proteins, 5, DIM)).astype(np.float32)
    level_1 = rng.integers(0, 2, size=(n_proteins, 2)).astype(np.float32)
    level_2 = rng.integers(0, 2, size=(n_proteins, 3)).astype(np.float32)
    batches = []
    for start in range(0, n_proteins, batch_size):
        stop = start + batch_size
        batches.append(
            types.SimpleNamespace(
                protein_emb=torch.from_numpy(emb[start:stop]),
                protein_mask=np.ones((stop - start, 5), dtype=bool),
                context_feat=None,
                target={"level_1": level_1[start:stop], "level_2": level_2[start:stop]},
                target_mask={
                    "level_1": np.ones(stop - start, dtype=bool),
                    "level_2": np.ones(stop - start, dtype=bool),
                },
                sample_id=[f"P{i}" for i in range(start, stop)],
            )
        )
    # A real DataLoader: the training protocol records its loader contract.
    return DataLoader(batches, batch_size=1, collate_fn=lambda items: items[0])


def _one_sgd_ec_epoch(loader, accumulation: int):
    from enzymado.training.attention_wrappers import (
        AttentionECMultiHeadClassifierTorch,
    )

    torch.manual_seed(0)
    wrapper = AttentionECMultiHeadClassifierTorch(
        hidden_dims=(),
        lr=0.5,
        epochs=1,
        optimizer_name="sgd",
        device="cpu",
        gradient_accumulation_steps=accumulation,
    )
    with mock.patch(
        "enzymado.training.attention_ec_classification.attention_autocast",
        lambda device: contextlib.nullcontext(),
    ):
        wrapper.fit_loader(loader, "protein_global")
    return wrapper


def test_ec_accumulated_step_matches_one_full_batch_step() -> None:
    full = _one_sgd_ec_epoch(_ec_loader(8, 8), accumulation=1)
    accumulated = _one_sgd_ec_epoch(_ec_loader(8, 2), accumulation=4)
    plain = _one_sgd_ec_epoch(_ec_loader(8, 2), accumulation=1)
    assert full.pooled_standardizer_fitted_
    torch.testing.assert_close(full.model.pooled_scale, accumulated.model.pooled_scale)
    for (name, a), (_, b) in zip(
        full.model.named_parameters(), accumulated.model.named_parameters()
    ):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)
    assert _max_parameter_gap(full, plain) > 1e-2


# --- registry parity -----------------------------------------------------------


def _resolved_params(entry) -> dict:
    """Registry params on top of the wrapper's own constructor defaults."""
    signature = inspect.signature(entry["wrap"].__init__).parameters
    params = {
        name: p.default
        for name, p in signature.items()
        if p.default is not inspect.Parameter.empty
    }
    params.update(entry["params"])
    return params


@pytest.mark.parametrize(
    ("task_type", "label_kind", "name"),
    [
        ("regression", "scalar_regression", "linear_regressor"),
        ("regression", "scalar_regression", "mlp_regressor"),
        ("classification", "binary", "linear_classification"),
        ("classification", "binary", "mlp_classification"),
        ("classification", "multilabel_ec", "linear_classification"),
        ("classification", "multilabel_ec", "mlp_classification"),
    ],
)
def test_global_attention_trains_like_its_mean_pool_counterpart(
    task_type, label_kind, name
) -> None:
    """Everything but the attention block matches: optimizer, schedule, batch."""
    import enzymado.training.model_registry as registry

    mean_pool = _resolved_params(registry.get_model_map(task_type, label_kind)[name])
    attention = _resolved_params(
        registry.get_attention_model_map(task_type, label_kind)[f"attention_{name}"]
    )
    linear = name.startswith("linear")
    for key in (
        "lr",
        "epochs",
        "optimizer_name",
        "early_stopping_patience",
        "early_stopping_min_delta",
    ):
        assert attention[key] == mean_pool[key], key
    # The mean-pool linear head has no hidden layer, dropout or weight decay.
    for key, default in (("weight_decay", 0.0), ("dropout", 0.0)):
        assert attention.get(key, default) == mean_pool.get(key, default), key
    assert tuple(attention["hidden_dims"]) == tuple(
        mean_pool.get("hidden_dims", ())
    )

    accumulation = attention["gradient_accumulation_steps"]
    full_batch_linear = linear and label_kind != "multilabel_ec"
    if full_batch_linear:
        assert "batch_size" not in mean_pool  # trained on the whole split per step
        assert accumulation == 0
    else:
        loader_batch = 16  # train.py default for the attention route
        assert loader_batch * accumulation == mean_pool["batch_size"]
