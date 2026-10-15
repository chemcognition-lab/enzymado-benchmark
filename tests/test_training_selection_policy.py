"""Behavioral checks for trained-epoch selection and strict resume contracts."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import enzymado.models.common as common
import enzymado.training.torch_standard_wrappers as wrappers


def test_tied_ranking_keeps_trained_weights_and_records_baseline(tmp_path):
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    wrapper = wrappers.LinearClassifierTorch(
        epochs=3, lr=0.1, early_stopping_patience=1, device="cpu"
    )
    with mock.patch.object(type(wrapper), "_selection_objective", return_value=-1.0):
        wrapper.fit(x, y, x_val=x, y_val=y, checkpoint_path=tmp_path / "state.bin")
    assert wrapper.best_epoch_ == 1
    assert wrapper.baseline_score_ == -1.0
    assert wrapper.training_log_[-1]["improved_over_initialization"] is False
    assert wrapper.training_log_[-1]["selection_objective"] == "negative_auprc"
    initial = torch.load(tmp_path / "initial_model.pt", weights_only=True)
    assert any(
        not torch.equal(initial[key], value)
        for key, value in wrapper.model.state_dict().items()
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_selection_rejects_nonfinite_predictions(bad):
    with pytest.raises(FloatingPointError, match="Non-finite"):
        common.validation_objective([0, 1, 0, 1], [bad, 0.9, 0.1, 0.8], "auprc")


def test_constant_predictions_are_not_numerical_failures():
    assert common.validation_objective([0, 1, 0, 1], [0.5] * 4, "auprc") == -0.5
    assert common.validation_objective([1, 2, 3], [0.5] * 3, "kendall_tau") is None
    with pytest.raises(ValueError, match="labels"):
        common.validation_objective([1, 1], [0.2, 0.9], "auprc")


def test_explicit_unknown_targets_are_masked_before_finiteness_check():
    assert (
        common.validation_objective(
            [0, 1, np.nan], [0.1, 0.9, np.nan], "auprc", [True, True, False]
        )
        == -1.0
    )


def test_no_defined_trained_epoch_is_a_failed_run():
    with pytest.raises(ValueError, match="No trained epoch"):
        common.run_training_with_early_stopping(
            torch.nn.Linear(1, 1), 3, lambda: None, lambda: None, patience=2
        )


def test_undefined_epoch_can_be_followed_by_eligible_epoch():
    scores = iter([None, None, -0.5, -0.4])
    best, score = common.run_training_with_early_stopping(
        torch.nn.Linear(1, 1), 3, lambda: None, lambda: next(scores), patience=2
    )
    assert (best, score) == (2, -0.5)


def test_optimizer_rejects_bad_gradients_before_update():
    model = torch.nn.Linear(1, 1)
    optimizer = common.make_optimizer(model.parameters(), "sgd", 0.1)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer.step()
    assert torch.equal(model.weight, before)


@pytest.mark.parametrize("change", ["metric", "patience", "epochs", "legacy"])
def test_resume_rejects_changed_contract_before_loading_weights(tmp_path, change):
    x = np.array([[-1.0], [1.0]], dtype=np.float32)
    y = np.array([-1.0, 1.0], dtype=np.float32)
    path = tmp_path / "checkpoint.bin"
    model = wrappers.LinearRegressorTorch(epochs=1, device="cpu")
    model.fit(x, y, checkpoint_path=path)
    fresh = wrappers.LinearRegressorTorch(epochs=1, device="cpu")
    if change == "metric":
        model.training_contract_["selection"]["metric"] = "validation_loss"
        common.save_torch_wrapper_checkpoint(model, path)
    elif change == "patience":
        fresh.early_stopping_patience = 7
    elif change == "epochs":
        fresh.epochs = 3
    else:
        del model.training_contract_
        common.save_torch_wrapper_checkpoint(model, path)
    with pytest.raises(ValueError, match="Incompatible training checkpoint"):
        fresh.fit(x, y, checkpoint_path=path)


def test_pipeline_manifest_is_part_of_checkpoint_contract(tmp_path: Path):
    x = np.array([[-1.0], [1.0]], dtype=np.float32)
    y = np.array([-1.0, 1.0], dtype=np.float32)
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text('{"data": "original"}')
    model = wrappers.LinearRegressorTorch(epochs=1, device="cpu")
    model.fit(x, y, checkpoint_path=tmp_path / "state.bin")
    manifest.write_text('{"data": "changed"}')
    fresh = wrappers.LinearRegressorTorch(epochs=1, device="cpu")
    with pytest.raises(ValueError, match="Incompatible training checkpoint"):
        fresh.fit(x, y, checkpoint_path=tmp_path / "state.bin")


def test_initial_weights_cannot_count_as_a_trained_artifact(tmp_path):
    import enzymado.artifacts.results_layout

    torch.save({"weight": torch.ones(1)}, tmp_path / "initial_model.pt")
    assert enzymado.artifacts.results_layout.find_model_artifact(tmp_path) is None


def test_logits_cannot_hide_infinity_behind_sigmoid():
    wrapper = wrappers.LinearClassifierTorch(device="cpu")
    with pytest.raises(FloatingPointError, match="validation logits"):
        wrapper._selection_objective(
            torch.tensor([[float("inf")], [-1.0]]), torch.tensor([[1.0], [0.0]])
        )


def test_changed_pipeline_data_preserves_original_manifest(tmp_path):
    import enzymado.artifacts.reproducibility as provenance

    dataset = tmp_path / "data.csv"
    embedding = tmp_path / "embeddings.bin"
    hp = tmp_path / "hparams.json"
    dataset.write_text("original data")
    embedding.write_bytes(b"embeddings")
    hp.write_text("{}")
    kwargs = dict(
        dataset_path=dataset,
        embedding_path=embedding,
        hparams_path=hp,
        split_manifest={"train": {"count": 1}},
        deterministic_algorithms=False,
    )
    with mock.patch.object(
        provenance,
        "build_environment_manifest",
        return_value={
            "packages": {},
            "source_tree_sha256": "fixed",
            "cuda_runtime": None,
            "gpu_names": [],
        },
    ):
        path = provenance.write_run_manifest(tmp_path, **kwargs)
        original = path.read_bytes()
        (tmp_path / "training_checkpoint.bin").write_bytes(b"checkpoint")
        provenance.write_run_manifest(tmp_path, **kwargs)
        assert path.read_bytes() == original
        dataset.write_text("changed data")
        with pytest.raises(ValueError, match="data, splits or runtime changed"):
            provenance.write_run_manifest(tmp_path, **kwargs)
        assert path.read_bytes() == original


def test_validation_resume_matches_uninterrupted_selection(tmp_path):
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

    def fit(path):
        return wrappers.MLPRegressorTorch(
            hidden_dims=(4,),
            epochs=4,
            lr=0.01,
            early_stopping_patience=3,
            dropout=0.2,
            batch_size=2,
            device="cpu",
        ).fit(x, y, x_val=x, y_val=y, checkpoint_path=path)

    common.seed_random_generators(12)
    continuous = fit(tmp_path / "continuous.bin")
    save = common.save_torch_wrapper_checkpoint

    def interrupt(wrapper, path):
        save(wrapper, path)
        if wrapper.current_epoch_ == 2:
            raise InterruptedError("interrupted")

    common.seed_random_generators(12)
    with mock.patch.object(
        common, "save_torch_wrapper_checkpoint", side_effect=interrupt
    ):
        with pytest.raises(InterruptedError):
            fit(tmp_path / "interrupted.bin")
    resumed = fit(tmp_path / "interrupted.bin")
    assert resumed.training_log_ == continuous.training_log_
    assert resumed.baseline_score_ == continuous.baseline_score_
    assert resumed.best_epoch_ == continuous.best_epoch_ >= 1
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key], value, rtol=0, atol=0
        )


def test_streaming_auprc_matches_the_leaderboard_estimator_on_all_scores():
    from enzymado.evaluation import metrics_registry

    rng = np.random.default_rng(0)
    scores = rng.random((300, 7))
    labels = (rng.random((300, 7)) < 0.2).astype(np.float64)
    label_mask = rng.random((300, 7)) < 0.9
    row_mask = rng.random(300) < 0.8
    streaming = common.StreamingAUPRC(torch.device("cpu"))
    for start in range(0, 300, 64):
        rows = slice(start, start + 64)
        streaming.update(
            torch.from_numpy(scores[rows]).float(),
            torch.from_numpy(labels[rows]),
            torch.from_numpy(label_mask[rows] & row_mask[rows, None]),
        )
    keep = label_mask & row_mask[:, None]
    expected = metrics_registry.auprc_from_scores(
        labels[keep], scores.astype(np.float32).astype(np.float64)[keep]
    )
    assert streaming.value() == expected


def test_streaming_auprc_broadcasts_a_row_mask():
    from enzymado.evaluation import metrics_registry

    scores = torch.tensor([[0.9, 0.2], [0.8, 0.7], [0.1, 0.6]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    streaming = common.StreamingAUPRC(torch.device("cpu"))
    streaming.update(scores, labels, torch.tensor([True, True, False]))
    expected = metrics_registry.auprc_from_scores(
        labels[:2].numpy().ravel(), scores[:2].double().numpy().ravel()
    )
    assert streaming.value() == expected


def test_ec_heads_select_on_validation_auprc():
    from enzymado.models import training_protocol
    from enzymado.training import attention_wrappers, torch_structured_wrappers

    for wrapper in (
        torch_structured_wrappers.ECMultiHeadClassifierTorch,
        attention_wrappers.AttentionECMultiHeadClassifierTorch,
    ):
        protocol = training_protocol.selection_protocol(wrapper)
        assert protocol["metric"] == "auprc"
        assert protocol["objective"] == "negative_auprc"
