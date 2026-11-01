"""Laptop tests for the Appendix B search, model defaults and submission flow.

Example:
    pytest tests/test_mlp_hpo.py
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scripts.mlp_hpo import aggregate_hpo_results, hpo_job_submitter, run_hpo
import torch
from torch import nn

from enzymado import paths_and_constants
from enzymado.models import nn_blocks, training_protocol
from enzymado.training import model_registry, torch_standard_wrappers

_P = paths_and_constants.Paths
ROOT = _P.project_root
SEARCH_SPACE = {
    "activation": ["relu", "silu", "swiglu"],
    "num_layers": [1, 2, 3],
    "hidden_dim": [512, 1024, 2048, 4096],
    "dropout_prob": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5],
    "normalize": [False, True],
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    "weight_decay": [0.3, 0.1, 0.03, 0.01, 0],
}


@pytest.fixture(autouse=True)
def single_cpu_thread() -> object:
    """Avoid threading overhead for the tiny synthetic networks."""
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def _generate(output: Path, *extra: str) -> list[dict[str, object]]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/mlp_hpo/generate_configs.py"),
            "--output",
            str(output),
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text())


def test_lhs_regenerates_committed_configs_exactly(tmp_path: Path) -> None:
    """The historical seed, ordering and random draws must remain unchanged."""
    output = tmp_path / "configs.json"
    generated = _generate(output)
    assert generated == run_hpo.load_configs(run_hpo.DEFAULT_CONFIG_FILE)
    assert output.read_bytes() == run_hpo.DEFAULT_CONFIG_FILE.read_bytes()
    assert len(generated) == 400
    assert [config["config_index"] for config in generated] == list(range(400))
    assert len({tuple(config.items()) for config in generated}) == 400
    combinations = {tuple(config[key] for key in SEARCH_SPACE) for config in generated}
    assert len(combinations) == 400
    for config in generated:
        for key, values in SEARCH_SPACE.items():
            assert config[key] in values
        assert config["batch_size"] == 256
        assert config["epochs"] == 20
        assert config["optimizer"] == "adamw"


def test_config_generation_is_reproducible(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first.json", "--n-configs", "15")
    repeated = _generate(tmp_path / "repeated.json", "--n-configs", "15")
    other_seed = _generate(tmp_path / "other.json", "--n-configs", "15", "--seed", "43")
    assert first == repeated
    assert first != other_seed


@pytest.mark.parametrize("hidden_dims", [(), (2048,), (16, 4)])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_default_mlp_matches_previous_architecture_and_rng(
    hidden_dims: tuple[int, ...], dropout: float
) -> None:
    """Compare initialization, keys and outputs with the pre-HPO SiLU stack."""
    torch.manual_seed(123)
    layers = []
    in_dim = 8
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    previous = nn.Module()
    previous.net = nn.Sequential(*layers)
    previous_rng = torch.get_rng_state()

    torch.manual_seed(123)
    current = nn_blocks.MLPModel(8, hidden_dims, 1, dropout=dropout)
    assert torch.equal(torch.get_rng_state(), previous_rng)
    assert current.state_dict().keys() == previous.state_dict().keys()
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, previous.state_dict()[name], rtol=0, atol=0)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    torch.manual_seed(456)
    expected = previous.net(x)
    torch.manual_seed(456)
    torch.testing.assert_close(current(x), expected, rtol=0, atol=0)


def test_default_registry_and_resume_contract_are_unchanged() -> None:
    info = model_registry.MODEL_MAP["regression"]["mlp_regressor"]
    assert info["wrap"] is torch_standard_wrappers.MLPRegressorTorch
    assert info["params"] == {
        "hidden_dims": (2048,),
        "lr": 1e-3,
        "epochs": 100,
        "optimizer_name": "adamw",
        "weight_decay": 0.1,
        "dropout": 0.1,
        "batch_size": 256,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-4,
    }
    wrapper = info["wrap"](**info["params"], device="cpu")
    contract = training_protocol.resume_contract(wrapper, None)
    expected = {**info["params"], "hidden_dims": [2048]}
    assert contract["params"] == expected


@pytest.mark.parametrize("activation", ["relu", "silu", "swiglu"])
def test_hpo_options_flow_into_current_model(activation: str) -> None:
    config = {
        **run_hpo.load_configs(run_hpo.DEFAULT_CONFIG_FILE)[0],
        "activation": activation,
        "normalize": True,
        "num_layers": 3,
    }
    registry_before = copy.deepcopy(model_registry.MODEL_MAP)
    info = run_hpo.build_model_info(config, device="cpu")
    wrapper = info["wrap"](**info["params"])
    x = np.arange(32, dtype=np.float32).reshape(8, 4)
    wrapper.fit(x, np.linspace(-1, 1, 8, dtype=np.float32))
    layers = list(wrapper.model.net)
    assert sum(isinstance(layer, nn.LayerNorm) for layer in layers) == 3
    assert sum(isinstance(layer, nn.Linear) for layer in layers) == 4
    linears = [layer for layer in layers if isinstance(layer, nn.Linear)]
    width = 352 if activation == "swiglu" else 512
    assert linears[0].out_features == width * (2 if activation == "swiglu" else 1)
    assert linears[-1].in_features == width
    assert wrapper.current_epoch_ == 20
    assert wrapper.best_epoch_ is None
    assert wrapper.x_mean_ is None
    assert wrapper.predict(x).shape == (8,)
    assert model_registry.MODEL_MAP == registry_before


def test_one_hpo_config_trains_and_reports_validation_metrics() -> None:
    config = run_hpo.load_configs(run_hpo.DEFAULT_CONFIG_FILE)[0]
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 4)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    first = run_hpo.fit_and_score(config, x[:12], y[:12], x[12:], y[12:], device="cpu")
    second = run_hpo.fit_and_score(config, x[:12], y[:12], x[12:], y[12:], device="cpu")
    assert first == second
    assert first["completed_epochs"] == 20
    assert first["best_epoch"] is None
    assert first["split_variant"] == "similarity"
    assert first["task_name"] == "km"
    assert first["embedding"] == "esmc_300m"
    assert all(np.isfinite(first[key]) for key in ("val_R2", "val_MAE", "val_Kendall"))


def test_hpo_cli_uses_historical_split_and_writes_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the runner with synthetic prepared data and no embedding files."""
    x = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
    cleanup_calls = []
    payload = {
        field.name: None
        for field in dataclasses.fields(run_hpo.task_data.PooledTaskData)
    }
    payload.update(
        x_train=x[:4],
        y_train=x[:4, 0],
        x_val=x[4:],
        y_val=x[4:, 0],
        cleanup=lambda: cleanup_calls.append(True),
    )
    prepared = run_hpo.task_data.PooledTaskData(**payload)
    calls = []

    def load_task_data(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return prepared

    monkeypatch.setattr(run_hpo.task_data, "load_task_data", load_task_data)
    monkeypatch.setattr(
        run_hpo.stores,
        "load_embeddings",
        lambda _: {"esmc_300m": {"path": "synthetic.safetensors", "dim": 4}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hpo.py",
            "--config-index",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ],
    )
    run_hpo.main()
    assert calls == [
        (("km", "synthetic.safetensors", "similarity"), {"embedding_dim": 4})
    ]
    assert cleanup_calls == [True]
    artifact = tmp_path / "hpo_metrics/mlp_hpo_000.json"
    result = json.loads(artifact.read_text())
    assert result["completed_epochs"] == 20
    assert result["run_id"] == "mlp_hpo_000"
    assert list(tmp_path.rglob("*.json")) == [artifact]
    assert not list(tmp_path.rglob("*.bin"))


def test_aggregation_selects_r2_instead_of_mae_or_kendall(tmp_path: Path) -> None:
    for index, (r2, mae, tau) in enumerate([(0.2, 0.1, 0.9), (0.8, 0.4, 0.5)]):
        record = {
            "run_id": f"mlp_hpo_{index:03d}",
            "val_R2": r2,
            "val_MAE": mae,
            "val_Kendall": tau,
        }
        (tmp_path / f"mlp_hpo_{index:03d}.json").write_text(json.dumps(record))
    ranked = aggregate_hpo_results.aggregate(tmp_path, expected_count=2)
    assert list(ranked["run_id"]) == ["mlp_hpo_001", "mlp_hpo_000"]
    assert ranked.iloc[0]["val_MAE"] == 0.4
    assert ranked.iloc[0]["val_Kendall"] == 0.5
    with pytest.raises(ValueError, match="Expected 400 runs, found 2"):
        aggregate_hpo_results.aggregate(tmp_path)


def test_submitter_builds_400_job_array_without_side_effects(tmp_path: Path) -> None:
    output = tmp_path / "runs with spaces"
    args = hpo_job_submitter.build_parser().parse_args(
        ["--output-dir", str(output), "--account", "test-account"]
    )
    command = hpo_job_submitter.build_command(args)
    assert command[0] == "sbatch"
    assert "--array=0-399%10" in command
    assert "--account=test-account" in command
    assert str(output) in command
    assert command[command.index("-u") + 1] == str(ROOT / "scripts/mlp_hpo/run_hpo.py")
    assert not output.exists()


def test_slurm_array_passes_the_selected_index(tmp_path: Path) -> None:
    """Execute the actual shell entrypoint with a harmless fake training command."""
    script = ROOT / "scripts/mlp_hpo/run_hpo.slurm"
    result = subprocess.run(
        ["bash", str(script), sys.executable, "-c", "import sys; print(sys.argv[1:])"],
        env={"PATH": "/usr/bin:/bin", "SLURM_ARRAY_TASK_ID": "17"},
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "['--config-index', '17']"
