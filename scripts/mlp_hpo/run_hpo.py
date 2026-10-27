"""Run one Appendix B MLP configuration through the current training package.

The historical SLURM script on origin/hpo did not override the split: it used
the benchmark card's K_M default, similarity_mmseqs2_80_10_10 (80/10/10).
This runner explicitly selects that split, K_M, esmc_300m mean-pooled inputs,
and the original 20-epoch AdamW budget without early stopping. Configurations
are ranked afterwards by validation R^2; MAE and Kendall's tau are also saved.
Features stay unscaled as in origin/hpo. Targets and ligand features come from
the current K_M loader, including its standardized target column. The original
SLURM allowlist was empty, so no embedding-intersection filter is applied.

Example:
    python scripts/mlp_hpo/run_hpo.py --config-index 0 --device cpu
"""

from __future__ import annotations

import argparse
import collections.abc
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import scipy.stats
import sklearn.metrics

from enzymado import paths_and_constants, types
from enzymado.data_loading import stores, task_data
from enzymado.models import common
from enzymado.training import model_registry, torch_standard_wrappers

DEFAULT_CONFIG_FILE = Path(__file__).with_name("configs_400_lhs.json")
DEFAULT_OUTPUT_DIR = paths_and_constants.Paths.project_root / "mlp_hpo_runs"
TASK_NAME = "km"
EMBEDDING_NAME = "esmc_300m"
SPLIT_VARIANT = "similarity"


def load_configs(path: Path) -> list[dict[str, object]]:
    """Read a config list and reject missing, duplicated or reordered indices."""
    configs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configs, list) or not configs:
        raise ValueError("Expected a non-empty JSON list of HPO configurations.")
    for index, config in enumerate(configs):
        if not isinstance(config, dict) or config.get("config_index") != index:
            raise ValueError(f"Expected config_index={index} at list position {index}.")
    return configs


def build_model_info(
    config: collections.abc.Mapping[str, object],
    *,
    device: str | None = None,
) -> types.ModelInfo:
    """Translate the historical config schema without mutating any registry.

    ``num_layers`` is expressed using the current wrapper's ``hidden_dims``.
    No early stopping or feature standardization was used by the original HPO.
    The benchmark registry and the wrapper's defaults remain unchanged.
    """
    if (
        config["optimizer"] != "adamw"
        or config["batch_size"] != 256
        or config["epochs"] != 20
    ):
        raise ValueError("Appendix B requires AdamW, batch size 256 and 20 epochs.")
    params: dict[str, object] = {
        "hidden_dims": (int(config["hidden_dim"]),) * int(config["num_layers"]),
        "activation": str(config["activation"]),
        "normalize": config["normalize"],
        "dropout": float(config["dropout_prob"]),
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "optimizer_name": "adamw",
        "batch_size": 256,
        "epochs": 20,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "standardize_features": False,
    }
    if device is not None:
        params["device"] = device
    return {"wrap": torch_standard_wrappers.MLPRegressorTorch, "params": params}


def fit_and_score(
    config: collections.abc.Mapping[str, object],
    x_train: types.Array,
    y_train: types.Array,
    x_val: types.Array,
    y_val: types.Array,
    *,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    """Fit all 20 epochs and score the final model on validation data only."""
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("HPO requires mean-pooled, two-dimensional input arrays.")
    if len(y_train) == 0 or len(y_val) < 2:
        raise ValueError("HPO requires training data and at least two validation rows.")
    common.seed_random_generators(seed)
    model = model_registry.train_model(
        build_model_info(config, device=device), x_train, y_train
    )
    prediction = model.predict(x_val)
    target = np.asarray(y_val).reshape(-1)
    r2 = float(sklearn.metrics.r2_score(target, prediction))
    mae = float(sklearn.metrics.mean_absolute_error(target, prediction))
    tau = float(scipy.stats.kendalltau(target, prediction).statistic)
    if not math.isfinite(r2) or not math.isfinite(mae):
        raise ValueError("HPO produced non-finite validation R^2 or MAE.")
    return {
        **config,
        "run_id": f"mlp_hpo_{int(config['config_index']):03d}",
        "task_name": TASK_NAME,
        "embedding": EMBEDDING_NAME,
        "pooling": "mean",
        "split_variant": SPLIT_VARIANT,
        "training_seed": seed,
        "standardize_features": False,
        "val_R2": r2,
        "val_MAE": mae,
        "val_Kendall": tau if math.isfinite(tau) else None,
        "best_epoch": model.best_epoch_,
        "completed_epochs": model.current_epoch_,
    }


def main() -> None:
    """Load the current pooled K_M arrays and write one isolated metrics JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=(
            paths_and_constants.Paths.project_root
            / "embeddings_models_output_mean_pooled.json"
        ),
        help="Mean-pooled embedding manifest containing esmc_300m.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="For example cpu or cuda.")
    args = parser.parse_args()
    configs = load_configs(args.config_file)
    if not 0 <= args.config_index < len(configs):
        parser.error(f"--config-index must be between 0 and {len(configs) - 1}.")
    embedding = stores.load_embeddings(str(args.embeddings_json))[EMBEDDING_NAME]
    data = task_data.load_task_data(
        TASK_NAME,
        embedding["path"],
        SPLIT_VARIANT,
        embedding_dim=embedding["dim"],
    )
    try:
        if not isinstance(data, task_data.PooledTaskData):
            raise TypeError("K_M HPO requires pooled task data.")
        if any(
            value is None
            for value in (data.x_train, data.y_train, data.x_val, data.y_val)
        ):
            raise ValueError("K_M training or validation arrays are missing.")
        result = fit_and_score(
            configs[args.config_index],
            data.x_train,
            data.y_train,
            data.x_val,
            data.y_val,
            seed=args.seed,
            device=args.device,
        )
    finally:
        data.cleanup()
    metrics_dir = args.output_dir / "hpo_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    destination = metrics_dir / f"{result['run_id']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
