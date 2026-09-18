"""XGBoost wrappers for standard, residue-level, and flat EC tasks."""

from __future__ import annotations

import time
from pathlib import Path

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    from xgboost.core import XGBoostError
except Exception:  # noqa: BLE001 - see below
    # Deliberately broad. An absent package raises ModuleNotFoundError, but an
    # *installed* xgboost whose native library cannot load raises XGBoostError
    # (e.g. missing libomp.dylib on macOS) — and XGBoostError is only importable
    # from inside this try, so it cannot be named here. Catching just
    # ModuleNotFoundError let that case propagate and break the import of every
    # module that reaches model_registry, taking run_benchmark.py down with it.
    xgb = None
    XGBClassifier = None
    XGBRegressor = None
    XGBoostError = None

import numpy as np
import torch
import joblib

from models.common import pad_1d_array_list


def _progress_log(message: str, *, label: str | None = None) -> None:
    """Emit a timestamped training-progress log line from inside native wrappers."""
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}] "
    if label:
        prefix += f"[{label}] "
    print(prefix + message, flush=True)


def _should_log_progress(step_index: int, total_steps: int | None) -> bool:
    """Log the first, last, and roughly 20%%-spaced intermediate steps."""
    if total_steps is None or total_steps <= 0:
        return step_index == 1 or step_index % 100 == 0
    interval = max(1, total_steps // 5)
    return step_index == 1 or step_index == total_steps or step_index % interval == 0


def _estimate_gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _as_xgb_residue_features(value) -> np.ndarray:
    """Return residue features as float32 for XGBoost's NumPy input path."""
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    array = np.asarray(value)
    return array.astype(np.float32, copy=False)


def _is_xgb_cuda_oom(exc: Exception) -> bool:
    if XGBoostError is not None and not isinstance(exc, XGBoostError):
        return False
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "memory allocation error",
            "cudaerrormemoryallocation",
            "out of memory",
            "device_vector.cu",
        )
    )


def _is_xgb_cuda_failure(exc: Exception) -> bool:
    """Heuristic for any XGBoost failure that should trigger a CPU fallback.

    Catches OOM plus the broader class of CUDA / device-mismatch errors that
    XGBoost 3.x can surface from QuantileDMatrix(iter, ...) when fed CPU numpy
    arrays while the booster targets cuda — a vague C++ `vector::_M_range_check`
    is the typical symptom.
    """
    if XGBoostError is not None and not isinstance(exc, XGBoostError):
        return False
    if _is_xgb_cuda_oom(exc):
        return True
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "vector::_m_range_check",
            "cuda",
            "mismatched device",
            "device_vector",
            "gpu",
        )
    )


def _atomic_joblib_dump(payload, path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path_obj)


def _save_xgb_checkpoint(wrapper, path: str | Path | None) -> None:
    if path is None:
        return
    _atomic_joblib_dump(wrapper, path)


def _maybe_resume_xgb_wrapper(wrapper, path: str | Path | None) -> bool:
    if path is None:
        return False
    path_obj = Path(path)
    if not path_obj.exists():
        return False
    loaded = joblib.load(path_obj)
    if not isinstance(loaded, wrapper.__class__):
        raise TypeError(
            f"Checkpoint at {path_obj} contains {type(loaded).__name__}, "
            f"expected {wrapper.__class__.__name__}."
        )
    current_device = getattr(wrapper, "device", None)
    wrapper.__dict__.update(loaded.__dict__)
    if current_device is not None:
        wrapper.device = current_device
    return True


class XGBMultiOutputFlatECExperimental:
    """Experimental multi-output XGBoost baseline for dense flat EC labels.

    This wrapper is intended for benchmarking against the current one-model-per-column
    implementation. It requires a dense multi-hot label matrix with no unknown entries,
    so callers should filter rows with incomplete labels before using it.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.1,
        max_bin=128,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        tree_method="hist",
        multi_strategy="multi_output_tree",
    ):
        if xgb is None:
            raise ImportError("xgboost is required to use XGBMultiOutputFlatECExperimental.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.multi_strategy = multi_strategy
        self.booster = None
        self.n_outputs_: int | None = None

    @staticmethod
    def _prepare_dense_targets(y, target_mask=None) -> np.ndarray:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(
                "XGBMultiOutputFlatECExperimental expects 2D dense multi-output targets, "
                f"got shape {y_np.shape}."
            )

        if target_mask is None:
            return y_np

        mask_np = np.asarray(target_mask, dtype=bool)
        if mask_np.shape != y_np.shape:
            raise ValueError(
                "target_mask shape must match y shape for XGBMultiOutputFlatECExperimental. "
                f"Received mask={mask_np.shape}, y={y_np.shape}."
            )
        if not np.all(mask_np):
            raise ValueError(
                "XGBMultiOutputFlatECExperimental requires dense labels with no unknown entries. "
                "Filter rows/columns with incomplete EC labels before fitting."
            )
        return y_np

    def fit(self, X, y, target_mask=None, X_val=None, y_val=None, target_mask_val=None):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(
                "XGBMultiOutputFlatECExperimental expects pooled 2D inputs, "
                f"got shape {X_np.shape}."
            )
        y_np = self._prepare_dense_targets(y, target_mask)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows for "
                "XGBMultiOutputFlatECExperimental. "
                f"Received X={X_np.shape[0]}, y={y_np.shape[0]}."
            )

        dtrain = xgb.QuantileDMatrix(X_np, label=y_np)
        params = {
            "objective": "binary:logistic",
            "tree_method": self.tree_method,
            "device": self.device,
            "multi_strategy": self.multi_strategy,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "nthread": self.n_jobs,
            "seed": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }

        evals = []
        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            y_val_np = self._prepare_dense_targets(y_val, target_mask_val)
            if X_val_np.ndim != 2:
                raise ValueError(f"X_val must be 2D, got shape {X_val_np.shape}")
            if X_val_np.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows for "
                    "XGBMultiOutputFlatECExperimental. "
                    f"Received X_val={X_val_np.shape[0]}, y_val={y_val_np.shape[0]}."
                )
            dval = xgb.QuantileDMatrix(X_val_np, label=y_val_np, ref=dtrain)
            evals.append((dval, "validation"))

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            verbose_eval=False,
        )
        self.n_outputs_ = y_np.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        if self.booster is None or self.n_outputs_ is None:
            raise ValueError("XGBMultiOutputFlatECExperimental must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")
        dtest = xgb.QuantileDMatrix(X_np)
        pred = self.booster.predict(dtest)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(X_np.shape[0], self.n_outputs_)
        return (pred >= 0.5).astype(np.float32, copy=False)


class XGBFlatECBlockClassifier:
    """Multi-output XGBoost over contiguous blocks of flat EC columns.

    This keeps Ben's idea of training multiple EC outputs together, but avoids
    fitting one huge multi-output tree over the full flat EC label space.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.1,
        max_bin=64,
        block_size=256,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        tree_method="hist",
        multi_strategy="multi_output_tree",
        early_stopping_rounds=None,
    ):
        if xgb is None:
            raise ImportError("xgboost is required to use XGBFlatECBlockClassifier.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.block_size = int(block_size)
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.multi_strategy = multi_strategy
        self.early_stopping_rounds = early_stopping_rounds
        self.n_outputs_: int | None = None
        self.block_slices_: list[tuple[int, int]] = []
        self.block_boosters_: list[object | None] = []
        self.block_trainable_indices_: list[np.ndarray] = []
        self.block_constant_values_: list[np.ndarray] = []
        self.block_devices_: list[str | None] = []
        self.block_known_rows_: list[int] = []

    @staticmethod
    def _normalize_targets(y, target_mask=None) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(f"XGBFlatECBlockClassifier expects 2D targets, got shape {y_np.shape}")

        inferred_mask = ~np.isnan(y_np)
        if target_mask is None:
            mask_np = inferred_mask
        else:
            mask_np = np.asarray(target_mask, dtype=bool)
            if mask_np.shape != y_np.shape:
                raise ValueError(
                    "target_mask shape must match y shape for XGBFlatECBlockClassifier. "
                    f"Received mask={mask_np.shape}, y={y_np.shape}."
                )
            mask_np = mask_np & inferred_mask
        return y_np, mask_np

    def _make_train_params(self, *, device_override: str | None = None) -> dict:
        return {
            "objective": "binary:logistic",
            "tree_method": self.tree_method,
            "device": device_override or self.device,
            "multi_strategy": self.multi_strategy,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "nthread": self.n_jobs,
            "seed": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }

    def fit(
        self,
        X,
        y,
        target_mask=None,
        X_val=None,
        y_val=None,
        target_mask_val=None,
        progress_label: str | None = None,
        checkpoint_path=None,
    ):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"XGBFlatECBlockClassifier expects pooled 2D inputs, got shape {X_np.shape}")

        y_np, target_mask_np = self._normalize_targets(y, target_mask)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows for XGBFlatECBlockClassifier. "
                f"Received X={X_np.shape[0]}, y={y_np.shape[0]}."
            )

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            y_val_np, target_mask_val_np = self._normalize_targets(y_val, target_mask_val)
            if X_val_np.ndim != 2:
                raise ValueError(f"X_val must be 2D, got shape {X_val_np.shape}")
            if X_val_np.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows for XGBFlatECBlockClassifier. "
                    f"Received X_val={X_val_np.shape[0]}, y_val={y_val_np.shape[0]}."
                )
            val_payload = (X_val_np, y_val_np, target_mask_val_np)

        resumed = _maybe_resume_xgb_wrapper(self, checkpoint_path)
        checkpoint_outputs = self.n_outputs_
        self.n_outputs_ = y_np.shape[1]
        if resumed:
            if checkpoint_outputs is not None and int(checkpoint_outputs) != y_np.shape[1]:
                raise ValueError("Checkpoint output dimension does not match current EC target width.")
        else:
            self.block_slices_ = []
            self.block_boosters_ = []
            self.block_trainable_indices_ = []
            self.block_constant_values_ = []
            self.block_devices_ = []
            self.block_known_rows_ = []

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}.")

        total_blocks = (self.n_outputs_ + self.block_size - 1) // self.block_size
        start_column = self.block_slices_[-1][1] if self.block_slices_ else 0
        completed_blocks = len(self.block_slices_)
        if progress_label is not None:
            known_labels = int(target_mask_np.sum())
            total_labels = int(target_mask_np.size)
            _progress_log(
                "Preparing block-wise flat EC XGBoost fits "
                f"| train_shape={X_np.shape} "
                f"| outputs={self.n_outputs_} "
                f"| block_size={self.block_size} "
                f"| blocks={total_blocks} "
                f"| known_labels={known_labels:,}/{total_labels:,} ({100 * known_labels / max(total_labels, 1):.1f}%)",
                label=progress_label,
            )
            if resumed and completed_blocks:
                _progress_log(
                    f"Resuming block-wise EC XGBoost from block {completed_blocks + 1}/{total_blocks}.",
                    label=progress_label,
                )

        fit_start = time.perf_counter()

        for block_index, start in enumerate(range(start_column, self.n_outputs_, self.block_size), start=completed_blocks + 1):
            stop = min(start + self.block_size, self.n_outputs_)
            block_slice = slice(start, stop)
            y_block = y_np[:, block_slice]
            block_mask = target_mask_np[:, block_slice]
            known_row_mask = np.all(block_mask, axis=1)
            constants = np.full(stop - start, np.nan, dtype=np.float32)
            trainable_local: list[int] = []

            if np.any(known_row_mask):
                y_block_dense = y_block[known_row_mask]
                for local_index in range(y_block_dense.shape[1]):
                    unique_classes = np.unique(y_block_dense[:, local_index].astype(np.int32, copy=False))
                    if unique_classes.size == 1:
                        constants[local_index] = float(unique_classes[0])
                    else:
                        trainable_local.append(local_index)
            else:
                y_block_dense = np.empty((0, stop - start), dtype=np.float32)

            trainable_local_np = np.asarray(trainable_local, dtype=np.int32)
            booster = None
            booster_device: str | None = None
            trained_rows = int(known_row_mask.sum())

            if progress_label is not None:
                _progress_log(
                    f"EC block {block_index}/{total_blocks} cols=[{start}:{stop}) "
                    f"| known_rows={trained_rows:,} "
                    f"| trainable_outputs={int(trainable_local_np.size):,} "
                    f"| constant_outputs={int(np.isfinite(constants).sum()):,}",
                    label=progress_label,
                )

            if trainable_local_np.size > 0 and trained_rows > 0:
                X_block_train = X_np[known_row_mask]
                y_block_train = y_block_dense[:, trainable_local_np].astype(np.float32, copy=False)
                dtrain = xgb.QuantileDMatrix(
                    X_block_train,
                    label=y_block_train,
                    max_bin=self.max_bin,
                )
                evals = []
                train_kwargs = {
                    "params": self._make_train_params(),
                    "dtrain": dtrain,
                    "num_boost_round": self.n_estimators,
                    "evals": evals,
                    "verbose_eval": False,
                }

                if val_payload is not None:
                    X_val_np, y_val_np, target_mask_val_np = val_payload
                    val_known_mask = np.all(target_mask_val_np[:, block_slice][:, trainable_local_np], axis=1)
                    if np.any(val_known_mask):
                        dval = xgb.QuantileDMatrix(
                            X_val_np[val_known_mask],
                            label=y_val_np[val_known_mask][:, block_slice][:, trainable_local_np].astype(
                                np.float32,
                                copy=False,
                            ),
                            max_bin=self.max_bin,
                            ref=dtrain,
                        )
                        evals.append((dval, "validation"))
                        if self.early_stopping_rounds is not None:
                            train_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

                block_start = time.perf_counter()
                booster_device = self.device
                try:
                    booster = xgb.train(**train_kwargs)
                except Exception as exc:
                    if booster_device != "cpu" and _is_xgb_cuda_oom(exc):
                        _progress_log(
                            f"EC block {block_index}/{total_blocks}: GPU OOM, retrying on CPU.",
                            label=progress_label,
                        )
                        booster_device = "cpu"
                        train_kwargs["params"] = self._make_train_params(device_override="cpu")
                        booster = xgb.train(**train_kwargs)
                    else:
                        raise

                if progress_label is not None:
                    elapsed = time.perf_counter() - block_start
                    _progress_log(
                        f"EC block {block_index}/{total_blocks}: done in {elapsed:.1f}s "
                        f"| device={booster_device}",
                        label=progress_label,
                    )

            self.block_slices_.append((start, stop))
            self.block_boosters_.append(booster)
            self.block_trainable_indices_.append(trainable_local_np)
            self.block_constant_values_.append(constants)
            self.block_devices_.append(booster_device)
            self.block_known_rows_.append(trained_rows)
            _save_xgb_checkpoint(self, checkpoint_path)

            if progress_label is not None and _should_log_progress(block_index, total_blocks):
                total_elapsed = time.perf_counter() - fit_start
                eta_seconds = (total_elapsed / block_index) * (total_blocks - block_index)
                _progress_log(
                    f"Flat EC block progress: {block_index}/{total_blocks} blocks done "
                    f"| elapsed={total_elapsed:.1f}s "
                    f"| eta={eta_seconds:.1f}s",
                    label=progress_label,
                )

        return self

    def predict(self, X) -> np.ndarray:
        if self.n_outputs_ is None:
            raise ValueError("XGBFlatECBlockClassifier must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")

        pred = np.full((X_np.shape[0], self.n_outputs_), np.nan, dtype=np.float32)

        for (start, stop), booster, trainable_local, constants, booster_device in zip(
            self.block_slices_,
            self.block_boosters_,
            self.block_trainable_indices_,
            self.block_constant_values_,
            self.block_devices_,
        ):
            block_pred = pred[:, start:stop]
            finite_constants = np.isfinite(constants)
            if np.any(finite_constants):
                block_pred[:, finite_constants] = constants[finite_constants]

            if booster is None or trainable_local.size == 0:
                continue

            if booster_device == "cpu":
                try:
                    raw_pred = booster.inplace_predict(X_np, validate_features=False)
                except Exception:
                    raw_pred = booster.predict(xgb.QuantileDMatrix(X_np))
            else:
                raw_pred = booster.predict(xgb.QuantileDMatrix(X_np))

            raw_pred = np.asarray(raw_pred, dtype=np.float32)
            if raw_pred.ndim == 1:
                raw_pred = raw_pred.reshape(X_np.shape[0], trainable_local.size)
            block_pred[:, trainable_local] = (raw_pred >= 0.5).astype(np.float32, copy=False)

        return pred


class XGBFlatECClassifier:
    """One binary XGBoost model per concatenated EC column."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        early_stopping_rounds=None,
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use XGBFlatECClassifier.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.models_: list[XGBClassifier | None] = []
        self.constant_predictions_: list[float | None] = []
        self.model_devices_: list[str | None] = []
        self.n_outputs_: int | None = None

    @staticmethod
    def _normalize_targets(y, target_mask=None) -> tuple[np.ndarray, np.ndarray]:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(f"XGBFlatECClassifier expects 2D targets, got shape {y_np.shape}")

        inferred_mask = ~np.isnan(y_np)
        if target_mask is None:
            mask_np = inferred_mask
        else:
            mask_np = np.asarray(target_mask, dtype=bool)
            if mask_np.shape != y_np.shape:
                raise ValueError(
                    "target_mask shape must match y shape for XGBFlatECClassifier. "
                    f"Received mask={mask_np.shape}, y={y_np.shape}."
                )
            mask_np = mask_np & inferred_mask
        return y_np, mask_np

    def _make_model(self, use_early_stopping: bool, *, device_override: str | None = None) -> XGBClassifier:
        params = {
            "objective": "binary:logistic",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "max_bin": self.max_bin,
            "n_jobs": self.n_jobs,
            "device": device_override or self.device,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }
        if use_early_stopping and self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        return XGBClassifier(**params)

    def fit(
        self,
        X,
        y,
        target_mask=None,
        X_val=None,
        y_val=None,
        target_mask_val=None,
        progress_label: str | None = None,
        checkpoint_path=None,
    ):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"XGBFlatECClassifier expects pooled 2D inputs, got shape {X_np.shape}")

        y_np, target_mask_np = self._normalize_targets(y, target_mask)
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows for XGBFlatECClassifier. "
                f"Received X={X_np.shape[0]}, y={y_np.shape[0]}."
            )

        val_payload = None
        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            if X_val_np.ndim != 2:
                raise ValueError(f"X_val must be 2D, got shape {X_val_np.shape}")
            y_val_np, target_mask_val_np = self._normalize_targets(y_val, target_mask_val)
            if X_val_np.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows for XGBFlatECClassifier. "
                    f"Received X_val={X_val_np.shape[0]}, y_val={y_val_np.shape[0]}."
                )
            val_payload = (X_val_np, y_val_np, target_mask_val_np)

        resumed = _maybe_resume_xgb_wrapper(self, checkpoint_path)
        checkpoint_outputs = self.n_outputs_
        self.n_outputs_ = y_np.shape[1]
        if resumed:
            if checkpoint_outputs is not None and int(checkpoint_outputs) != y_np.shape[1]:
                raise ValueError("Checkpoint output dimension does not match current EC target width.")
        else:
            self.models_ = []
            self.constant_predictions_ = []
            self.model_devices_ = []

        if progress_label is not None:
            known_labels = int(target_mask_np.sum())
            total_labels = int(target_mask_np.size)
            _progress_log(
                "Preparing flat EC XGBoost fits "
                f"| train_shape={X_np.shape} "
                f"| outputs={self.n_outputs_} "
                f"| known_labels={known_labels:,}/{total_labels:,} ({100 * known_labels / max(total_labels, 1):.1f}%)",
                label=progress_label,
            )
            if resumed and self.models_:
                _progress_log(
                    f"Resuming flat EC XGBoost from column {len(self.models_) + 1}/{self.n_outputs_}.",
                    label=progress_label,
                )

        fit_start = time.perf_counter()

        start_column = len(self.models_)
        for column_index in range(start_column, self.n_outputs_):
            known_mask = target_mask_np[:, column_index]
            if not np.any(known_mask):
                self.models_.append(None)
                self.constant_predictions_.append(np.nan)
                self.model_devices_.append(None)
                _save_xgb_checkpoint(self, checkpoint_path)
                if progress_label is not None:
                    _progress_log(
                        f"EC column {column_index + 1}/{self.n_outputs_}: skipped (no known labels).",
                        label=progress_label,
                    )
                continue

            y_column = y_np[known_mask, column_index].astype(np.int32, copy=False)
            unique_classes = np.unique(y_column)
            if unique_classes.size == 1:
                self.models_.append(None)
                self.constant_predictions_.append(float(unique_classes[0]))
                self.model_devices_.append(None)
                _save_xgb_checkpoint(self, checkpoint_path)
                if progress_label is not None:
                    _progress_log(
                        f"EC column {column_index + 1}/{self.n_outputs_}: constant target="
                        f"{int(unique_classes[0])}.",
                        label=progress_label,
                    )
                continue

            X_column = X_np[known_mask]
            use_early_stopping = False
            fit_kwargs = {"verbose": False}
            val_rows = 0
            if val_payload is not None:
                X_val_np, y_val_np, target_mask_val_np = val_payload
                val_known_mask = target_mask_val_np[:, column_index]
                if np.any(val_known_mask):
                    val_rows = int(val_known_mask.sum())
                    fit_kwargs["eval_set"] = [
                        (
                            X_val_np[val_known_mask],
                            y_val_np[val_known_mask, column_index].astype(np.int32, copy=False),
                        )
                    ]
                    fit_kwargs["verbose"] = max(1, self.n_estimators // 5)
                    use_early_stopping = True

            if progress_label is not None:
                _progress_log(
                    f"EC column {column_index + 1}/{self.n_outputs_}: "
                    f"train_rows={X_column.shape[0]:,} "
                    f"positives={int(y_column.sum()):,} "
                    f"({100 * float(y_column.mean()):.2f}%) "
                    f"val_rows={val_rows:,} "
                    f"device={self.device}",
                    label=progress_label,
                )

            model_device = self.device
            model = self._make_model(use_early_stopping=use_early_stopping, device_override=model_device)
            column_start = time.perf_counter()
            try:
                model.fit(X_column, y_column, **fit_kwargs)
            except Exception as exc:
                if model_device != "cpu" and _is_xgb_cuda_oom(exc):
                    _progress_log(
                        f"EC column {column_index + 1}/{self.n_outputs_}: GPU OOM, retrying on CPU.",
                        label=progress_label,
                    )
                    model_device = "cpu"
                    model = self._make_model(
                        use_early_stopping=use_early_stopping,
                        device_override=model_device,
                    )
                    model.fit(X_column, y_column, **fit_kwargs)
                else:
                    raise
            self.models_.append(model)
            self.constant_predictions_.append(None)
            self.model_devices_.append(model_device)
            _save_xgb_checkpoint(self, checkpoint_path)

            if progress_label is not None:
                elapsed = time.perf_counter() - column_start
                best_iteration = getattr(model, "best_iteration", None)
                best_score = getattr(model, "best_score", None)
                summary = (
                    f"EC column {column_index + 1}/{self.n_outputs_}: done in {elapsed:.1f}s "
                    f"| device={model_device}"
                )
                if best_iteration is not None:
                    summary += f" | best_iteration={best_iteration}"
                if best_score is not None:
                    summary += f" | best_score={best_score}"
                _progress_log(summary, label=progress_label)

                columns_done = column_index + 1
                if _should_log_progress(columns_done, self.n_outputs_):
                    total_elapsed = time.perf_counter() - fit_start
                    eta_seconds = (total_elapsed / columns_done) * (self.n_outputs_ - columns_done)
                    _progress_log(
                        f"Flat EC progress: {columns_done}/{self.n_outputs_} columns done "
                        f"| elapsed={total_elapsed:.1f}s "
                        f"| eta={eta_seconds:.1f}s",
                        label=progress_label,
                    )

        return self

    def predict(self, X) -> np.ndarray:
        if self.n_outputs_ is None:
            raise ValueError("XGBFlatECClassifier must be fit before calling predict().")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be pooled 2D, got shape {X_np.shape}")

        pred = np.zeros((X_np.shape[0], self.n_outputs_), dtype=np.float32)
        for column_index, (model, constant_prediction, model_device) in enumerate(
            zip(self.models_, self.constant_predictions_, self.model_devices_)
        ):
            if constant_prediction is not None:
                if np.isnan(constant_prediction):
                    pred[:, column_index] = np.nan
                else:
                    pred[:, column_index] = np.float32(constant_prediction)
                continue

            if model_device == "cpu":
                booster = model.get_booster()
                try:
                    raw_pred = booster.inplace_predict(X_np, validate_features=False)
                    raw_pred = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
                    pred[:, column_index] = (raw_pred >= 0.5).astype(np.float32, copy=False)
                    continue
                except Exception:
                    pass
            pred[:, column_index] = model.predict(X_np).astype(np.float32, copy=False)
        return pred


class _ResidueDataIter:
    """Streaming DataIter over a residue dataloader for XGBoost QuantileDMatrix.

    Decodes BF16-raw embeddings to float32 one batch at a time so the full residue
    matrix never lives in RAM. Tracks unique labels seen and totals for logging.
    """

    def __init__(self, loader_factory, decode_fn):
        if xgb is None:
            raise ImportError("xgboost is required to build a residue DataIter.")
        # xgb.DataIter is the upstream base class; subclass dynamically so this file
        # still imports when xgboost is missing.
        self._iter_impl = _build_residue_data_iter_impl(loader_factory, decode_fn)

    def as_xgb_iter(self):
        return self._iter_impl


def _build_residue_data_iter_impl(loader_factory, decode_fn):
    class _Impl(xgb.DataIter):
        def __init__(self) -> None:
            super().__init__()
            self._loader_factory = loader_factory
            self._decode_fn = decode_fn
            self._iterator = None
            self.labels_seen: set[int] = set()
            self.residues_seen = 0
            self.proteins_seen = 0

        def next(self, input_data):
            if self._iterator is None:
                self._iterator = iter(self._loader_factory())
            while True:
                try:
                    batch = next(self._iterator)
                except StopIteration:
                    return 0
                X, y, lengths = self._decode_fn(batch)
                if X.size == 0:
                    continue
                input_data(data=X, label=y)
                self.residues_seen += int(X.shape[0])
                self.proteins_seen += int(lengths.shape[0])
                for value in np.unique(y).tolist():
                    self.labels_seen.add(int(value))
                return 1

        def reset(self) -> None:
            self._iterator = None

    return _Impl()


class XGBResidueBinaryClassifier:
    """XGBoost binary classifier over residue embeddings."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        device=None,
        random_state=0,
        eval_metric="logloss",
        early_stopping_rounds=None,
    ):
        if xgb is None:
            raise ImportError("xgboost is required to use XGBResidueBinaryClassifier.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.constant_prediction_ = None
        self.model_device_ = self.device
        self.best_iteration_ = None
        self.best_score_ = None

    @staticmethod
    def _flatten_batch(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = _as_xgb_residue_features(batch.protein_emb)
        y = np.asarray(batch.target_per_residue, dtype=np.int8)

        if batch.lengths is not None:
            lengths = np.asarray(batch.lengths, dtype=np.int32)
            if X.ndim != 2:
                raise ValueError(
                    "Concatenated residue batches must expose a 2D protein_emb array. "
                    f"Received shape {X.shape}."
                )
            if y.ndim != 1:
                raise ValueError(
                    "Concatenated residue batches must expose a 1D target_per_residue array. "
                    f"Received shape {y.shape}."
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "Concatenated residue batch protein_emb and target_per_residue length mismatch: "
                    f"X={X.shape[0]}, y={y.shape[0]}."
                )
            if lengths.sum() != X.shape[0]:
                raise ValueError(
                    "Concatenated residue batch lengths must sum to the flattened residue count. "
                    f"sum(lengths)={lengths.sum()}, X={X.shape[0]}."
                )
            return X.astype(np.float32, copy=False), y.astype(np.int8, copy=False), lengths

        mask = np.asarray(batch.target_mask, dtype=bool)
        return (
            X[mask].astype(np.float32, copy=False),
            y[mask].astype(np.int8, copy=False),
            mask.sum(axis=1).astype(np.int32, copy=False),
        )

    @classmethod
    def _flatten_loader(
        cls,
        dataloader,
        *,
        progress_label: str | None = None,
        split_name: str = "train",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        lengths: list[int] = []
        buffered_bytes = 0
        residues_seen = 0
        proteins_seen = 0

        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None

        flatten_start = time.perf_counter()

        for batch_index, batch in enumerate(dataloader, start=1):
            X_flat, y_flat, batch_lengths = cls._flatten_batch(batch)
            if X_flat.size == 0:
                continue
            X_chunks.append(X_flat)
            y_chunks.append(y_flat)
            lengths.extend(batch_lengths.tolist())
            buffered_bytes += X_flat.nbytes + y_flat.nbytes
            residues_seen += int(X_flat.shape[0])
            proteins_seen += int(batch_lengths.shape[0])

            if progress_label is not None and _should_log_progress(batch_index, total_batches):
                percent = ""
                if total_batches is not None and total_batches > 0:
                    percent = f" ({100 * batch_index / total_batches:.0f}%)"
                _progress_log(
                    f"Flatten {split_name} residue batches: {batch_index}"
                    + (f"/{total_batches}" if total_batches is not None else "")
                    + percent
                    + f" | proteins={proteins_seen:,}"
                    + f" | residues={residues_seen:,}"
                    + f" | buffered≈{_estimate_gib(buffered_bytes):.1f} GiB"
                    + f" | elapsed={time.perf_counter() - flatten_start:.1f}s",
                    label=progress_label,
                )

        if not X_chunks:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int8),
                np.empty((0,), dtype=np.int32),
            )

        X_all = np.concatenate(X_chunks, axis=0)
        y_all = np.concatenate(y_chunks, axis=0)
        lengths_all = np.asarray(lengths, dtype=np.int32)
        if progress_label is not None:
            _progress_log(
                f"Flattened {split_name} residues "
                f"| shape={X_all.shape} "
                f"| positives={int(y_all.sum()):,}"
                f" ({100 * float(y_all.mean()):.2f}%) "
                f"| approx_ram={_estimate_gib(X_all.nbytes + y_all.nbytes):.1f} GiB "
                f"| elapsed={time.perf_counter() - flatten_start:.1f}s",
                label=progress_label,
            )

        return (
            X_all,
            y_all,
            lengths_all,
        )

    @staticmethod
    def _split_flat_predictions(flat_pred: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        start = 0
        for length in lengths.tolist():
            stop = start + int(length)
            rows.append(np.asarray(flat_pred[start:stop], dtype=np.float32))
            start = stop
        return rows

    def _make_train_params(self, *, device_override: str | None = None) -> dict:
        return {
            "objective": "binary:logistic",
            "tree_method": self.tree_method,
            "device": device_override or self.device,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "nthread": self.n_jobs,
            "seed": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }

    def _build_quantile_dmatrix(
        self,
        dataloader,
        *,
        ref=None,
        progress_label: str | None = None,
        split_name: str = "train",
    ):
        """Stream the dataloader through xgb.DataIter to build a QuantileDMatrix in-place."""
        data_iter = _ResidueDataIter(
            loader_factory=lambda: dataloader,
            decode_fn=self._flatten_batch,
        )
        impl = data_iter.as_xgb_iter()
        build_start = time.perf_counter()
        dmatrix = xgb.QuantileDMatrix(impl, max_bin=self.max_bin, ref=ref, nthread=self.n_jobs)
        if progress_label is not None:
            _progress_log(
                f"Built {split_name} QuantileDMatrix "
                f"| proteins={impl.proteins_seen:,} "
                f"| residues={impl.residues_seen:,} "
                f"| classes_seen={sorted(impl.labels_seen)} "
                f"| max_bin={self.max_bin} "
                f"| elapsed={time.perf_counter() - build_start:.1f}s",
                label=progress_label,
            )
        return dmatrix, impl

    def fit_loader(
        self,
        dataloader,
        batch_kind: str,
        val_dataloader=None,
        progress_label: str | None = None,
        checkpoint_path=None,
    ):
        if batch_kind != "protein_residue":
            raise ValueError(f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'")
        resumed = _maybe_resume_xgb_wrapper(self, checkpoint_path)
        if resumed and (self.booster is not None or self.constant_prediction_ is not None):
            if progress_label is not None:
                _progress_log("Resuming from existing residue XGBoost checkpoint.", label=progress_label)
            return self

        self.constant_prediction_ = None
        self.booster = None
        self.best_iteration_ = None
        self.best_score_ = None
        self.model_device_ = self.device

        if progress_label is not None:
            _progress_log(
                f"Residue XGBoost streaming fit starting | device={self.model_device_}",
                label=progress_label,
            )

        try:
            dtrain, train_impl = self._build_quantile_dmatrix(
                dataloader,
                progress_label=progress_label,
                split_name="train",
            )
        except Exception as exc:
            if self.model_device_ != "cpu" and _is_xgb_cuda_failure(exc):
                _progress_log(
                    "GPU failure building residue QuantileDMatrix "
                    f"({type(exc).__name__}: {exc}); retrying on CPU.",
                    label=progress_label,
                )
                self.model_device_ = "cpu"
                dtrain, train_impl = self._build_quantile_dmatrix(
                    dataloader,
                    progress_label=progress_label,
                    split_name="train",
                )
            else:
                raise
        if train_impl.residues_seen == 0:
            raise ValueError("No residue-level training samples found for XGBoost.")

        if len(train_impl.labels_seen) == 1:
            only_class = next(iter(train_impl.labels_seen))
            self.constant_prediction_ = float(only_class)
            self.model_device_ = None
            if progress_label is not None:
                _progress_log(
                    f"Residue XGBoost collapsed to constant prediction={only_class}.",
                    label=progress_label,
                )
            _save_xgb_checkpoint(self, checkpoint_path)
            return self

        evals: list[tuple] = []
        if val_dataloader is not None:
            dval, val_impl = self._build_quantile_dmatrix(
                val_dataloader,
                ref=dtrain,
                progress_label=progress_label,
                split_name="val",
            )
            if val_impl.residues_seen > 0:
                evals.append((dval, "validation"))

        train_kwargs = {
            "params": self._make_train_params(device_override=self.model_device_),
            "dtrain": dtrain,
            "num_boost_round": self.n_estimators,
            "evals": evals,
            "verbose_eval": False,
        }
        if evals and self.early_stopping_rounds is not None:
            train_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        fit_start = time.perf_counter()
        try:
            self.booster = xgb.train(**train_kwargs)
        except Exception as exc:
            if self.model_device_ != "cpu" and _is_xgb_cuda_failure(exc):
                _progress_log(
                    f"GPU failure during residue XGBoost fit ({type(exc).__name__}: {exc}); "
                    "retrying on CPU.",
                    label=progress_label,
                )
                self.model_device_ = "cpu"
                train_kwargs["params"] = self._make_train_params(device_override="cpu")
                self.booster = xgb.train(**train_kwargs)
            else:
                raise

        self.best_iteration_ = getattr(self.booster, "best_iteration", None)
        self.best_score_ = getattr(self.booster, "best_score", None)

        if progress_label is not None:
            summary = f"Residue XGBoost fit done in {time.perf_counter() - fit_start:.1f}s"
            if self.best_iteration_ is not None:
                summary += f" | best_iteration={self.best_iteration_}"
            if self.best_score_ is not None:
                summary += f" | best_score={self.best_score_}"
            summary += f" | device={self.model_device_}"
            _progress_log(summary, label=progress_label)
        _save_xgb_checkpoint(self, checkpoint_path)
        return self

    def predict_loader(self, dataloader, batch_kind: str) -> np.ndarray:
        if batch_kind != "protein_residue":
            raise ValueError(f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'")

        if self.booster is None and self.constant_prediction_ is None:
            raise ValueError("XGBResidueBinaryClassifier must be fit before calling predict_loader().")

        pred_rows: list[np.ndarray] = []
        for batch in dataloader:
            X_flat, _, lengths = self._flatten_batch(batch)
            if self.constant_prediction_ is not None:
                flat_pred = np.full((X_flat.shape[0],), self.constant_prediction_, dtype=np.float32)
            else:
                raw_pred = self.booster.inplace_predict(X_flat, validate_features=False)
                flat_pred = (np.asarray(raw_pred, dtype=np.float32).reshape(-1) >= 0.5).astype(
                    np.float32,
                    copy=False,
                )
            pred_rows.extend(self._split_flat_predictions(flat_pred, lengths))

        return pad_1d_array_list(pred_rows, dtype=np.float32)
