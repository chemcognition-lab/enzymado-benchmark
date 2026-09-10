"""Block-wise multi-output XGBoost classifier for flat EC labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import typing

import numpy as np
import numpy.typing as npt
import torch

import enzymado.types

from . import xgb_support

type FloatArray = npt.NDArray[np.float32]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int32]
type ValidationPayload = tuple[FloatArray, FloatArray, BoolArray]


class _Booster(typing.Protocol):
    """Subset of the XGBoost booster API used for EC block prediction."""

    def inplace_predict(
        self,
        features: FloatArray,
        *,
        validate_features: bool,
    ) -> FloatArray:
        """Predict directly from NumPy features."""

        ...

    def predict(self, matrix: object) -> FloatArray:
        """Predict from an XGBoost matrix."""

        ...


@dataclass(frozen=True)
class _FitData:
    """Validated arrays used by one block-wise fit."""

    features: FloatArray
    targets: FloatArray
    target_mask: BoolArray
    validation: ValidationPayload | None


@dataclass(frozen=True)
class _BlockData:
    """Dense, fully known target subset for one EC output block."""

    column_slice: slice
    known_rows: BoolArray
    dense_targets: FloatArray
    trainable_indices: IntArray
    constants: FloatArray

    @property
    def trained_rows(self) -> int:
        """Return the number of rows whose labels are all known in this block."""
        return int(self.known_rows.sum())


@dataclass(frozen=True)
class _BlockFitResult:
    """Booster state persisted for one completed EC output block."""

    booster: _Booster | None
    device: str | None


class XGBFlatECBlockClassifier:
    """Multi-output XGBoost over contiguous blocks of flat EC columns.

    This keeps Ben's idea of training multiple EC outputs together, but avoids
    fitting one huge multi-output tree over the full flat EC label space.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.1,
        max_bin: int = 64,
        block_size: int = 256,
        n_jobs: int = -1,
        device: str | None = None,
        random_state: int = 0,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        multi_strategy: str = "multi_output_tree",
        early_stopping_rounds: int | None = None,
    ) -> None:
        if xgb_support.xgb is None:
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
        self.block_boosters_: list[_Booster | None] = []
        self.block_trainable_indices_: list[IntArray] = []
        self.block_constant_values_: list[FloatArray] = []
        self.block_devices_: list[str | None] = []
        self.block_known_rows_: list[int] = []

    @staticmethod
    def _normalize_targets(
        y: enzymado.types.Array,
        target_mask: enzymado.types.Array | None = None,
    ) -> tuple[FloatArray, BoolArray]:
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2:
            raise ValueError(
                f"XGBFlatECBlockClassifier expects 2D targets, got shape {y_np.shape}"
            )

        inferred_mask = ~np.isnan(y_np)
        if target_mask is None:
            mask_np = inferred_mask
        else:
            mask_np = np.asarray(target_mask, dtype=bool)
            if mask_np.shape != y_np.shape:
                raise ValueError(
                    "target_mask shape must match y shape for "
                    "XGBFlatECBlockClassifier. "
                    f"Received mask={mask_np.shape}, y={y_np.shape}."
                )
            mask_np = mask_np & inferred_mask
        return y_np, mask_np

    def _make_train_params(
        self,
        *,
        device_override: str | None = None,
    ) -> dict[str, object]:
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

    def _prepare_fit_data(
        self,
        x: enzymado.types.Array,
        y: enzymado.types.Array,
        target_mask: enzymado.types.Array | None,
        x_val: enzymado.types.Array | None,
        y_val: enzymado.types.Array | None,
        target_mask_val: enzymado.types.Array | None,
    ) -> _FitData:
        """Normalize and validate training and optional validation arrays."""
        features = np.asarray(x, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                "XGBFlatECBlockClassifier expects pooled 2D inputs, "
                f"got shape {features.shape}"
            )

        targets, mask = self._normalize_targets(y, target_mask)
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                "x and y must contain the same number of rows for "
                "XGBFlatECBlockClassifier. "
                f"Received x={features.shape[0]}, y={targets.shape[0]}."
            )

        validation = None
        if x_val is not None and y_val is not None:
            val_features = np.asarray(x_val, dtype=np.float32)
            val_targets, val_mask = self._normalize_targets(
                y_val,
                target_mask_val,
            )
            if val_features.ndim != 2:
                raise ValueError(
                    f"x_val must be 2D, got shape {val_features.shape}"
                )
            if val_features.shape[0] != val_targets.shape[0]:
                raise ValueError(
                    "x_val and y_val must contain the same number of rows for "
                    "XGBFlatECBlockClassifier. "
                    f"Received x_val={val_features.shape[0]}, "
                    f"y_val={val_targets.shape[0]}."
                )
            validation = val_features, val_targets, val_mask
        return _FitData(features, targets, mask, validation)

    def _restore_fit_state(
        self,
        target_width: int,
        checkpoint_path: str | Path | None,
    ) -> tuple[bool, int, int, int]:
        """Restore checkpoint state and return block-loop coordinates."""
        resumed = xgb_support._maybe_resume_xgb_wrapper(self, checkpoint_path)
        checkpoint_outputs = self.n_outputs_
        self.n_outputs_ = target_width
        if (
            resumed
            and checkpoint_outputs is not None
            and int(checkpoint_outputs) != target_width
        ):
            raise ValueError(
                "Checkpoint output dimension does not match current EC target width."
            )
        if not resumed:
            self.block_slices_ = []
            self.block_boosters_ = []
            self.block_trainable_indices_ = []
            self.block_constant_values_ = []
            self.block_devices_ = []
            self.block_known_rows_ = []

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}.")
        total_blocks = (target_width + self.block_size - 1) // self.block_size
        start_column = self.block_slices_[-1][1] if self.block_slices_ else 0
        completed_blocks = len(self.block_slices_)
        return resumed, total_blocks, start_column, completed_blocks

    def _log_fit_setup(
        self,
        data: _FitData,
        *,
        resumed: bool,
        total_blocks: int,
        completed_blocks: int,
        progress_label: str | None,
    ) -> None:
        """Log target coverage and checkpoint position for an EC block fit."""
        if progress_label is None:
            return
        known_labels = int(data.target_mask.sum())
        total_labels = int(data.target_mask.size)
        xgb_support._progress_log(
            "Preparing block-wise flat EC XGBoost fits "
            f"| train_shape={data.features.shape} "
            f"| outputs={data.targets.shape[1]} "
            f"| block_size={self.block_size} "
            f"| blocks={total_blocks} "
            f"| known_labels={known_labels:,}/{total_labels:,} "
            f"({100 * known_labels / max(total_labels, 1):.1f}%)",
            label=progress_label,
        )
        if resumed and completed_blocks:
            xgb_support._progress_log(
                "Resuming block-wise EC XGBoost from block "
                f"{completed_blocks + 1}/{total_blocks}.",
                label=progress_label,
            )

    @staticmethod
    def _prepare_block_data(
        targets: FloatArray,
        target_mask: BoolArray,
        start: int,
        stop: int,
    ) -> _BlockData:
        """Separate constant and trainable outputs for one EC column block."""
        column_slice = slice(start, stop)
        target_block = targets[:, column_slice]
        known_rows = np.all(target_mask[:, column_slice], axis=1)
        constants = np.full(stop - start, np.nan, dtype=np.float32)
        trainable: list[int] = []

        if np.any(known_rows):
            dense_targets = target_block[known_rows]
            for local_index in range(dense_targets.shape[1]):
                unique_classes = np.unique(
                    dense_targets[:, local_index].astype(np.int32, copy=False)
                )
                if unique_classes.size == 1:
                    constants[local_index] = float(unique_classes[0])
                else:
                    trainable.append(local_index)
        else:
            dense_targets = np.empty((0, stop - start), dtype=np.float32)

        return _BlockData(
            column_slice=column_slice,
            known_rows=known_rows,
            dense_targets=dense_targets,
            trainable_indices=np.asarray(trainable, dtype=np.int32),
            constants=constants,
        )

    @staticmethod
    def _log_block_setup(
        block: _BlockData,
        block_index: int,
        total_blocks: int,
        start: int,
        stop: int,
        progress_label: str | None,
    ) -> None:
        """Log label coverage for one EC output block."""
        if progress_label is None:
            return
        xgb_support._progress_log(
            f"EC block {block_index}/{total_blocks} cols=[{start}:{stop}) "
            f"| known_rows={block.trained_rows:,} "
            f"| trainable_outputs={int(block.trainable_indices.size):,} "
            f"| constant_outputs={int(np.isfinite(block.constants).sum()):,}",
            label=progress_label,
        )

    def _build_validation_matrix(
        self,
        validation: ValidationPayload | None,
        block: _BlockData,
        dtrain: object,
    ) -> object | None:
        """Build a validation matrix for rows known across trainable outputs."""
        if validation is None:
            return None
        features, targets, target_mask = validation
        trainable = block.trainable_indices
        block_mask = target_mask[:, block.column_slice][:, trainable]
        known_rows = np.all(block_mask, axis=1)
        if not np.any(known_rows):
            return None
        labels = targets[:, block.column_slice][:, trainable][known_rows]
        return xgb_support.xgb.QuantileDMatrix(
            features[known_rows],
            label=labels.astype(np.float32, copy=False),
            max_bin=self.max_bin,
            ref=dtrain,
        )

    def _train_block(
        self,
        data: _FitData,
        block: _BlockData,
        block_index: int,
        total_blocks: int,
        progress_label: str | None,
    ) -> _BlockFitResult:
        """Train one non-constant EC output block with a CPU OOM fallback."""
        if block.trainable_indices.size == 0 or block.trained_rows == 0:
            return _BlockFitResult(booster=None, device=None)

        labels = block.dense_targets[:, block.trainable_indices].astype(
            np.float32,
            copy=False,
        )
        dtrain = xgb_support.xgb.QuantileDMatrix(
            data.features[block.known_rows],
            label=labels,
            max_bin=self.max_bin,
        )
        evals: list[tuple[object, str]] = []
        train_kwargs: dict[str, object] = {
            "params": self._make_train_params(),
            "dtrain": dtrain,
            "num_boost_round": self.n_estimators,
            "evals": evals,
            "verbose_eval": False,
        }
        dval = self._build_validation_matrix(data.validation, block, dtrain)
        if dval is not None:
            evals.append((dval, "validation"))
            if self.early_stopping_rounds is not None:
                train_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        block_start = time.perf_counter()
        booster_device = self.device
        try:
            booster = typing.cast(_Booster, xgb_support.xgb.train(**train_kwargs))
        except Exception as exc:  # noqa: BLE001 - preserve retry contract.
            if booster_device == "cpu" or not xgb_support._is_xgb_cuda_oom(exc):
                raise
            xgb_support._progress_log(
                f"EC block {block_index}/{total_blocks}: GPU OOM, retrying on CPU.",
                label=progress_label,
            )
            booster_device = "cpu"
            train_kwargs["params"] = self._make_train_params(
                device_override="cpu"
            )
            booster = typing.cast(
                _Booster,
                xgb_support.xgb.train(**train_kwargs),
            )

        if progress_label is not None:
            elapsed = time.perf_counter() - block_start
            xgb_support._progress_log(
                f"EC block {block_index}/{total_blocks}: done in {elapsed:.1f}s "
                f"| device={booster_device}",
                label=progress_label,
            )
        return _BlockFitResult(booster=booster, device=booster_device)

    def _store_block_result(
        self,
        start: int,
        stop: int,
        block: _BlockData,
        result: _BlockFitResult,
    ) -> None:
        """Append one block result to the checkpoint-compatible state lists."""
        self.block_slices_.append((start, stop))
        self.block_boosters_.append(result.booster)
        self.block_trainable_indices_.append(block.trainable_indices)
        self.block_constant_values_.append(block.constants)
        self.block_devices_.append(result.device)
        self.block_known_rows_.append(block.trained_rows)

    @staticmethod
    def _log_overall_progress(
        block_index: int,
        total_blocks: int,
        fit_start: float,
        progress_label: str | None,
    ) -> None:
        """Log elapsed time and ETA at the configured progress cadence."""
        if progress_label is None or not xgb_support._should_log_progress(
            block_index,
            total_blocks,
        ):
            return
        total_elapsed = time.perf_counter() - fit_start
        eta_seconds = (total_elapsed / block_index) * (total_blocks - block_index)
        xgb_support._progress_log(
            f"Flat EC block progress: {block_index}/{total_blocks} blocks done "
            f"| elapsed={total_elapsed:.1f}s | eta={eta_seconds:.1f}s",
            label=progress_label,
        )

    def fit(
        self,
        x: enzymado.types.Array,
        y: enzymado.types.Array,
        target_mask: enzymado.types.Array | None = None,
        x_val: enzymado.types.Array | None = None,
        y_val: enzymado.types.Array | None = None,
        target_mask_val: enzymado.types.Array | None = None,
        progress_label: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> typing.Self:
        data = self._prepare_fit_data(
            x,
            y,
            target_mask,
            x_val,
            y_val,
            target_mask_val,
        )
        resumed, total_blocks, start_column, completed_blocks = (
            self._restore_fit_state(data.targets.shape[1], checkpoint_path)
        )
        self._log_fit_setup(
            data,
            resumed=resumed,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            progress_label=progress_label,
        )

        fit_start = time.perf_counter()

        for block_index, start in enumerate(
            range(start_column, data.targets.shape[1], self.block_size),
            start=completed_blocks + 1,
        ):
            stop = min(start + self.block_size, data.targets.shape[1])
            block = self._prepare_block_data(
                data.targets,
                data.target_mask,
                start,
                stop,
            )
            self._log_block_setup(
                block,
                block_index,
                total_blocks,
                start,
                stop,
                progress_label,
            )
            result = self._train_block(
                data,
                block,
                block_index,
                total_blocks,
                progress_label,
            )
            self._store_block_result(start, stop, block, result)
            xgb_support._save_xgb_checkpoint(self, checkpoint_path)
            self._log_overall_progress(
                block_index,
                total_blocks,
                fit_start,
                progress_label,
            )

        return self

    def _predict(
        self,
        x: enzymado.types.Array,
        *,
        proba: bool,
    ) -> FloatArray:
        if self.n_outputs_ is None:
            raise ValueError(
                "XGBFlatECBlockClassifier must be fit before calling predict()."
            )

        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"x must be pooled 2D, got shape {x_np.shape}")

        pred = np.full((x_np.shape[0], self.n_outputs_), np.nan, dtype=np.float32)

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
                    raw_pred = booster.inplace_predict(x_np, validate_features=False)
                except Exception:  # noqa: BLE001 - preserve prediction fallback.
                    raw_pred = booster.predict(xgb_support.xgb.QuantileDMatrix(x_np))
            else:
                raw_pred = booster.predict(xgb_support.xgb.QuantileDMatrix(x_np))

            raw_pred = np.asarray(raw_pred, dtype=np.float32)
            if raw_pred.ndim == 1:
                raw_pred = raw_pred.reshape(x_np.shape[0], trainable_local.size)
            block_pred[:, trainable_local] = (
                raw_pred if proba else (raw_pred >= 0.5).astype(np.float32, copy=False)
            )

        return pred

    def predict(self, x: enzymado.types.Array) -> FloatArray:
        return self._predict(x, proba=False)

    def predict_proba(self, x: enzymado.types.Array) -> FloatArray:
        """Return per-column booster probabilities instead of hard labels."""
        return self._predict(x, proba=True)
