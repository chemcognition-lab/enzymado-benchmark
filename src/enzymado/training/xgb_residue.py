"""Residue-level binary XGBoost training and external-memory support."""

from __future__ import annotations

import collections.abc
import os
from pathlib import Path
import shutil
import tempfile
import time
import typing

import numpy as np
import numpy.typing as npt
import torch

import enzymado.data_loading as data_loading
import enzymado.data_loading.stores
import enzymado.models.common

from . import xgb_support

type BatchLoader = collections.abc.Iterable[data_loading.Batch]
type DecodedBatch = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int8],
    npt.NDArray[np.int32],
]
type LoaderFactory = collections.abc.Callable[[], BatchLoader]
type DecodeBatch = collections.abc.Callable[[data_loading.Batch], DecodedBatch]


class _ResidueIterState(typing.Protocol):
    """Statistics exposed by the dynamic XGBoost data iterator."""

    labels_seen: set[int]
    residues_seen: int
    proteins_seen: int


class _Booster(typing.Protocol):
    """Subset of the XGBoost booster API used by this wrapper."""

    best_iteration: int
    best_score: float

    def inplace_predict(
        self,
        features: npt.NDArray[np.float32],
        *,
        validate_features: bool,
    ) -> npt.NDArray[np.float32]:
        """Predict from a NumPy feature matrix."""

        ...

    def num_boosted_rounds(self) -> int:
        """Return the number of completed boosting rounds."""

        ...


class _ResidueDataIter:
    """Streaming DataIter over a residue dataloader for XGBoost QuantileDMatrix.

    Decodes BF16-raw embeddings to float32 one batch at a time so the full residue
    matrix never lives in RAM. Tracks unique labels seen and totals for logging.
    """

    def __init__(
        self,
        loader_factory: LoaderFactory,
        decode_fn: DecodeBatch,
        cache_prefix: str | None = None,
    ) -> None:
        if xgb_support.xgb is None:
            raise ImportError("xgboost is required to build a residue DataIter.")
        # xgb.DataIter is the upstream base class; subclass dynamically so this file
        # still imports when xgboost is missing. A non-None cache_prefix routes the
        # binned pages to that on-disk cache (external memory) instead of RAM.
        self._iter_impl = _build_residue_data_iter_impl(
            loader_factory, decode_fn, cache_prefix
        )

    def as_xgb_iter(self) -> _ResidueIterState:
        return self._iter_impl


def _build_residue_data_iter_impl(
    loader_factory: LoaderFactory,
    decode_fn: DecodeBatch,
    cache_prefix: str | None = None,
) -> _ResidueIterState:
    class _Impl(xgb_support.xgb.DataIter):
        def __init__(self) -> None:
            super().__init__(cache_prefix=cache_prefix)
            self._loader_factory = loader_factory
            self._decode_fn = decode_fn
            self._iterator = None
            self.labels_seen: set[int] = set()
            self.residues_seen = 0
            self.proteins_seen = 0

        def next(
            self,
            input_data: collections.abc.Callable[..., object],
        ) -> int:
            if self._iterator is None:
                self._iterator = iter(self._loader_factory())
            while True:
                try:
                    batch = next(self._iterator)
                except StopIteration:
                    return 0
                x, y, lengths = self._decode_fn(batch)
                if x.size == 0:
                    continue
                input_data(data=x, label=y)
                self.residues_seen += int(x.shape[0])
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
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        tree_method: str = "hist",
        max_bin: int = 256,
        n_jobs: int = -1,
        device: str | None = None,
        random_state: int = 0,
        eval_metric: str = "logloss",
        early_stopping_rounds: int | None = None,
        external_memory: str = "auto",
        cache_dir: str | None = None,
        checkpoint_interval: int = 1,
    ) -> None:
        if xgb_support.xgb is None:
            raise ImportError("xgboost is required to use XGBResidueBinaryClassifier.")
        if external_memory not in {"auto", "on", "off"}:
            raise ValueError(
                "external_memory must be one of 'auto', 'on', or 'off'."
                f" Received '{external_memory}'."
            )

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
        # "auto" spills the binned matrix to an on-disk cache (external memory)
        # whenever the embedding store streams from disk (i.e. it was too large
        # to materialize in RAM), so a QuantileDMatrix that would not fit host
        # RAM no longer OOMs. See _should_use_external_memory.
        self.external_memory = external_memory
        self.cache_dir = cache_dir
        self.checkpoint_interval = checkpoint_interval
        self.booster: _Booster | None = None
        self.constant_prediction_: float | None = None
        self.model_device_: str | None = self.device
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.rounds_completed_ = 0
        # `training_complete_` is deliberately NOT initialized here: its absence is
        # how fit_loader recognizes a checkpoint written before mid-training
        # checkpointing existed (those were only ever saved after a finished fit,
        # so they must resume as "done" rather than be boosted a second time).
        self._cache_root_: str | None = None

    @staticmethod
    def _flatten_batch(batch: data_loading.Batch) -> DecodedBatch:
        x = xgb_support._as_xgb_residue_features(batch.protein_emb)
        y = np.asarray(batch.target_per_residue, dtype=np.int8)

        if batch.lengths is not None:
            lengths = np.asarray(batch.lengths, dtype=np.int32)
            if x.ndim != 2:
                raise ValueError(
                    "Concatenated residue batches must expose a 2D protein_emb array. "
                    f"Received shape {x.shape}."
                )
            if y.ndim != 1:
                raise ValueError(
                    "Concatenated residue batches must expose a 1D "
                    "target_per_residue array. "
                    f"Received shape {y.shape}."
                )
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "Concatenated residue batch protein_emb and "
                    "target_per_residue length mismatch: "
                    f"x={x.shape[0]}, y={y.shape[0]}."
                )
            if lengths.sum() != x.shape[0]:
                raise ValueError(
                    "Concatenated residue batch lengths must sum to the "
                    "flattened residue count. "
                    f"sum(lengths)={lengths.sum()}, x={x.shape[0]}."
                )
            return (
                x.astype(np.float32, copy=False),
                y.astype(np.int8, copy=False),
                lengths,
            )

        mask = np.asarray(batch.target_mask, dtype=bool)
        return (
            x[mask].astype(np.float32, copy=False),
            y[mask].astype(np.int8, copy=False),
            mask.sum(axis=1).astype(np.int32, copy=False),
        )

    @staticmethod
    def _split_flat_predictions(
        flat_pred: np.ndarray, lengths: np.ndarray
    ) -> list[np.ndarray]:
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

    @staticmethod
    def _dataloader_embedding_store(dataloader: BatchLoader) -> object | None:
        """Best-effort reach into the loader's residue embedding store, or None."""
        dataset = getattr(dataloader, "dataset", None)
        builder = getattr(dataset, "protein_builder", None)
        return getattr(builder, "embedding_store", None)

    def _should_use_external_memory(self, dataloader: BatchLoader) -> bool:
        """Decide whether the binned matrix should spill to disk instead of RAM."""
        if self.external_memory == "on":
            return True
        if self.external_memory == "off":
            return False
        # "auto": mirror the store decision. A lazy (disk-streamed) store was
        # chosen because the residue embeddings did not fit RAM, so hold the
        # binned matrix externally too rather than risk a host-RAM OOM.
        store = self._dataloader_embedding_store(dataloader)
        return isinstance(
            store,
            enzymado.data_loading.stores.LazySafetensorsResidueEmbeddingStore,
        )

    def _resolve_cache_base(self) -> str:
        """Pick a base dir for the external cache.

        An explicit ``cache_dir`` or ``XGB_RESIDUE_CACHE_DIR`` wins; otherwise the
        candidate with the most free space is chosen. This avoids the common trap
        of writing a hundreds-of-GiB binned cache to small node-local /tmp
        (SLURM_TMPDIR), which fails mid-write with ENOSPC.
        """
        explicit = self.cache_dir or os.environ.get("XGB_RESIDUE_CACHE_DIR")
        if explicit:
            os.makedirs(explicit, exist_ok=True)
            return explicit
        best_base, best_free = None, -1
        candidates = (
            os.environ.get("SCRATCH"),
            os.environ.get("SLURM_TMPDIR"),
            os.environ.get("TMPDIR"),
            tempfile.gettempdir(),
        )
        for base in candidates:
            if not base:
                continue
            try:
                os.makedirs(base, exist_ok=True)
                free = shutil.disk_usage(base).free
            except OSError:
                continue
            if free > best_free:
                best_base, best_free = base, free
        if best_base is None:
            raise RuntimeError("No writable directory for the external-memory cache.")
        return best_base

    def _cache_root(self) -> str:
        """Create (once) a private on-disk cache directory for external memory.

        ``_cache_root_`` is persisted in the training checkpoint (it is a plain
        wrapper attribute), but the directory it names is not: a resumed run may
        find it already cleaned up (e.g. after a prior timeout was tidied up
        manually) or simply gone. Re-mint a fresh one rather than handing back a
        dangling path — the DMatrix is always rebuilt from scratch on resume
        anyway, so nothing relies on the old cache's contents surviving.
        """
        if self._cache_root_ is None or not os.path.isdir(self._cache_root_):
            base = self._resolve_cache_base()
            self._cache_root_ = tempfile.mkdtemp(prefix="xgb_residue_cache_", dir=base)
        return self._cache_root_

    def _release_cache(self) -> None:
        """Remove the external-memory cache directory, if one was created."""
        if self._cache_root_ is not None:
            shutil.rmtree(self._cache_root_, ignore_errors=True)
            self._cache_root_ = None

    def _build_quantile_dmatrix(
        self,
        dataloader: BatchLoader,
        *,
        ref: object | None = None,
        progress_label: str | None = None,
        split_name: str = "train",
    ) -> tuple[object, _ResidueIterState]:
        """Stream the dataloader through xgb.DataIter into a (external or RAM) DMatrix.

        When external memory is selected, the binned pages are spilled to an
        on-disk cache via ExtMemQuantileDMatrix so a matrix larger than host RAM
        no longer OOMs; otherwise an in-RAM QuantileDMatrix is built. Both replay
        the same batch sequence across XGBoost's sketch/fill passes via reset().
        """
        use_external = self._should_use_external_memory(dataloader)
        cache_prefix = None
        if use_external:
            cache_root = self._cache_root()
            cache_prefix = str(Path(cache_root) / split_name)
            if progress_label is not None:
                free_gib = shutil.disk_usage(cache_root).free / 1024**3
                xgb_support._progress_log(
                    f"External-memory cache dir: {cache_root} "
                    f"| free={free_gib:.0f} GiB",
                    label=progress_label,
                )
        data_iter = _ResidueDataIter(
            loader_factory=lambda: dataloader,
            decode_fn=self._flatten_batch,
            cache_prefix=cache_prefix,
        )
        impl = data_iter.as_xgb_iter()
        build_start = time.perf_counter()
        dmatrix_cls = (
            xgb_support.xgb.ExtMemQuantileDMatrix
            if use_external
            else xgb_support.xgb.QuantileDMatrix
        )
        try:
            dmatrix = dmatrix_cls(
                impl, max_bin=self.max_bin, ref=ref, nthread=self.n_jobs
            )
        except Exception as exc:  # noqa: BLE001 - preserve retry contract.
            if use_external and xgb_support._is_disk_write_error(exc):
                raise RuntimeError(
                    f"External-memory cache ran out of disk space at "
                    f"'{self._cache_root_}' while building the {split_name} matrix. "
                    "Point XGB_RESIDUE_CACHE_DIR at a larger filesystem (e.g. "
                    "$SCRATCH), or reduce the residue count fed to XGBoost."
                ) from exc
            raise
        if progress_label is not None:
            kind = "ExtMemQuantileDMatrix" if use_external else "QuantileDMatrix"
            xgb_support._progress_log(
                f"Built {split_name} {kind} "
                f"| proteins={impl.proteins_seen:,} "
                f"| residues={impl.residues_seen:,} "
                f"| classes_seen={sorted(impl.labels_seen)} "
                f"| max_bin={self.max_bin} "
                f"| elapsed={time.perf_counter() - build_start:.1f}s",
                label=progress_label,
            )
        return dmatrix, impl

    def _prepare_fit_resume(
        self,
        checkpoint_path: str | Path | None,
        progress_label: str | None,
    ) -> tuple[_Booster | None, int, bool]:
        """Restore checkpoint state and report whether training is already done."""
        resumed = xgb_support._maybe_resume_xgb_wrapper(self, checkpoint_path)
        if resumed and self.constant_prediction_ is not None:
            if progress_label is not None:
                xgb_support._progress_log(
                    "Resuming from constant-prediction residue XGBoost checkpoint.",
                    label=progress_label,
                )
            return None, 0, True

        # A checkpoint without `training_complete_` predates mid-training
        # checkpointing and was only ever written after a finished fit.
        was_complete = getattr(self, "training_complete_", True)
        if resumed and self.booster is not None and was_complete:
            if progress_label is not None:
                xgb_support._progress_log(
                    "Resuming from completed residue XGBoost checkpoint.",
                    label=progress_label,
                )
            return self.booster, 0, True

        resume_booster = self.booster if resumed else None
        rounds_done = (
            int(getattr(self, "rounds_completed_", 0))
            if resume_booster is not None
            else 0
        )
        if resume_booster is None:
            self.constant_prediction_ = None
            self.booster = None
            self.best_iteration_ = None
            self.best_score_ = None
            self.rounds_completed_ = 0
        self.model_device_ = self.device
        self._log_fit_start(resume_booster, rounds_done, progress_label)
        return resume_booster, rounds_done, False

    def _log_fit_start(
        self,
        resume_booster: _Booster | None,
        rounds_done: int,
        progress_label: str | None,
    ) -> None:
        """Log either a fresh or resumed residue training run."""
        if progress_label is None:
            return
        if resume_booster is not None:
            xgb_support._progress_log(
                f"Resuming residue XGBoost boosting at round {rounds_done}"
                f"/{self.n_estimators} | device={self.model_device_}",
                label=progress_label,
            )
            return
        xgb_support._progress_log(
            "Residue XGBoost streaming fit starting "
            f"| device={self.model_device_}",
            label=progress_label,
        )

    def _build_training_matrix(
        self,
        dataloader: BatchLoader,
        progress_label: str | None,
    ) -> tuple[object, _ResidueIterState]:
        """Build the training matrix, falling back from CUDA to CPU when needed."""
        try:
            return self._build_quantile_dmatrix(
                dataloader,
                progress_label=progress_label,
                split_name="train",
            )
        except Exception as exc:  # noqa: BLE001 - preserve retry contract.
            if self.model_device_ == "cpu" or not xgb_support._is_xgb_cuda_failure(
                exc
            ):
                raise
            xgb_support._progress_log(
                "GPU failure building residue QuantileDMatrix "
                f"({type(exc).__name__}: {exc}); retrying on CPU.",
                label=progress_label,
            )
            self.model_device_ = "cpu"
            return self._build_quantile_dmatrix(
                dataloader,
                progress_label=progress_label,
                split_name="train",
            )

    def _finish_constant_fit(
        self,
        only_class: int,
        checkpoint_path: str | Path | None,
        progress_label: str | None,
    ) -> None:
        """Persist the degenerate one-class training result."""
        self.constant_prediction_ = float(only_class)
        self.model_device_ = None
        self.training_complete_ = True
        if progress_label is not None:
            xgb_support._progress_log(
                f"Residue XGBoost collapsed to constant prediction={only_class}.",
                label=progress_label,
            )
        xgb_support._save_xgb_checkpoint(self, checkpoint_path)

    def _build_validation_evals(
        self,
        val_dataloader: BatchLoader | None,
        dtrain: object,
        progress_label: str | None,
    ) -> list[tuple[object, str]]:
        """Build the optional validation matrix used for early stopping."""
        if val_dataloader is None:
            return []
        dval, val_impl = self._build_quantile_dmatrix(
            val_dataloader,
            ref=dtrain,
            progress_label=progress_label,
            split_name="val",
        )
        return [(dval, "validation")] if val_impl.residues_seen > 0 else []

    def _make_fit_kwargs(
        self,
        dtrain: object,
        evals: list[tuple[object, str]],
        remaining: int,
        resume_booster: _Booster | None,
        checkpoint_path: str | Path | None,
    ) -> dict[str, object]:
        """Assemble XGBoost training options for a fresh or resumed fit."""
        train_kwargs: dict[str, object] = {
            "params": self._make_train_params(device_override=self.model_device_),
            "dtrain": dtrain,
            "num_boost_round": remaining,
            "evals": evals,
            "verbose_eval": False,
        }
        if resume_booster is not None:
            train_kwargs["xgb_model"] = resume_booster
        checkpoint_cb = xgb_support._make_xgb_checkpoint_callback(
            self,
            checkpoint_path,
            self.checkpoint_interval,
        )
        if checkpoint_cb is not None:
            train_kwargs["callbacks"] = [checkpoint_cb]
        if evals and self.early_stopping_rounds is not None:
            train_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        return train_kwargs

    def _train_with_fallback(
        self,
        train_kwargs: dict[str, object],
        progress_label: str | None,
    ) -> _Booster:
        """Train a booster, retrying a failed CUDA run on CPU."""
        try:
            return typing.cast(_Booster, xgb_support.xgb.train(**train_kwargs))
        except Exception as exc:  # noqa: BLE001 - preserve retry contract.
            if self.model_device_ == "cpu" or not xgb_support._is_xgb_cuda_failure(
                exc
            ):
                raise
            xgb_support._progress_log(
                "GPU failure during residue XGBoost fit "
                f"({type(exc).__name__}: {exc}); retrying on CPU.",
                label=progress_label,
            )
            self.model_device_ = "cpu"
            train_kwargs["params"] = self._make_train_params(
                device_override="cpu"
            )
            return typing.cast(_Booster, xgb_support.xgb.train(**train_kwargs))

    def _finish_booster_fit(
        self,
        fit_start: float,
        checkpoint_path: str | Path | None,
        progress_label: str | None,
    ) -> None:
        """Record final booster metadata, log the result, and save a checkpoint."""
        if self.booster is None:
            raise RuntimeError("XGBoost training returned no booster.")
        self.best_iteration_ = getattr(self.booster, "best_iteration", None)
        self.best_score_ = getattr(self.booster, "best_score", None)
        self.rounds_completed_ = self.booster.num_boosted_rounds()
        self.training_complete_ = True

        if progress_label is not None:
            summary = (
                f"Residue XGBoost fit done in "
                f"{time.perf_counter() - fit_start:.1f}s"
            )
            if self.best_iteration_ is not None:
                summary += f" | best_iteration={self.best_iteration_}"
            if self.best_score_ is not None:
                summary += f" | best_score={self.best_score_}"
            summary += f" | device={self.model_device_}"
            xgb_support._progress_log(summary, label=progress_label)
        xgb_support._save_xgb_checkpoint(self, checkpoint_path)

    def fit_loader(
        self,
        dataloader: BatchLoader,
        batch_kind: str,
        val_dataloader: BatchLoader | None = None,
        progress_label: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> typing.Self:
        if batch_kind != "protein_residue":
            raise ValueError(
                f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'"
            )
        try:
            resume_booster, rounds_done, already_complete = (
                self._prepare_fit_resume(checkpoint_path, progress_label)
            )
            if already_complete:
                return self

            dtrain, train_impl = self._build_training_matrix(
                dataloader,
                progress_label,
            )
            if train_impl.residues_seen == 0:
                raise ValueError("No residue-level training samples found for XGBoost.")
            if len(train_impl.labels_seen) == 1:
                self._finish_constant_fit(
                    next(iter(train_impl.labels_seen)),
                    checkpoint_path,
                    progress_label,
                )
                return self

            evals = self._build_validation_evals(
                val_dataloader,
                dtrain,
                progress_label,
            )
            remaining = max(int(self.n_estimators) - rounds_done, 0)
            if remaining == 0:
                self.training_complete_ = True
                xgb_support._save_xgb_checkpoint(self, checkpoint_path)
                return self

            train_kwargs = self._make_fit_kwargs(
                dtrain,
                evals,
                remaining,
                resume_booster,
                checkpoint_path,
            )
            fit_start = time.perf_counter()
            self.booster = self._train_with_fallback(train_kwargs, progress_label)
            self._finish_booster_fit(
                fit_start,
                checkpoint_path,
                progress_label,
            )
            return self
        finally:
            self._release_cache()

    def _predict_loader(
        self,
        dataloader: BatchLoader,
        batch_kind: str,
        *,
        proba: bool,
    ) -> npt.NDArray[np.float32]:
        if batch_kind != "protein_residue":
            raise ValueError(
                f"XGBResidueBinaryClassifier does not support batch_kind='{batch_kind}'"
            )

        if self.booster is None and self.constant_prediction_ is None:
            raise ValueError(
                "XGBResidueBinaryClassifier must be fit before calling "
                "predict_loader()."
            )

        pred_rows: list[np.ndarray] = []
        for batch in dataloader:
            x_flat, _, lengths = self._flatten_batch(batch)
            if self.constant_prediction_ is not None:
                flat_pred = np.full(
                    (x_flat.shape[0],), self.constant_prediction_, dtype=np.float32
                )
            else:
                raw_pred = self.booster.inplace_predict(x_flat, validate_features=False)
                scores = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
                flat_pred = (
                    scores if proba else (scores >= 0.5).astype(np.float32, copy=False)
                )
            pred_rows.extend(self._split_flat_predictions(flat_pred, lengths))

        return enzymado.models.common.pad_1d_array_list(pred_rows, dtype=np.float32)

    def predict_loader(
        self,
        dataloader: BatchLoader,
        batch_kind: str,
    ) -> npt.NDArray[np.float32]:
        return self._predict_loader(dataloader, batch_kind, proba=False)

    def predict_proba_loader(
        self,
        dataloader: BatchLoader,
        batch_kind: str,
    ) -> npt.NDArray[np.float32]:
        """Return raw booster probabilities instead of thresholded hard labels."""
        return self._predict_loader(dataloader, batch_kind, proba=True)
