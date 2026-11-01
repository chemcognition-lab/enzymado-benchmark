"""Shared backend, checkpoint, logging, and error helpers for XGBoost."""

from __future__ import annotations

from pathlib import Path
import time

from enzymado.models import training_protocol

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    from xgboost.core import XGBoostError
except Exception:  # noqa: BLE001 - see below
    # Deliberately broad. An absent package raises ModuleNotFoundError, but an
    # *installed* xgboost whose native library cannot load raises XGBoostError
    # (e.g. missing libomp.dylib on macOS), and XGBoostError is only importable
    # from inside this try, so it cannot be named here. Catching just
    # ModuleNotFoundError let that case propagate and break the import of every
    # module that reaches model_registry, taking run_benchmark.py down with it.
    xgb = None
    XGBClassifier = None
    XGBRegressor = None
    XGBoostError = None

import joblib
import numpy as np
import torch


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


def _is_disk_write_error(exc: Exception) -> bool:
    """Detect an external-memory cache write that failed for lack of disk space.

    XGBoost/dmlc raise `FileStream.Write incomplete` (a short std::fwrite) when
    the cache filesystem runs out of space or hits a quota — not a memory issue.
    """
    message = str(exc).lower()
    return "filestream.write" in message or "write incomplete" in message


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


def _make_xgb_checkpoint_callback(wrapper, path: str | Path | None, interval: int = 1):
    """Persist the wrapper (with its partially-boosted booster) every `interval` rounds.

    Without this a residue XGBoost fit that is killed mid-boosting loses every
    round it completed, since the wrapper is otherwise only saved once
    `xgb.train` returns — unlike the torch wrappers, which checkpoint per epoch.
    """
    if path is None or xgb is None:
        return None

    interval = max(1, int(interval))

    class _Checkpoint(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            # num_boosted_rounds() is authoritative when continuing from xgb_model,
            # where `epoch` restarts at 0 for the resumed call.
            rounds_done = model.num_boosted_rounds()
            if rounds_done % interval:
                return False
            wrapper.booster = model
            wrapper.rounds_completed_ = rounds_done
            wrapper.training_complete_ = False
            _save_xgb_checkpoint(wrapper, path)
            return False

    return _Checkpoint()


def _maybe_resume_xgb_wrapper(wrapper, path: str | Path | None) -> bool:
    path_obj = Path(path) if path is not None else None
    wrapper.training_contract_ = training_protocol.resume_contract(wrapper, path_obj)
    if path_obj is None:
        return False
    if not path_obj.exists():
        return False
    loaded = joblib.load(path_obj)
    if not isinstance(loaded, wrapper.__class__):
        raise TypeError(
            f"Checkpoint at {path_obj} contains {type(loaded).__name__}, "
            f"expected {wrapper.__class__.__name__}."
        )
    if getattr(loaded, "training_contract_", None) != wrapper.training_contract_:
        raise ValueError(
            "Incompatible XGBoost checkpoint; use a new run directory. "
            "Old models remain available for inference."
        )
    current_device = getattr(wrapper, "device", None)
    wrapper.__dict__.update(loaded.__dict__)
    if current_device is not None:
        wrapper.device = current_device
    return True
