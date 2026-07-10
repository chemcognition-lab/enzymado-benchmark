"""Uniform readers for prediction artifacts across the two on-disk formats.

Tabular runs (mean, attention) store predictions as `<stem>_test_predictions.safetensors.xz`.
Per-residue runs store them as `test_predictions.npz`.

Some attention-based EC runs use a multi-head schema where each of the four EC
levels has its own `(y_true_level_k, predictions_level_k, target_mask_level_k)`
trio. We unify those by horizontally concatenating the four levels into a
single `(N, sum(C_k))` tensor. Older payloads used a per-sample 1D mask per
level; newer payloads may use a per-label 2D mask matching `y_true`.
"""

from __future__ import annotations

import json
import lzma
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PredictionPayload:
    y_true: np.ndarray
    predictions: np.ndarray
    target_mask: Optional[np.ndarray]


def _safetensors_load(blob: bytes) -> dict[str, np.ndarray]:
    from safetensors.numpy import load
    return load(blob)


_LEVEL_RE = re.compile(r"^(?P<kind>y_true|predictions|target_mask)_level_(?P<level>\d+)$")


def _unify_multi_head_levels(out: dict[str, np.ndarray]) -> PredictionPayload | None:
    """Concatenate per-level multi-head EC tensors into a single payload.

    Returns None if the dict does not match the multi-head schema.
    """
    levels: dict[int, dict[str, np.ndarray]] = {}
    for key, value in out.items():
        match = _LEVEL_RE.match(key)
        if not match:
            return None
        levels.setdefault(int(match.group("level")), {})[match.group("kind")] = value

    expected_kinds = {"y_true", "predictions", "target_mask"}
    for level, parts in levels.items():
        if set(parts.keys()) != expected_kinds:
            raise ValueError(
                f"Multi-head EC level {level} is missing keys; "
                f"got {sorted(parts.keys())}"
            )

    y_blocks: list[np.ndarray] = []
    p_blocks: list[np.ndarray] = []
    m_blocks: list[np.ndarray] = []
    for level in sorted(levels.keys()):
        y_k = levels[level]["y_true"]
        p_k = levels[level]["predictions"]
        m_k = levels[level]["target_mask"]
        if y_k.ndim != 2 or p_k.ndim != 2 or y_k.shape != p_k.shape:
            raise ValueError(
                f"Multi-head EC level {level}: y_true/predictions must be matching 2D arrays, "
                f"got {y_k.shape} and {p_k.shape}."
            )
        if m_k.ndim == 1 and m_k.shape[0] == y_k.shape[0]:
            m_2d = np.broadcast_to(m_k.astype(bool)[:, None], y_k.shape)
        elif m_k.ndim == 2 and m_k.shape == y_k.shape:
            m_2d = m_k.astype(bool)
        else:
            raise ValueError(
                f"Multi-head EC level {level}: target_mask must be 1D with length "
                f"{y_k.shape[0]} or 2D matching y_true shape {y_k.shape}; "
                f"got shape {m_k.shape}."
            )
        y_blocks.append(y_k)
        p_blocks.append(p_k)
        m_blocks.append(m_2d)

    return PredictionPayload(
        y_true=np.concatenate(y_blocks, axis=1),
        predictions=np.concatenate(p_blocks, axis=1),
        target_mask=np.concatenate(m_blocks, axis=1),
    )


def _read_safetensors_xz(path: Path) -> PredictionPayload:
    with lzma.open(path, "rb") as f:
        blob = f.read()
    out = _safetensors_load(blob)
    if "y_true" in out and "predictions" in out:
        return PredictionPayload(
            y_true=out["y_true"],
            predictions=out["predictions"],
            target_mask=out.get("target_mask"),
        )
    unified = _unify_multi_head_levels(out)
    if unified is not None:
        return unified
    raise KeyError(
        f"{path.name}: could not interpret keys {sorted(out.keys())} "
        "(expected either 'y_true'/'predictions' or per-level multi-head EC keys)."
    )


def _read_npz(path: Path) -> PredictionPayload:
    arr = np.load(path)
    return PredictionPayload(
        y_true=arr["y_true"],
        predictions=arr["predictions"],
        target_mask=arr["target_mask"] if "target_mask" in arr.files else None,
    )


def load_split_predictions(run_dir: Path, split_name: str) -> PredictionPayload | None:
    """Locate and read this experiment's <split_name> predictions, returning None if missing."""
    xz = sorted(run_dir.glob(f"*_{split_name}_predictions.safetensors.xz"))
    if xz:
        return _read_safetensors_xz(xz[0])
    npz = run_dir / f"{split_name}_predictions.npz"
    if npz.exists():
        return _read_npz(npz)
    return None


def load_test_predictions(run_dir: Path) -> PredictionPayload | None:
    """Locate and read this experiment's test predictions, returning None if missing."""
    return load_split_predictions(run_dir, "test")


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_metrics(run_dir: Path, payload: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=4) + "\n")


def read_hparams(run_dir: Path) -> dict:
    path = run_dir / "hparams.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def derive_task_type(hparams: dict) -> str | None:
    """Infer 'classification'/'regression' from hparams.json's label_kind.

    Mirrors the task_type computation in scripts/run_benchmark.py so evaluation
    can determine task_type without depending on metrics.json already existing.
    """
    label_kind = (hparams.get("task_config") or {}).get("label_kind")
    if label_kind is None:
        return None
    return "classification" if label_kind == "binary" else "regression"
