"""Uniform readers for prediction artifacts across the two on-disk formats.

Tabular runs (mean, attention) store predictions as `<stem>_test_predictions.safetensors.xz`.
Per-residue runs store them as `test_predictions.npz`.
"""

from __future__ import annotations

import json
import lzma
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


def _read_safetensors_xz(path: Path) -> PredictionPayload:
    with lzma.open(path, "rb") as f:
        blob = f.read()
    out = _safetensors_load(blob)
    return PredictionPayload(
        y_true=out["y_true"],
        predictions=out["predictions"],
        target_mask=out.get("target_mask"),
    )


def _read_npz(path: Path) -> PredictionPayload:
    arr = np.load(path)
    return PredictionPayload(
        y_true=arr["y_true"],
        predictions=arr["predictions"],
        target_mask=arr["target_mask"] if "target_mask" in arr.files else None,
    )


def load_test_predictions(run_dir: Path) -> PredictionPayload | None:
    """Locate and read this experiment's test predictions, returning None if missing."""
    xz = sorted(run_dir.glob("*_test_predictions.safetensors.xz"))
    if xz:
        return _read_safetensors_xz(xz[0])
    npz = run_dir / "test_predictions.npz"
    if npz.exists():
        return _read_npz(npz)
    return None


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
