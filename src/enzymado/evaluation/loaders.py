"""Load prediction artifacts without coupling evaluation to training."""

from __future__ import annotations

import dataclasses
import json
import lzma
import math
import pathlib
import re

import numpy as np
import numpy.typing as npt
import safetensors.numpy

from enzymado.artifacts import sample_identity, serialization
from enzymado.evaluation import task_types


@dataclasses.dataclass(frozen=True)
class PredictionPayload:
    """Arrays required to evaluate one prediction block."""

    y_true: npt.NDArray
    predictions: npt.NDArray
    target_mask: npt.NDArray | None
    sample_ids: tuple[str, ...] | None = None
    proba: npt.NDArray | None = None


def _safetensors_load(blob: bytes) -> dict[str, npt.NDArray]:
    """Deserialize one in-memory safetensors payload."""
    return safetensors.numpy.load(blob)


def _separate_sample_ids(
    tensors: dict[str, npt.NDArray],
) -> tuple[dict[str, npt.NDArray], tuple[str, ...] | None]:
    """Remove and decode optional sample-identity tensors."""
    present_keys = sample_identity.SAMPLE_ID_TENSOR_KEYS & tensors.keys()
    if not present_keys:
        return tensors, None
    if present_keys != sample_identity.SAMPLE_ID_TENSOR_KEYS:
        missing = sample_identity.SAMPLE_ID_TENSOR_KEYS - present_keys
        raise ValueError(
            f"Incomplete sample identity tensors; missing {sorted(missing)}."
        )

    prediction_tensors = {
        key: value
        for key, value in tensors.items()
        if key not in sample_identity.SAMPLE_ID_TENSOR_KEYS
    }
    sample_ids = sample_identity.decode(
        tensors[sample_identity.SAMPLE_ID_BYTES_KEY],
        tensors[sample_identity.SAMPLE_ID_OFFSETS_KEY],
    )
    return prediction_tensors, sample_ids


def _validate_sample_id_count(
    sample_ids: tuple[str, ...] | None,
    n_rows: int,
) -> None:
    """Reject identity metadata that does not describe every prediction row."""
    if sample_ids is not None and len(sample_ids) != n_rows:
        raise ValueError(
            f"Prediction artifact has {len(sample_ids)} sample IDs for {n_rows} rows."
        )


def _multi_head_blocks(
    tensors: dict[str, npt.NDArray],
) -> tuple[PredictionPayload, ...] | None:
    """Return validated EC heads while retaining their compact masks."""
    prediction_tensors, sample_ids = _separate_sample_ids(tensors)
    pattern = re.compile(
        r"^(?P<kind>y_true|predictions|target_mask|proba)_level_(?P<level>\d+)$"
    )
    levels: dict[int, dict[str, npt.NDArray]] = {}
    for key, value in prediction_tensors.items():
        match = pattern.match(key)
        if match is None:
            return None
        level = int(match.group("level"))
        levels.setdefault(level, {})[match.group("kind")] = value

    expected_kinds = {"y_true", "predictions", "target_mask"}
    blocks: list[PredictionPayload] = []
    for level in sorted(levels):
        parts = levels[level]
        proba = parts.pop("proba", None)
        if set(parts) != expected_kinds:
            raise ValueError(
                f"Multi-head EC level {level} has keys {sorted(parts)}; "
                f"expected {sorted(expected_kinds)}."
            )

        y_true = parts["y_true"]
        predictions = parts["predictions"]
        target_mask = parts["target_mask"]
        if proba is not None:
            proba = serialization.dequantize_probabilities(proba)
        if y_true.ndim != 2 or predictions.shape != y_true.shape:
            raise ValueError(
                f"Multi-head EC level {level} requires matching 2D arrays; "
                f"got {y_true.shape} and {predictions.shape}."
            )
        valid_mask_shape = target_mask.shape == y_true.shape
        valid_row_mask = (
            target_mask.ndim == 1 and target_mask.shape[0] == y_true.shape[0]
        )
        if not valid_mask_shape and not valid_row_mask:
            raise ValueError(
                f"Multi-head EC level {level} mask {target_mask.shape} is not "
                f"compatible with targets {y_true.shape}."
            )
        _validate_sample_id_count(sample_ids, y_true.shape[0])
        blocks.append(
            PredictionPayload(
                y_true,
                predictions,
                target_mask,
                sample_ids=sample_ids,
                proba=proba,
            )
        )
    return tuple(blocks)


def _broadcast_mask(payload: PredictionPayload) -> npt.NDArray[np.bool_]:
    """Expand a per-row mask only when a dense payload is requested."""
    if payload.target_mask is None:
        return np.ones(payload.y_true.shape, dtype=bool)

    target_mask = np.asarray(payload.target_mask, dtype=bool)
    if target_mask.shape == payload.y_true.shape:
        return target_mask
    shape = (target_mask.shape[0],) + (1,) * (payload.y_true.ndim - 1)
    return np.broadcast_to(target_mask.reshape(shape), payload.y_true.shape)


def _read_safetensors_blocks(
    path: pathlib.Path,
) -> tuple[PredictionPayload, ...]:
    """Read one compressed safetensors artifact as logical prediction blocks."""
    with lzma.open(path, "rb") as handle:
        tensors = _safetensors_load(handle.read())

    prediction_tensors, sample_ids = _separate_sample_ids(tensors)
    if "y_true" in prediction_tensors and "predictions" in prediction_tensors:
        y_true = prediction_tensors["y_true"]
        _validate_sample_id_count(sample_ids, y_true.shape[0])
        raw_proba = prediction_tensors.get("proba")
        return (
            PredictionPayload(
                y_true=y_true,
                predictions=prediction_tensors["predictions"],
                target_mask=prediction_tensors.get("target_mask"),
                sample_ids=sample_ids,
                proba=(
                    serialization.dequantize_probabilities(raw_proba)
                    if raw_proba is not None
                    else None
                ),
            ),
        )

    blocks = _multi_head_blocks(tensors)
    if blocks is None:
        raise KeyError(f"{path.name} has unsupported tensor keys {sorted(tensors)}.")
    return blocks


def _read_npz(path: pathlib.Path) -> PredictionPayload:
    """Read the legacy NumPy prediction representation."""
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = (
            tuple(str(value) for value in archive["sample_ids"])
            if "sample_ids" in archive.files
            else None
        )
        y_true = archive["y_true"]
        _validate_sample_id_count(sample_ids, y_true.shape[0])
        return PredictionPayload(
            y_true=y_true,
            predictions=archive["predictions"],
            target_mask=(
                archive["target_mask"] if "target_mask" in archive.files else None
            ),
            sample_ids=sample_ids,
        )


def find_split_prediction_path(
    run_dir: pathlib.Path,
    split_name: str,
) -> pathlib.Path | None:
    """Locate the canonical prediction artifact for one split."""
    safetensors_paths = sorted(
        run_dir.glob(f"*_{split_name}_predictions.safetensors.xz")
    )
    if safetensors_paths:
        return safetensors_paths[0]

    npz_path = run_dir / f"{split_name}_predictions.npz"
    return npz_path if npz_path.exists() else None


def load_split_blocks(
    run_dir: pathlib.Path,
    split_name: str,
) -> tuple[PredictionPayload, ...] | None:
    """Load prediction blocks without concatenating multi-head EC tensors."""
    path = find_split_prediction_path(run_dir, split_name)
    if path is None:
        return None
    if path.name.endswith(".safetensors.xz"):
        blocks = _read_safetensors_blocks(path)
    else:
        blocks = (_read_npz(path),)
    _validate_artifact_sample_identity(
        run_dir,
        split_name,
        blocks[0].sample_ids,
    )
    return blocks


def load_split_predictions(
    run_dir: pathlib.Path,
    split_name: str,
) -> PredictionPayload | None:
    """Load one split as a dense payload for backward-compatible callers."""
    blocks = load_split_blocks(run_dir, split_name)
    if blocks is None:
        return None
    if len(blocks) == 1:
        return blocks[0]
    return PredictionPayload(
        y_true=np.concatenate([block.y_true for block in blocks], axis=1),
        predictions=np.concatenate(
            [block.predictions for block in blocks],
            axis=1,
        ),
        target_mask=np.concatenate(
            [_broadcast_mask(block) for block in blocks],
            axis=1,
        ),
        sample_ids=blocks[0].sample_ids,
    )


def load_test_predictions(run_dir: pathlib.Path) -> PredictionPayload | None:
    """Load test predictions as one dense payload."""
    return load_split_predictions(run_dir, "test")


def _read_json_object(path: pathlib.Path) -> dict[str, object]:
    """Read and validate a top-level JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def read_metrics(run_dir: pathlib.Path) -> dict[str, object]:
    """Read metrics when present without treating them as authoritative metadata."""
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return _read_json_object(path)


def _json_ready(value: object) -> object:
    """Convert scientific values to strict JSON-compatible objects."""
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def write_metrics(run_dir: pathlib.Path, payload: dict[str, object]) -> None:
    """Write an experiment metrics artifact."""
    path = run_dir / "metrics.json"
    path.write_text(
        json.dumps(_json_ready(payload), indent=4, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_hparams(run_dir: pathlib.Path) -> dict[str, object]:
    """Read optional training metadata."""
    path = run_dir / "hparams.json"
    if not path.exists():
        return {}
    return _read_json_object(path)


def read_sample_identity(
    run_dir: pathlib.Path,
    split_name: str,
) -> tuple[int, str] | None:
    """Read and validate an ordered sample fingerprint from hparams.json."""
    identity_block = read_hparams(run_dir).get("sample_identity")
    if not isinstance(identity_block, dict):
        return None
    split_block = identity_block.get(split_name)
    if not isinstance(split_block, dict):
        return None

    count = split_block.get("count")
    ordered_sha256 = split_block.get("ordered_sha256")
    valid_count = isinstance(count, int) and not isinstance(count, bool)
    valid_digest = (
        isinstance(ordered_sha256, str)
        and len(ordered_sha256) == 64
        and all(character in "0123456789abcdef" for character in ordered_sha256)
    )
    if not valid_count or not valid_digest:
        raise ValueError(f"Invalid {split_name} sample identity metadata in {run_dir}.")
    return count, ordered_sha256


def _validate_artifact_sample_identity(
    run_dir: pathlib.Path,
    split_name: str,
    sample_ids: tuple[str, ...] | None,
) -> None:
    """Ensure embedded IDs agree with the compact training manifest."""
    expected = read_sample_identity(run_dir, split_name)
    if expected is None:
        return
    if sample_ids is None:
        raise ValueError(
            f"{run_dir} declares sample identity metadata but its "
            f"{split_name} predictions do not contain sample IDs."
        )
    observed = (len(sample_ids), sample_identity.fingerprint(sample_ids))
    if observed != expected:
        raise ValueError(
            f"Embedded {split_name} sample IDs do not match hparams.json in {run_dir}."
        )


def derive_task_type(hparams: dict[str, object]) -> str | None:
    """Map every supported label kind to classification or regression."""
    return task_types.from_hparams(hparams)
