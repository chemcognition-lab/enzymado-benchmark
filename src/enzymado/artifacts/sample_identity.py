"""Encode sample identifiers and build deterministic dataset fingerprints."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np
import numpy.typing as npt

SAMPLE_ID_BYTES_KEY = "sample_id_utf8"
SAMPLE_ID_OFFSETS_KEY = "sample_id_offsets"
SAMPLE_ID_TENSOR_KEYS = frozenset({SAMPLE_ID_BYTES_KEY, SAMPLE_ID_OFFSETS_KEY})


def normalize(sample_ids: Sequence[str]) -> tuple[str, ...]:
    """Return validated, non-empty string identifiers without changing order."""
    normalized = tuple(str(sample_id).strip() for sample_id in sample_ids)
    empty_positions = [
        index for index, sample_id in enumerate(normalized) if not sample_id
    ]
    if empty_positions:
        preview = ", ".join(str(index) for index in empty_positions[:5])
        raise ValueError(f"Empty sample identifiers at row positions: {preview}.")
    return normalized


def encode(
    sample_ids: Sequence[str],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Encode variable-length UTF-8 identifiers as bytes plus row offsets."""
    normalized = normalize(sample_ids)
    encoded = [sample_id.encode("utf-8") for sample_id in normalized]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for index, value in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    joined = b"".join(encoded)
    byte_values = np.frombuffer(joined, dtype=np.uint8).copy()
    return byte_values, offsets


def decode(
    byte_values: npt.NDArray,
    offsets: npt.NDArray,
) -> tuple[str, ...]:
    """Decode and validate UTF-8 identifiers stored in numeric tensors."""
    bytes_array = np.asarray(byte_values, dtype=np.uint8)
    offset_array = np.asarray(offsets, dtype=np.int64)
    if bytes_array.ndim != 1 or offset_array.ndim != 1:
        raise ValueError("Sample ID bytes and offsets must both be one-dimensional.")
    if not len(offset_array) or offset_array[0] != 0:
        raise ValueError("Sample ID offsets must start at zero.")
    if np.any(offset_array[1:] < offset_array[:-1]):
        raise ValueError("Sample ID offsets must be monotonically increasing.")
    if offset_array[-1] != len(bytes_array):
        raise ValueError(
            "Final sample ID offset does not match the encoded byte length."
        )

    buffer = bytes_array.tobytes()
    decoded = tuple(
        buffer[start:end].decode("utf-8")
        for start, end in zip(offset_array[:-1], offset_array[1:])
    )
    return normalize(decoded)


def fingerprint(sample_ids: Sequence[str]) -> str:
    """Hash an ordered identifier sequence using an unambiguous JSON encoding."""
    normalized = normalize(sample_ids)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest(sample_ids: Sequence[str]) -> dict[str, int | str]:
    """Describe an ordered sample sequence without duplicating every identifier."""
    normalized = normalize(sample_ids)
    return {
        "count": len(normalized),
        "ordered_sha256": fingerprint(normalized),
    }
