from __future__ import annotations
"""Batch collation utilities for the benchmark's supported task types."""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _pad_residue_embeddings(embeddings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length residue embeddings and return a boolean residue mask."""
    if not embeddings:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(embedding.shape[0] for embedding in embeddings)
    embedding_dim = embeddings[0].shape[1]

    batch = np.zeros((len(embeddings), max_length, embedding_dim), dtype=np.float32)
    mask = np.zeros((len(embeddings), max_length), dtype=bool)
    for index, embedding in enumerate(embeddings):
        length = embedding.shape[0]
        batch[index, :length, :] = embedding
        mask[index, :length] = True
    return batch, mask


def _pad_vectors(vectors: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    """Pad 1D arrays to the maximum length present in the batch."""
    max_length = max(vector.shape[0] for vector in vectors)
    batch = np.zeros((len(vectors), max_length), dtype=dtype)
    for index, vector in enumerate(vectors):
        batch[index, : vector.shape[0]] = vector
    return batch


def _stack_targets(targets: list[Any]) -> np.ndarray:
    """Stack scalar, vector, or nested target payloads into batch arrays."""
    if isinstance(targets[0], dict):
        return {
            key: _stack_targets([target[key] for target in targets])
            for key in targets[0]
        }
    arrays = [np.asarray(target) for target in targets]
    if all(array.ndim == 0 for array in arrays):
        dtype = bool if arrays[0].dtype == np.bool_ else np.float32
        return np.asarray(arrays, dtype=dtype)
    dtype = bool if all(array.dtype == np.bool_ for array in arrays) else np.float32
    return np.stack(arrays).astype(dtype, copy=False)


def _validate_residue_sample_lengths(samples: list[dict[str, Any]]) -> None:
    """Validate residue-level samples before they are padded into a batch."""
    for sample in samples:
        sample_id = sample.get("sample_id", "")
        embedding_length = sample["protein_emb"].shape[0]
        target_length = sample["target_per_residue"].shape[0]
        mask_length = sample["target_mask"].shape[0]

        if embedding_length != target_length:
            raise ValueError(
                "Protein embedding and residue target length mismatch for sample "
                f"'{sample_id}': embedding={embedding_length}, target={target_length}."
            )
        if target_length != mask_length:
            raise ValueError(
                "Residue target and residue mask length mismatch for sample "
                f"'{sample_id}': target={target_length}, mask={mask_length}."
            )


@dataclass
class ProteinGlobalBatch:
    """Padded batch for protein-level tasks without ligand features."""
    sample_id: list[str]
    protein_emb: np.ndarray
    protein_mask: np.ndarray
    target: Any
    target_mask: Any | None = None


@dataclass
class ProteinLigandGlobalBatch:
    """Padded batch for protein-ligand tasks."""
    sample_id: list[str]
    protein_emb: np.ndarray
    protein_mask: np.ndarray
    ligand_feat: np.ndarray
    target: Any
    target_mask: Any | None = None


@dataclass
class MutationDeltaGlobalBatch:
    """Padded batch for mutation-delta tasks."""
    sample_id: list[str]
    protein_delta_emb: np.ndarray
    protein_mask: np.ndarray
    target: Any
    target_mask: Any | None = None


@dataclass
class ProteinResidueBatch:
    """Padded batch for residue-wise supervision."""
    sample_id: list[str]
    protein_emb: np.ndarray
    protein_mask: np.ndarray
    target_per_residue: np.ndarray
    target_mask: np.ndarray


class ProteinGlobalCollator:
    """Collate global protein samples into a padded batch."""
    def __call__(self, samples: list[dict[str, Any]]) -> ProteinGlobalBatch:
        protein_emb, protein_mask = _pad_residue_embeddings(
            [sample["protein_emb"] for sample in samples]
        )
        return ProteinGlobalBatch(
            sample_id=[sample["sample_id"] for sample in samples],
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            target=_stack_targets([sample["target"] for sample in samples]),
            target_mask=(
                _stack_targets([sample["target_mask"] for sample in samples])
                if "target_mask" in samples[0]
                else None
            ),
        )


class ProteinLigandGlobalCollator:
    """Collate protein and ligand features for joint prediction tasks."""
    def __call__(self, samples: list[dict[str, Any]]) -> ProteinLigandGlobalBatch:
        protein_emb, protein_mask = _pad_residue_embeddings(
            [sample["protein_emb"] for sample in samples]
        )
        ligand_feat = np.stack([sample["ligand_feat"] for sample in samples]).astype(
            np.float32,
            copy=False,
        )
        return ProteinLigandGlobalBatch(
            sample_id=[sample["sample_id"] for sample in samples],
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            ligand_feat=ligand_feat,
            target=_stack_targets([sample["target"] for sample in samples]),
            target_mask=(
                _stack_targets([sample["target_mask"] for sample in samples])
                if "target_mask" in samples[0]
                else None
            ),
        )


class MutationDeltaGlobalCollator:
    """Collate mutation-delta embeddings for regression or classification."""
    def __call__(self, samples: list[dict[str, Any]]) -> MutationDeltaGlobalBatch:
        protein_delta_emb, protein_mask = _pad_residue_embeddings(
            [sample["protein_delta_emb"] for sample in samples]
        )
        return MutationDeltaGlobalBatch(
            sample_id=[sample["sample_id"] for sample in samples],
            protein_delta_emb=protein_delta_emb,
            protein_mask=protein_mask,
            target=_stack_targets([sample["target"] for sample in samples]),
            target_mask=(
                _stack_targets([sample["target_mask"] for sample in samples])
                if "target_mask" in samples[0]
                else None
            ),
        )


class ProteinResidueCollator:
    """Collate residue-level labels while preserving residue masks."""
    def __call__(self, samples: list[dict[str, Any]]) -> ProteinResidueBatch:
        _validate_residue_sample_lengths(samples)
        protein_emb, protein_mask = _pad_residue_embeddings(
            [sample["protein_emb"] for sample in samples]
        )
        target_per_residue = _pad_vectors(
            [sample["target_per_residue"] for sample in samples],
            dtype=np.float32,
        )
        target_mask = _pad_vectors(
            [sample["target_mask"].astype(np.int8, copy=False) for sample in samples],
            dtype=np.int8,
        ).astype(bool, copy=False)
        return ProteinResidueBatch(
            sample_id=[sample["sample_id"] for sample in samples],
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            target_per_residue=target_per_residue,
            target_mask=target_mask,
        )
