"""Batch collation utilities for the benchmark's supported task types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def _to_torch_residue_tensor(embedding: Any) -> torch.Tensor:
    """Coerce a residue embedding to a 2D torch tensor (bfloat16 if available).

    Embeddings from the safetensors-backed store arrive as np.ndarray with
    dtype=int16 holding raw BF16 bits, since NumPy has no native bfloat16 type.
    """
    if torch.is_tensor(embedding):
        return embedding
    array = np.asarray(embedding)
    if array.dtype == np.int16:
        return torch.from_numpy(array).view(torch.bfloat16)
    return torch.from_numpy(array)


def _pad_residue_embeddings(
    embeddings: list[Any],
) -> tuple[torch.Tensor, np.ndarray]:
    """Pad variable-length residue embeddings and return a boolean residue mask.

    Embeddings arrive as ``torch.bfloat16`` tensors from the safetensors-backed
    store, but plain numpy arrays (e.g. mutation deltas materialized as float32)
    are also accepted.  Padding uses ``+0.0`` in the embedding dtype, which is
    the correct fill for both bfloat16 and float32.
    """
    if not embeddings:
        raise ValueError("Cannot collate an empty batch.")
    tensors = [_to_torch_residue_tensor(emb) for emb in embeddings]
    lengths = [tensor.shape[0] for tensor in tensors]
    max_length = max(lengths)

    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)

    mask = np.zeros((len(tensors), max_length), dtype=bool)
    for index, length in enumerate(lengths):
        mask[index, :length] = True
    return padded, mask


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

        if embedding_length != target_length:
            raise ValueError(
                "Protein embedding and residue target length mismatch for sample "
                f"'{sample_id}': embedding={embedding_length}, target={target_length}."
            )


@dataclass
class Batch:
    """General padded batch for protein prediction tasks"""

    sample_id: list[str]

    # Protein related
    protein_emb: torch.Tensor | None = None
    protein_mask: np.ndarray | None = None
    protein_delta_emb: torch.Tensor | None = None

    # Ligand related
    ligand_feat: np.ndarray | None = None

    # Numeric experimental context
    context_feat: np.ndarray | None = None

    # target related
    target: Any | None = None
    target_per_residue: np.ndarray | None = None
    target_mask: np.ndarray | None = None

    # Concat-related
    lengths: np.ndarray | None = None


class Collator:
    def _extract(self, samples: list[dict[str, Any]], key: str) -> list[Any]:
        return [sample[key] for sample in samples]

    def _optional_target_mask(self, samples: list[dict[str, Any]]) -> np.ndarray | None:
        if "target_mask" in samples[0]:
            return _stack_targets(self._extract(samples, "target_mask"))
        return None


class _GlobalCollatorBase(Collator):
    def _global_batch_kwargs(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(
            sample_id=self._extract(samples, "sample_id"),
            target=_stack_targets(self._extract(samples, "target")),
            target_mask=self._optional_target_mask(samples),
        )

    def _optional_context_features(
        self,
        samples: list[dict[str, Any]],
    ) -> np.ndarray | None:
        present = ["context_feat" in sample for sample in samples]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(
                "context_feat must be present for every sample in a batch."
            )
        return np.stack(self._extract(samples, "context_feat")).astype(
            np.float32,
            copy=False,
        )


class ProteinGlobalCollator(_GlobalCollatorBase):
    """Collate global protein samples into a padded batch."""

    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        protein_emb, protein_mask = _pad_residue_embeddings(
            self._extract(samples, "protein_emb")
        )
        return Batch(
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            context_feat=self._optional_context_features(samples),
            **self._global_batch_kwargs(samples),
        )


class ProteinLigandGlobalCollator(_GlobalCollatorBase):
    """Collate protein and ligand features for joint prediction tasks."""

    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        protein_emb, protein_mask = _pad_residue_embeddings(
            self._extract(samples, "protein_emb")
        )
        ligand_feat = np.stack(self._extract(samples, "ligand_feat")).astype(
            np.float32, copy=False
        )
        return Batch(
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            ligand_feat=ligand_feat,
            context_feat=self._optional_context_features(samples),
            **self._global_batch_kwargs(samples),
        )


class MutationDeltaGlobalCollator(_GlobalCollatorBase):
    """Collate mutation-delta embeddings for regression or classification."""

    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        protein_delta_emb, protein_mask = _pad_residue_embeddings(
            self._extract(samples, "protein_delta_emb")
        )
        return Batch(
            protein_delta_emb=protein_delta_emb,
            protein_mask=protein_mask,
            **self._global_batch_kwargs(samples),
        )


class _ResidueCollatorBase(Collator):
    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        _validate_residue_sample_lengths(samples)
        return self._collate(samples)

    def _collate(self, samples: list[dict[str, Any]]) -> Batch:
        raise NotImplementedError


class ProteinResidueCollator(_ResidueCollatorBase):
    """Collate residue-level labels while preserving residue masks."""

    def _collate(self, samples: list[dict[str, Any]]) -> Batch:
        protein_emb, protein_mask = _pad_residue_embeddings(
            self._extract(samples, "protein_emb")
        )
        target_per_residue = _pad_vectors(
            self._extract(samples, "target_per_residue"), dtype=np.float32
        )
        # Residue targets are padded to the same length as protein_emb (enforced by
        # _validate_residue_sample_lengths), so protein_mask already flags which
        # positions are real residues vs. padding for the target array too.
        return Batch(
            sample_id=self._extract(samples, "sample_id"),
            protein_emb=protein_emb,
            protein_mask=protein_mask,
            target_per_residue=target_per_residue,
            target_mask=protein_mask,
        )


class ProteinResidueConcatCollator(_ResidueCollatorBase):
    """Collate residue-level labels by concatenating residues across proteins."""

    def _collate(self, samples: list[dict[str, Any]]) -> Batch:
        lengths = np.asarray(
            [sample["protein_emb"].shape[0] for sample in samples], dtype=np.int32
        )
        if not np.all(lengths > 0):
            raise ValueError(
                "ProteinResidueConcatCollator requires non-empty residue embeddings."
            )

        protein_emb = torch.cat(
            [_to_torch_residue_tensor(sample["protein_emb"]) for sample in samples],
            dim=0,
        )
        return Batch(
            sample_id=self._extract(samples, "sample_id"),
            protein_emb=protein_emb,
            target_per_residue=np.concatenate(
                [
                    np.asarray(sample["target_per_residue"], dtype=np.int8)
                    for sample in samples
                ],
                axis=0,
            ),
            lengths=lengths,
        )
