"""Feature builders and embedding stores used by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from enzymado.paths_and_constants import Paths

from .task_csv import clean_text, normalize_sequence


class EmbeddingLookupError(KeyError):
    """Raised when a row cannot be matched to a stored embedding."""

    pass


def _ensure_2d_tensor(value: torch.Tensor, key: str) -> torch.Tensor:
    """Validate that residue embeddings keep the expected `(length, dim)` shape."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim != 2:
        raise ValueError(
            f"Embedding for key '{key}' must be 2D. "
            f"Received shape {tuple(value.shape)}."
        )
    return value


@dataclass
class EmbeddingStore:
    """Base lookup store for embeddings indexed by sequence and/or identifier.

    Both dictionaries hold ``torch.Tensor`` values after ``__post_init__``.
    Subclasses override ``_coerce`` to enforce dtype or shape requirements.
    """

    by_sequence: dict[str, torch.Tensor] = field(default_factory=dict)
    by_id: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_sequence = {
            normalize_sequence(k): self._coerce(v, k)
            for k, v in self.by_sequence.items()
        }
        self.by_id = {clean_text(k): self._coerce(v, k) for k, v in self.by_id.items()}

    def _coerce(self, value, key: str) -> torch.Tensor:
        """Convert a raw value to a float32 tensor."""
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(np.asarray(value, dtype=np.float32))

    def has_sequence(self, sequence: str) -> bool:
        return normalize_sequence(sequence) in self.by_sequence

    def has_id(self, identifier: str) -> bool:
        return clean_text(identifier) in self.by_id

    def lookup_sequence(self, sequence: str) -> torch.Tensor:
        normalized = normalize_sequence(sequence)
        try:
            return self.by_sequence[normalized]
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing embedding for sequence '{normalized[:20]}...'"
            ) from exc

    def lookup_id(self, identifier: str) -> torch.Tensor:
        normalized = clean_text(identifier)
        try:
            return self.by_id[normalized]
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing embedding for id '{normalized}'."
            ) from exc

    def lookup_row(
        self,
        row: dict[str, str],
        id_columns: tuple[str, ...] = ("uniprot_id", "dataset_id"),
        sequence_column: str = "protein_sequence",
    ) -> torch.Tensor:
        """Resolve the best available embedding key for a dataset row."""
        sequence = normalize_sequence(row.get(sequence_column))
        if sequence and self.has_sequence(sequence):
            return self.lookup_sequence(sequence)
        for id_column in id_columns:
            if id_column not in row:
                continue
            identifier = clean_text(row.get(id_column))
            if identifier and self.has_id(identifier):
                return self.lookup_id(identifier)
        tried = [c for c in (*id_columns, sequence_column) if c in row]
        raise EmbeddingLookupError(
            "Missing embedding for row. "
            f"Tried columns: {tried}. sample_id={row.get('sample_id', '')}"
        )


@dataclass
class ResidueEmbeddingStore(EmbeddingStore):
    """Lookup store for per-residue embeddings of shape ``(length, dim)``.

    Tensor dtype is preserved from the source (typically bfloat16 from safetensors).
    """

    def _coerce(self, value, key: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            array = np.asarray(value)
            # Raw BF16 bit patterns arrive as int16 since NumPy has no native bfloat16.
            value = (
                torch.from_numpy(array).view(torch.bfloat16)
                if array.dtype == np.int16
                else torch.as_tensor(array)
            )
        return _ensure_2d_tensor(value, key)

    def lookup_sequence_length(self, sequence: str) -> int:
        normalized = normalize_sequence(sequence)
        try:
            return int(self.by_sequence[normalized].shape[0])
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing residue embedding for sequence '{normalized[:20]}...'"
            ) from exc

    def lookup_id_length(self, identifier: str) -> int:
        normalized = clean_text(identifier)
        try:
            return int(self.by_id[normalized].shape[0])
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing residue embedding for id '{normalized}'."
            ) from exc

    def lookup_row_length(
        self,
        row: dict[str, str],
        id_columns: tuple[str, ...] = ("uniprot_id", "dataset_id"),
        sequence_column: str = "protein_sequence",
    ) -> int:
        """Resolve the residue length for the best available embedding key for a row."""
        sequence = normalize_sequence(row.get(sequence_column))
        if sequence and self.has_sequence(sequence):
            return self.lookup_sequence_length(sequence)
        for id_column in id_columns:
            if id_column not in row:
                continue
            identifier = clean_text(row.get(id_column))
            if identifier and self.has_id(identifier):
                return self.lookup_id_length(identifier)
        tried = [c for c in (*id_columns, sequence_column) if c in row]
        raise EmbeddingLookupError(
            "Missing residue embedding for row. "
            f"Tried columns: {tried}. sample_id={row.get('sample_id', '')}"
        )


@dataclass
class PooledEmbeddingStore(EmbeddingStore):
    """Lookup store for pre-pooled (1D) embeddings of shape ``(dim,)``.

    Stored as float32 tensors. Callers that need NumPy arrays can call
    ``.float().cpu().numpy()`` on the result.
    """

    pass


@dataclass
class ProteinResidueEmbeddingBuilder:
    """Build residue embeddings for rows in protein-centric tasks."""

    embedding_store: EmbeddingStore
    sequence_column: str = "protein_sequence"
    index_columns: tuple[str, ...] = ("uniprot_id", "dataset_id")

    def build(self, row: dict[str, str]) -> torch.Tensor:
        return self.embedding_store.lookup_row(
            row,
            id_columns=self.index_columns,
            sequence_column=self.sequence_column,
        )

    def lookup_length(self, row: dict[str, str]) -> int:
        """Resolve residue length without materializing the full embedding."""
        return self.embedding_store.lookup_row_length(  # type: ignore[attr-defined]
            row,
            id_columns=self.index_columns,
            sequence_column=self.sequence_column,
        )

    def validate_row(self, row: dict[str, str]) -> str | None:
        for col in self.index_columns:
            identifier = clean_text(row.get(col))
            if identifier and self.embedding_store.has_id(identifier):
                return None
        sequence = normalize_sequence(row.get(self.sequence_column))
        if sequence and self.embedding_store.has_sequence(sequence):
            return None
        return (
            f"Missing residue embedding for row. "
            f"Tried columns: {list(self.index_columns) + [self.sequence_column]}. "
            f"sample_id={row.get('sample_id', '')}"
        )


LIGAND_FEATURES_PATH = Paths.data_processed_dir / "ligand_features/descriptastorus.npz"
_LIGAND_CACHE: dict[str, np.ndarray] | None = None


def _default_rdkit_featurize(smiles: str) -> np.ndarray:
    """Look up the 200 descriptastorus RDKit2DNormalized descriptors of a SMILES.

    Written by scripts/data_prep/build_ligand_features.py.
    """
    global _LIGAND_CACHE
    if _LIGAND_CACHE is None:
        if not LIGAND_FEATURES_PATH.exists():
            raise ValueError(
                f"Ligand feature cache {LIGAND_FEATURES_PATH} is missing; run "
                "scripts/data_prep/build_ligand_features.py."
            )
        with np.load(LIGAND_FEATURES_PATH) as data:
            _LIGAND_CACHE = dict(zip(data["smiles"].tolist(), data["features"]))
    try:
        return _LIGAND_CACHE[smiles]
    except KeyError:
        raise ValueError(
            f"SMILES '{smiles}' is not in {LIGAND_FEATURES_PATH.name}; rebuild it with "
            "scripts/data_prep/build_ligand_features.py."
        ) from None


@dataclass
class RDKitLigandFeatureBuilder:
    """Build fixed-size ligand features from SMILES strings."""

    ligand_column: str = "smiles"
    featurize_smiles: Callable[[str], np.ndarray] | None = None

    def build(self, row: dict[str, str]) -> torch.Tensor:
        smiles = clean_text(row.get(self.ligand_column))
        if not smiles:
            raise ValueError(
                f"Missing ligand value in column '{self.ligand_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        featurizer = self.featurize_smiles or _default_rdkit_featurize
        array = np.asarray(featurizer(smiles), dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                f"Ligand features must be 1D. Received shape {array.shape}."
            )
        return torch.from_numpy(array)

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except (ImportError, ValueError) as exc:
            return str(exc)
        return None


@dataclass(frozen=True)
class NumericFeatureBuilder:
    """Build a finite float vector from explicitly configured CSV columns."""

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("NumericFeatureBuilder requires at least one column.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("NumericFeatureBuilder columns must be unique.")

    def build(self, row: dict[str, str]) -> torch.Tensor:
        values: list[float] = []
        for column in self.columns:
            raw_value = clean_text(row.get(column))
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Missing or invalid numeric feature in column '{column}' for "
                    f"sample '{row.get('sample_id', '')}'."
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite numeric feature in column '{column}' for sample "
                    f"'{row.get('sample_id', '')}'."
                )
            values.append(value)
        return torch.tensor(values, dtype=torch.float32)

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except ValueError as exc:
            return str(exc)
        return None


@dataclass
class MutationDeltaEmbeddingBuilder:
    """Build mutant-minus-wildtype embedding deltas for mutation-effect tasks."""

    embedding_store: EmbeddingStore
    mutation_sequence_column: str = "protein_sequence"
    wildtype_sequence_column: str = "wildtype_sequence"

    def build(self, row: dict[str, str]) -> torch.Tensor:
        """Return the embedding delta (mutant minus wildtype) for a mutation row.

        Both ends are resolved by sequence and never by row identifier. A
        mutation row's ``uniprot_id`` names the protein, not the variant, so an
        identifier lookup would answer a missing mutant with the wildtype
        embedding and make the delta silently zero — a row carrying a real ddG
        target but no trace of the mutation it describes.
        """
        mutant_sequence = normalize_sequence(row.get(self.mutation_sequence_column))
        if not mutant_sequence:
            raise ValueError(
                f"Missing mutant sequence in column "
                f"'{self.mutation_sequence_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        wildtype_sequence = normalize_sequence(row.get(self.wildtype_sequence_column))
        if not wildtype_sequence:
            raise ValueError(
                f"Missing wildtype sequence in column "
                f"'{self.wildtype_sequence_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        mutant_embedding = self.embedding_store.lookup_sequence(mutant_sequence)
        wildtype_embedding = self.embedding_store.lookup_sequence(wildtype_sequence)

        if mutant_embedding.ndim == 2:
            expected_mutant_length = len(
                normalize_sequence(row.get(self.mutation_sequence_column))
            )
            expected_wildtype_length = len(
                normalize_sequence(row.get(self.wildtype_sequence_column))
            )
            if (
                expected_mutant_length
                and mutant_embedding.shape[0] != expected_mutant_length
            ):
                raise ValueError(
                    "Mutant embedding length mismatch for sample "
                    f"'{row.get('sample_id', '')}'."
                )
            if (
                expected_wildtype_length
                and wildtype_embedding.shape[0] != expected_wildtype_length
            ):
                raise ValueError(
                    "Wildtype embedding length mismatch for sample "
                    f"'{row.get('sample_id', '')}'."
                )
        if mutant_embedding.shape != wildtype_embedding.shape:
            raise ValueError(
                "Delta embedding requires equal shapes. "
                f"Mutant={tuple(mutant_embedding.shape)}, "
                f"wildtype={tuple(wildtype_embedding.shape)} for sample "
                f"'{row.get('sample_id', '')}'."
            )
        return (mutant_embedding.float() - wildtype_embedding.float()).cpu()

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Check that both mutant and wildtype embeddings resolve by sequence.

        Accepting a row on its identifier alone would let ``build`` fall back to
        the wildtype embedding for an absent mutant, so both sequences must be
        present in the store for the delta to carry any mutation signal.
        """
        store = self.embedding_store
        missing = []
        mut_seq = normalize_sequence(row.get(self.mutation_sequence_column))
        if not mut_seq or not store.has_sequence(mut_seq):
            missing.append(self.mutation_sequence_column)
        wt_seq = normalize_sequence(row.get(self.wildtype_sequence_column))
        if not wt_seq or not store.has_sequence(wt_seq):
            missing.append(self.wildtype_sequence_column)
        if missing:
            return (
                f"Missing embedding for {' and '.join(missing)}; a mutation "
                f"delta needs both sequences. "
                f"sample_id={row.get('sample_id', '')}"
            )
        return None
