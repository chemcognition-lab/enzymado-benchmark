from __future__ import annotations
"""Feature builders and embedding stores used by training and evaluation."""

import csv
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .splits import clean_text, normalize_identifier, normalize_sequence


class EmbeddingLookupError(KeyError):
    """Raised when a row cannot be matched to a stored residue embedding."""
    pass


def _h5_dataset_to_array(ds) -> np.ndarray:
    """Load an H5 dataset, preserving int16 storage for BF16-tagged datasets.

    BF16-tagged datasets stay as int16 raw bit-patterns so the collator can
    produce torch.bfloat16 tensors with zero extra copies.  Legacy float32
    datasets are returned as float32 unchanged.
    """
    arr = np.asarray(ds)
    attrs = getattr(ds, "attrs", {})
    if attrs.get("dtype", "") == "bfloat16":
        return arr  # int16 raw BF16 bits — collator will view as torch.bfloat16
    return np.asarray(arr, dtype=np.float32)


def _bf16_int16_to_float32(arr: np.ndarray) -> np.ndarray:
    """Decode BF16-as-int16 to float32 (needed only for arithmetic between embeddings)."""
    return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _ensure_2d_array(value: np.ndarray, key: str) -> np.ndarray:
    """Validate that residue embeddings keep the expected `(length, dim)` shape."""
    array = np.asarray(value)  # preserve dtype (int16 for BF16, float32 for legacy)
    if array.ndim != 2:
        raise ValueError(f"Embedding for key '{key}' must be 2D. Received shape {array.shape}.")
    return array


@lru_cache(maxsize=None)
def _load_sequence_index(sequence_index_path: str) -> dict[str, str]:
    """Load the normalized sequence-to-sequence-id index used for H5 subsets."""
    mapping: dict[str, str] = {}
    csv_path = Path(sequence_index_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence_id = clean_text(row.get("sequence_id"))
            sequence = normalize_sequence(row.get("protein_sequence"))
            if not sequence_id or not sequence:
                continue
            mapping[sequence] = sequence_id
    return mapping


@dataclass
class ResidueEmbeddingStore:
    """Lookup store for residue embeddings keyed by sequence and/or identifier."""
    by_sequence: dict[str, np.ndarray] = field(default_factory=dict)
    by_id: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize keys once so downstream lookups stay consistent."""
        self.by_sequence = {
            normalize_sequence(key): _ensure_2d_array(value, key)
            for key, value in self.by_sequence.items()
        }
        self.by_id = {
            clean_text(key): _ensure_2d_array(value, key)
            for key, value in self.by_id.items()
        }

    @classmethod
    def _load_h5_mapping(cls, file_path: Path | str) -> dict[str, np.ndarray]:
        """Load every top-level H5 dataset into memory as a float32 array."""
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ImportError(
                "h5py is required to load residue embeddings from .h5 files."
            ) from exc

        mapping: dict[str, np.ndarray] = {}
        with h5py.File(str(file_path), "r") as handle:
            for key in handle.keys():
                mapping[str(key)] = _h5_dataset_to_array(handle[key])
        return mapping

    @classmethod
    def from_h5(
        cls,
        file_path: Path | str,
        key_kind: str = "id",
    ) -> "ResidueEmbeddingStore":
        """Build a store from a single H5 file keyed by id or sequence."""
        mapping = cls._load_h5_mapping(file_path)
        if key_kind == "id":
            return cls(by_id=mapping)
        if key_kind == "sequence":
            return cls(by_sequence=mapping)
        raise ValueError("key_kind must be either 'id' or 'sequence'.")

    @classmethod
    def from_h5_files(
        cls,
        id_h5_path: Path | str | None = None,
        sequence_h5_path: Path | str | None = None,
    ) -> "ResidueEmbeddingStore":
        """Build a store from separate identifier-keyed and sequence-keyed H5 files."""
        if id_h5_path is None and sequence_h5_path is None:
            raise ValueError("Provide at least one of id_h5_path or sequence_h5_path.")

        by_id: dict[str, np.ndarray] = {}
        by_sequence: dict[str, np.ndarray] = {}
        if id_h5_path is not None:
            by_id = cls._load_h5_mapping(id_h5_path)
        if sequence_h5_path is not None:
            by_sequence = cls._load_h5_mapping(sequence_h5_path)
        return cls(by_id=by_id, by_sequence=by_sequence)

    @classmethod
    def from_h5_subset(
        cls,
        file_path: Path | str,
        needed_ids: set[str] | None = None,
        needed_sequences: set[str] | None = None,
        sequence_index_path: Path | str | None = None,
    ) -> "ResidueEmbeddingStore":
        """Load only the identifiers and sequences required for a dataset split."""
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ImportError(
                "h5py is required to load residue embeddings from .h5 files."
            ) from exc

        normalized_ids = {
            clean_text(value)
            for value in (needed_ids or set())
            if clean_text(value)
        }
        normalized_sequences = {
            normalize_sequence(value)
            for value in (needed_sequences or set())
            if normalize_sequence(value)
        }

        sequence_key_to_sequence = {
            sequence: sequence
            for sequence in normalized_sequences
        }
        if sequence_index_path is not None and normalized_sequences:
            sequence_to_id = _load_sequence_index(str(sequence_index_path))
            for sequence in normalized_sequences:
                sequence_id = sequence_to_id.get(sequence)
                if sequence_id:
                    sequence_key_to_sequence[sequence_id] = sequence

        by_id: dict[str, np.ndarray] = {}
        by_sequence: dict[str, np.ndarray] = {}
        with h5py.File(str(file_path), "r") as handle:
            for raw_key in handle.keys():
                key = clean_text(str(raw_key))
                array: np.ndarray | None = None

                if key in normalized_ids:
                    array = _h5_dataset_to_array(handle[raw_key])
                    by_id[key] = array

                sequence = sequence_key_to_sequence.get(key)
                if sequence is not None:
                    if array is None:
                        array = _h5_dataset_to_array(handle[raw_key])
                    by_sequence[sequence] = array

        return cls(by_id=by_id, by_sequence=by_sequence)

    def has_sequence(self, sequence: str) -> bool:
        """Return whether a normalized sequence key is present."""
        return normalize_sequence(sequence) in self.by_sequence

    def has_id(self, identifier: str) -> bool:
        """Return whether a normalized identifier key is present."""
        return clean_text(identifier) in self.by_id

    def lookup_sequence_length(self, sequence: str) -> int:
        """Fetch the residue length for a normalized sequence-keyed embedding."""
        normalized = normalize_sequence(sequence)
        try:
            return int(self.by_sequence[normalized].shape[0])
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing residue embedding for sequence '{normalized[:20]}...'"
            ) from exc

    def lookup_id_length(self, identifier: str) -> int:
        """Fetch the residue length for a normalized identifier-keyed embedding."""
        normalized = clean_text(identifier)
        try:
            return int(self.by_id[normalized].shape[0])
        except KeyError as exc:
            raise EmbeddingLookupError(
                f"Missing residue embedding for id '{normalized}'."
            ) from exc

    def lookup_sequence(self, sequence: str) -> np.ndarray:
        """Fetch an embedding by normalized sequence."""
        normalized = normalize_sequence(sequence)
        try:
            return self.by_sequence[normalized]
        except KeyError as exc:
            raise EmbeddingLookupError(f"Missing residue embedding for sequence '{normalized[:20]}...'") from exc

    def lookup_id(self, identifier: str) -> np.ndarray:
        """Fetch an embedding by normalized identifier."""
        normalized = clean_text(identifier)
        try:
            return self.by_id[normalized]
        except KeyError as exc:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{normalized}'.") from exc

    def lookup_row(
        self,
        row: dict[str, str],
        id_columns: tuple[str, ...] = ("uniprot_id", "dataset_id"),
        sequence_column: str = "protein_sequence",
    ) -> np.ndarray:
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

        tried = [column for column in (*id_columns, sequence_column) if column in row]
        raise EmbeddingLookupError(
            "Missing residue embedding for row. "
            f"Tried columns: {tried}. sample_id={row.get('sample_id', '')}"
        )

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

        tried = [column for column in (*id_columns, sequence_column) if column in row]
        raise EmbeddingLookupError(
            "Missing residue embedding for row. "
            f"Tried columns: {tried}. sample_id={row.get('sample_id', '')}"
        )


@dataclass
class ProteinResidueEmbeddingBuilder:
    """Build residue embeddings for rows in protein-centric tasks."""
    embedding_store: ResidueEmbeddingStore
    sequence_column: str = "protein_sequence"
    id_columns: tuple[str, ...] = ("uniprot_id", "dataset_id")

    def build(self, row: dict[str, str]) -> np.ndarray:
        """Materialize the residue embedding for one row."""
        return self.embedding_store.lookup_row(
            row,
            id_columns=self.id_columns,
            sequence_column=self.sequence_column,
        )

    def lookup_length(self, row: dict[str, str]) -> int:
        """Resolve the residue length without materializing the full embedding array."""
        return self.embedding_store.lookup_row_length(
            row,
            id_columns=self.id_columns,
            sequence_column=self.sequence_column,
        )

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Check lookup availability without forcing a full embedding read."""
        # Check existence only — avoids loading the full embedding during dataset construction.
        # Critical for lazy stores where build() triggers an H5 read per row.
        for col in self.id_columns:
            identifier = row.get(col, "").strip()
            if identifier and self.embedding_store.has_id(identifier):
                return None
        sequence = row.get(self.sequence_column, "").strip()
        if sequence and self.embedding_store.has_sequence(sequence):
            return None
        return (
            f"Missing residue embedding for row. "
            f"Tried columns: {list(self.id_columns) + [self.sequence_column]}. "
            f"sample_id={row.get('sample_id', '')}"
        )


def _default_rdkit_featurize(smiles: str) -> np.ndarray:
    """Build a compact ligand descriptor vector with RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen
        from rdkit.Chem import Descriptors
        from rdkit.Chem import Lipinski
        from rdkit.Chem import rdMolDescriptors
    except ModuleNotFoundError as exc:
        raise ImportError(
            "RDKit is required for ligand features unless a custom featurize_smiles "
            "callable is provided."
        ) from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Unable to parse SMILES '{smiles}'.")

    features = np.array(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            molecule.GetRingInfo().NumRings(),
            molecule.GetNumHeavyAtoms(),
        ],
        dtype=np.float32,
    )
    return features


@dataclass
class RDKitLigandFeatureBuilder:
    """Build fixed-size ligand features from SMILES strings."""
    ligand_column: str = "smiles"
    featurize_smiles: Callable[[str], np.ndarray] | None = None

    def build(self, row: dict[str, str]) -> np.ndarray:
        """Featurize the ligand referenced by a dataset row."""
        smiles = clean_text(row.get(self.ligand_column))
        if not smiles:
            raise ValueError(
                f"Missing ligand value in column '{self.ligand_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )

        featurizer = self.featurize_smiles or _default_rdkit_featurize
        array = np.asarray(featurizer(smiles), dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"Ligand features must be 1D. Received shape {array.shape}.")
        return array

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Validate a ligand row by attempting feature construction."""
        try:
            self.build(row)
        except (ImportError, ValueError) as exc:
            return str(exc)
        return None


@dataclass
class MutationDeltaEmbeddingBuilder:
    """Build mutant-minus-wildtype residue deltas for mutation-effect tasks."""
    embedding_store: ResidueEmbeddingStore
    mutation_sequence_column: str = "protein_sequence"
    wildtype_sequence_column: str = "wildtype_sequence"
    id_columns: tuple[str, ...] = ("uniprot_id", "dataset_id")

    def build(self, row: dict[str, str]) -> np.ndarray:
        """Return the residue-wise embedding delta for a mutation row."""
        mutant_embedding = self.embedding_store.lookup_row(
            row,
            id_columns=self.id_columns,
            sequence_column=self.mutation_sequence_column,
        )
        wildtype_sequence = normalize_identifier(
            self.wildtype_sequence_column,
            row.get(self.wildtype_sequence_column),
        )
        if not wildtype_sequence:
            raise ValueError(
                f"Missing wildtype sequence in column '{self.wildtype_sequence_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        wildtype_embedding = self.embedding_store.lookup_sequence(wildtype_sequence)

        expected_mutant_length = len(normalize_sequence(row.get(self.mutation_sequence_column)))
        expected_wildtype_length = len(normalize_sequence(row.get(self.wildtype_sequence_column)))
        if expected_mutant_length and mutant_embedding.shape[0] != expected_mutant_length:
            raise ValueError(
                f"Mutant embedding length mismatch for sample '{row.get('sample_id', '')}'."
            )
        if expected_wildtype_length and wildtype_embedding.shape[0] != expected_wildtype_length:
            raise ValueError(
                f"Wildtype embedding length mismatch for sample '{row.get('sample_id', '')}'."
            )
        if mutant_embedding.shape != wildtype_embedding.shape:
            raise ValueError(
                f"Delta embedding requires equal shapes. Mutant={mutant_embedding.shape}, "
                f"wildtype={wildtype_embedding.shape} for sample '{row.get('sample_id', '')}'."
            )
        # Arithmetic requires float32; decode BF16-as-int16 if necessary.
        if mutant_embedding.dtype == np.int16:
            mutant_embedding = _bf16_int16_to_float32(mutant_embedding)
        if wildtype_embedding.dtype == np.int16:
            wildtype_embedding = _bf16_int16_to_float32(wildtype_embedding)
        return (mutant_embedding - wildtype_embedding).astype(np.float32, copy=False)

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Check that both mutant and wildtype embeddings can be resolved."""
        # Check existence only — avoids loading the full embedding during dataset construction.
        # Critical for lazy stores where build() triggers an H5 read per row.
        store = self.embedding_store
        for col in self.id_columns:
            identifier = row.get(col, "").strip()
            if identifier and store.has_id(identifier):
                return None
        mut_seq = row.get(self.mutation_sequence_column, "").strip()
        wt_seq = row.get(self.wildtype_sequence_column, "").strip()
        if mut_seq and store.has_sequence(mut_seq) and wt_seq and store.has_sequence(wt_seq):
            return None
        return (
            f"Missing residue embedding for mutation or wildtype. "
            f"sample_id={row.get('sample_id', '')}"
        )
