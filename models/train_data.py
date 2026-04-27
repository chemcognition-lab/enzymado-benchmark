"""Data loading and feature-building helpers for the non-attention pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from data_loading import (
    BinaryLabelBuilder,
    FlatECLabelBuilder,
    HierarchicalECLabelBuilder,
    ResidueEmbeddingStore,
    ScalarRegressionLabelBuilder,
)
from data_loading.adapters import (
    MutationDeltaAdapter,
    ProteinGlobalAdapter,
    ProteinLigandAdapter,
    ProteinResidueAdapter,
)
from models.train_common import PROJECT_ROOT, log, resolve_sequence_index_path


def get_task_rows(config, split_name: str, split_variant: str) -> list[dict]:
    """Load CSV rows for a task split without touching any embeddings."""
    if split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        raise KeyError(
            f"Unknown split variant '{split_variant}' for task '{config.name}'. "
            f"Available variants: {available}"
        )

    adapter_cls = {
        "protein_global": ProteinGlobalAdapter,
        "protein_ligand_global": ProteinLigandAdapter,
        "mutation_delta_global": MutationDeltaAdapter,
        "protein_residue": ProteinResidueAdapter,
    }[config.batch_kind]

    return adapter_cls(
        dataset_path=config.dataset_paths[split_variant],
        sample_id_candidates=config.sample_id_candidates,
        split_name=split_name,
        split_column=config.split_column,
        split_path=None,
        split_id_candidates=config.split_id_candidates,
    ).load_rows()


def collect_needed_lookup_keys(rows: list[dict], config) -> tuple[set[str], set[str]]:
    """Return (id keys, sequence keys) needed to build a partial embedding store."""
    ids = set()
    sequences = set()
    for row in rows:
        for col in config.sample_id_candidates:
            val = row.get(col, "").strip()
            if val:
                ids.add(val)
        sequence = row.get(config.sequence_column, "").strip()
        if sequence:
            sequences.add(sequence)
        if config.wildtype_sequence_column:
            wildtype = row.get(config.wildtype_sequence_column, "").strip()
            if wildtype:
                sequences.add(wildtype)
    return ids, sequences


def _as_project_path(path: str | Path) -> Path:
    """Resolve repo-relative embedding paths while preserving absolute paths."""
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _path_with_data_segment(path: Path, old: str, new: str) -> Path | None:
    """Return a path with one directory component replaced, if present."""
    parts = list(path.parts)
    try:
        index = parts.index(old)
    except ValueError:
        return None
    parts[index] = new
    return Path(*parts)


def _candidate_safetensor_sources(path: str | Path) -> list[Path]:
    """Return plausible safetensors sources for a manifest path.

    Training manifests historically pointed at `.h5` files.  Current cluster
    embeddings are safetensors, either as a single mean-pooled file or as a
    directory of per-residue shards named `<model>_shards`.
    """
    full_path = _as_project_path(path)
    candidates: list[Path] = []

    if full_path.suffix == ".safetensors" or full_path.is_dir():
        candidates.append(full_path)
    if full_path.suffix == ".h5":
        candidates.append(full_path.with_suffix(".safetensors"))
        candidates.append(full_path.with_name(f"{full_path.stem}_shards"))

        swapped = _path_with_data_segment(full_path, "embeddings_mean_pooled", "embeddings")
        if swapped is not None:
            candidates.append(swapped.with_suffix(".safetensors"))
            candidates.append(swapped.with_name(f"{swapped.stem}_shards"))

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_embedding_source(path: str | Path, *, storage_format: str = "safetensors") -> Path:
    """Resolve an embedding manifest path to the on-disk source used for training."""
    if storage_format not in {"safetensors", "h5", "auto"}:
        raise ValueError("storage_format must be one of 'safetensors', 'h5', or 'auto'.")

    full_path = _as_project_path(path)
    if storage_format in {"safetensors", "auto"}:
        for candidate in _candidate_safetensor_sources(path):
            if candidate.is_file() or candidate.is_dir():
                return candidate
        if storage_format == "safetensors":
            candidates = ", ".join(str(candidate) for candidate in _candidate_safetensor_sources(path))
            raise FileNotFoundError(
                f"No safetensors embedding source found for '{path}'. Tried: {candidates}"
            )

    if full_path.exists():
        return full_path
    raise FileNotFoundError(f"Embedding source does not exist: {full_path}")


def _safetensor_files(source: Path) -> list[Path]:
    """Return one or more safetensors files from a source file or shard directory."""
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got '{source}'.")
        return [source]
    if source.is_dir():
        files = sorted(source.glob("*.safetensors"))
        if files:
            return files
    raise FileNotFoundError(f"No .safetensors files found in '{source}'.")


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert a safetensors torch tensor to NumPy, preserving BF16 compactly."""
    import torch

    if tensor.dtype == torch.bfloat16:
        return tensor.cpu().view(torch.int16).numpy()
    return tensor.cpu().numpy()


def _tensor_to_float32_numpy(tensor) -> np.ndarray:
    """Convert a safetensors tensor to a float32 NumPy array for arithmetic models."""
    return tensor.float().cpu().numpy().astype(np.float32, copy=False)


def _load_sequence_key_map(
    needed_sequences: set[str],
    available_keys: set[str],
) -> dict[str, str]:
    """Build normalized_sequence -> tensor key mapping via sequence index or direct key match."""
    from data_loading.splits import clean_text, normalize_sequence as _ns

    normalized_sequences = {_ns(s) for s in needed_sequences if _ns(s)}
    seq_to_key: dict[str, str] = {}

    seq_index_path = resolve_sequence_index_path()
    if seq_index_path is not None and normalized_sequences:
        import csv

        with open(seq_index_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = _ns(row.get("protein_sequence", ""))
                sid = clean_text(row.get("sequence_id", ""))
                if seq in normalized_sequences and sid in available_keys:
                    seq_to_key[seq] = sid

    for seq in normalized_sequences:
        if seq not in seq_to_key and seq in available_keys:
            seq_to_key[seq] = seq
    return seq_to_key


def _index_safetensors(source: Path) -> dict[str, Path]:
    """Index tensor keys to the safetensors file that stores them."""
    from safetensors import safe_open
    from data_loading.splits import clean_text

    key_to_file: dict[str, Path] = {}
    for file_path in _safetensor_files(source):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = clean_text(str(raw_key))
                if key and key not in key_to_file:
                    key_to_file[key] = file_path
    return key_to_file


class LazyResidueEmbeddingStore(ResidueEmbeddingStore):
    """Reads embeddings from H5 on-demand instead of loading all into RAM."""

    def __init__(
        self,
        h5_path: str | Path,
        available_ids: set[str],
        seq_to_h5key: dict[str, str],
    ):
        super().__init__(by_id={}, by_sequence={})
        self._h5_path = str(h5_path)
        self._h5 = None
        self._available_ids: set[str] = available_ids
        self._seq_to_h5key: dict[str, str] = seq_to_h5key

    def _open(self):
        """Open the H5 file lazily on first lookup."""
        import h5py

        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self._h5_path, "r")

    def has_id(self, identifier: str) -> bool:
        """Check id availability without loading any embedding arrays."""
        from data_loading.splits import clean_text

        return clean_text(identifier) in self._available_ids

    def has_sequence(self, sequence: str) -> bool:
        """Check sequence availability without loading any embedding arrays."""
        from data_loading.splits import normalize_sequence as _ns

        return _ns(sequence) in self._seq_to_h5key

    def lookup_id(self, identifier: str) -> np.ndarray:
        """Read one identifier-keyed embedding directly from the H5 file."""
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import clean_text

        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        self._open()
        try:
            return np.asarray(self._h5[key], dtype=np.float32)
        except KeyError:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")

    def lookup_id_length(self, identifier: str) -> int:
        """Read one identifier-keyed embedding shape without materializing its array."""
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import clean_text

        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        self._open()
        try:
            shape = self._h5[key].shape
        except KeyError:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        if len(shape) != 2:
            raise ValueError(f"Embedding for key '{key}' must be 2D. Received shape {shape}.")
        return int(shape[0])

    def lookup_sequence(self, sequence: str) -> np.ndarray:
        """Read one sequence-keyed embedding directly from the H5 file."""
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import normalize_sequence as _ns

        norm_seq = _ns(sequence)
        h5_key = self._seq_to_h5key.get(norm_seq)
        if h5_key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        self._open()
        try:
            return np.asarray(self._h5[h5_key], dtype=np.float32)
        except KeyError:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")

    def lookup_sequence_length(self, sequence: str) -> int:
        """Read one sequence-keyed embedding shape without materializing its array."""
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import normalize_sequence as _ns

        norm_seq = _ns(sequence)
        h5_key = self._seq_to_h5key.get(norm_seq)
        if h5_key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        self._open()
        try:
            shape = self._h5[h5_key].shape
        except KeyError:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        if len(shape) != 2:
            raise ValueError(f"Embedding for key '{h5_key}' must be 2D. Received shape {shape}.")
        return int(shape[0])

    def __del__(self):
        """Close the H5 handle when the store is garbage collected."""
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class LazySafetensorsResidueEmbeddingStore(ResidueEmbeddingStore):
    """Reads residue embeddings from safetensors shards on-demand."""

    def __init__(
        self,
        key_to_file: dict[str, Path],
        available_ids: set[str],
        seq_to_key: dict[str, str],
    ):
        super().__init__(by_id={}, by_sequence={})
        self._key_to_file = key_to_file
        self._available_ids = available_ids
        self._seq_to_h5key = seq_to_key  # Compatibility with existing logging/tests.
        self._seq_to_key = seq_to_key
        self._handles = {}

    def _handle_for_key(self, key: str):
        from safetensors import safe_open

        file_path = self._key_to_file[key]
        handle = self._handles.get(file_path)
        if handle is None:
            handle = safe_open(str(file_path), framework="pt", device="cpu")
            self._handles[file_path] = handle
        return handle

    def _lookup_key(self, key: str) -> np.ndarray:
        from data_loading.features import EmbeddingLookupError

        if key not in self._key_to_file:
            raise EmbeddingLookupError(f"Missing residue embedding for key '{key}'.")
        tensor = self._handle_for_key(key).get_tensor(key)
        array = _tensor_to_numpy(tensor)
        if array.ndim != 2:
            raise ValueError(f"Embedding for key '{key}' must be 2D. Received shape {array.shape}.")
        return array

    def has_id(self, identifier: str) -> bool:
        from data_loading.splits import clean_text

        return clean_text(identifier) in self._available_ids

    def has_sequence(self, sequence: str) -> bool:
        from data_loading.splits import normalize_sequence as _ns

        return _ns(sequence) in self._seq_to_key

    def lookup_id(self, identifier: str) -> np.ndarray:
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import clean_text

        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        return self._lookup_key(key)

    def lookup_id_length(self, identifier: str) -> int:
        return int(self.lookup_id(identifier).shape[0])

    def lookup_sequence(self, sequence: str) -> np.ndarray:
        from data_loading.features import EmbeddingLookupError
        from data_loading.splits import normalize_sequence as _ns

        norm_seq = _ns(sequence)
        key = self._seq_to_key.get(norm_seq)
        if key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        return self._lookup_key(key)

    def lookup_sequence_length(self, sequence: str) -> int:
        return int(self.lookup_sequence(sequence).shape[0])


def _build_seq_to_h5key(
    h5_path: str | Path,
    needed_sequences: set[str],
) -> dict[str, str]:
    """Build normalized_sequence → H5_key mapping via sequence index or direct key match."""
    import h5py
    from data_loading.splits import clean_text, normalize_sequence as _ns

    normalized_sequences = {_ns(s) for s in needed_sequences if _ns(s)}
    seq_to_h5key: dict[str, str] = {}

    seq_index_path = resolve_sequence_index_path()
    if seq_index_path is not None and normalized_sequences:
        import csv

        with open(seq_index_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = _ns(row.get("protein_sequence", ""))
                sid = clean_text(row.get("sequence_id", ""))
                if seq in normalized_sequences and sid:
                    seq_to_h5key[seq] = sid

    with h5py.File(str(h5_path), "r") as handle:
        for seq in normalized_sequences:
            if seq not in seq_to_h5key and seq in handle:
                seq_to_h5key[seq] = seq

    return seq_to_h5key


def load_lazy_embedding_store(
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> LazyResidueEmbeddingStore:
    """Index only the H5 keys needed for a lazily loaded training run."""
    import h5py
    from data_loading.splits import clean_text

    full_path = PROJECT_ROOT / h5_path

    with h5py.File(str(full_path), "r") as handle:
        h5_keys = {clean_text(k) for k in handle.keys()}

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & h5_keys
    seq_to_h5key = {
        seq: key
        for seq, key in _build_seq_to_h5key(full_path, needed_sequences).items()
        if key in h5_keys
    }

    return LazyResidueEmbeddingStore(
        h5_path=full_path,
        available_ids=normalized_ids,
        seq_to_h5key=seq_to_h5key,
    )


def load_lazy_safetensors_embedding_store(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    storage_format: str = "safetensors",
) -> LazySafetensorsResidueEmbeddingStore:
    """Index safetensors keys needed for lazy residue-embedding training."""
    from data_loading.splits import clean_text

    source = resolve_embedding_source(source_path, storage_format=storage_format)
    key_to_file = _index_safetensors(source)
    available_keys = set(key_to_file)
    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & available_keys
    seq_to_key = _load_sequence_key_map(needed_sequences, available_keys)

    return LazySafetensorsResidueEmbeddingStore(
        key_to_file=key_to_file,
        available_ids=normalized_ids,
        seq_to_key=seq_to_key,
    )


def load_embedding_store_subset(
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> ResidueEmbeddingStore:
    """Materialize only the embeddings required for a specific task split."""
    sequence_index_path = resolve_sequence_index_path()
    return ResidueEmbeddingStore.from_h5_subset(
        PROJECT_ROOT / h5_path,
        needed_ids=needed_ids,
        needed_sequences=needed_sequences,
        sequence_index_path=sequence_index_path,
    )


def load_safetensors_embedding_store_subset(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    storage_format: str = "safetensors",
) -> ResidueEmbeddingStore:
    """Materialize only the required residue embeddings from safetensors."""
    from data_loading.splits import clean_text

    source = resolve_embedding_source(source_path, storage_format=storage_format)
    key_to_file = _index_safetensors(source)
    available_keys = set(key_to_file)
    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & available_keys
    seq_to_key = _load_sequence_key_map(needed_sequences, available_keys)

    lazy_store = LazySafetensorsResidueEmbeddingStore(
        key_to_file=key_to_file,
        available_ids=normalized_ids,
        seq_to_key=seq_to_key,
    )
    by_id = {key: lazy_store.lookup_id(key) for key in normalized_ids}
    by_sequence = {seq: lazy_store.lookup_sequence(seq) for seq in seq_to_key}
    return ResidueEmbeddingStore(by_id=by_id, by_sequence=by_sequence)


def load_pooled_safetensors_subset(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    storage_format: str = "safetensors",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load needed safetensors embeddings and mean-pool per-residue tensors if needed."""
    from safetensors import safe_open
    from data_loading.splits import clean_text, normalize_sequence as _norm_seq

    source = resolve_embedding_source(source_path, storage_format=storage_format)
    key_to_file = _index_safetensors(source)
    available_keys = set(key_to_file)

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & available_keys
    normalized_sequences = {_norm_seq(v) for v in needed_sequences if _norm_seq(v)}
    seq_to_key = _load_sequence_key_map(normalized_sequences, available_keys)

    keys_to_load: dict[str, tuple[str | None, str | None]] = {}
    for id_key in normalized_ids:
        entry = keys_to_load.get(id_key, (None, None))
        keys_to_load[id_key] = (id_key, entry[1])
    for seq, tensor_key in seq_to_key.items():
        entry = keys_to_load.get(tensor_key, (None, None))
        keys_to_load[tensor_key] = (entry[0], seq)

    log(f"      seq_index mapped {len(seq_to_key)}/{len(normalized_sequences)} seqs -> safetensors keys")
    log(
        f"      will load {len(keys_to_load)} unique safetensors keys "
        f"({len(normalized_ids)} by id, {len(seq_to_key)} by seq)"
    )

    by_id: dict[str, np.ndarray] = {}
    by_seq: dict[str, np.ndarray] = {}
    total = len(keys_to_load)
    log_interval = max(1, total // 20)
    t_load_start = time.time()
    handles = {}

    log(f"      Safetensors source ready. Reading {total:,} embeddings from {source}...")
    for i, (tensor_key, (id_key, seq)) in enumerate(keys_to_load.items(), 1):
        file_path = key_to_file[tensor_key]
        handle = handles.get(file_path)
        if handle is None:
            handle = safe_open(str(file_path), framework="pt", device="cpu")
            handles[file_path] = handle
        arr = _tensor_to_float32_numpy(handle.get_tensor(tensor_key))
        if arr.ndim == 2:
            pooled = arr.mean(axis=0)
        elif arr.ndim == 1:
            pooled = arr
        else:
            raise ValueError(
                f"Expected 1D or 2D embedding array for key '{tensor_key}', got shape {arr.shape}."
            )
        if id_key is not None:
            by_id[id_key] = pooled
        if seq is not None:
            by_seq[seq] = pooled
        if i % log_interval == 0 or i == total:
            elapsed = time.time() - t_load_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            log(
                f"      ... {i:,}/{total:,} ({100*i/total:.0f}%)  "
                f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s"
            )
    return by_id, by_seq


def load_pooled_embedding_subset(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
    *,
    storage_format: str = "safetensors",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load pooled embeddings from the configured storage format."""
    if storage_format == "h5":
        return load_pooled_h5_subset(source_path, needed_ids, needed_sequences)
    return load_pooled_safetensors_subset(
        source_path,
        needed_ids,
        needed_sequences,
        storage_format=storage_format,
    )


def load_pooled_h5_subset(
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load only the needed H5 embeddings by key and mean-pool each one immediately."""
    import h5py
    from data_loading.splits import clean_text, normalize_sequence as _norm_seq

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)}
    normalized_sequences = {_norm_seq(v) for v in needed_sequences if _norm_seq(v)}
    seq_to_h5key = _build_seq_to_h5key(PROJECT_ROOT / h5_path, normalized_sequences)

    with h5py.File(str(PROJECT_ROOT / h5_path), "r") as handle:
        h5_keys = {clean_text(key) for key in handle.keys()}

    normalized_ids &= h5_keys
    seq_to_h5key = {
        seq: key
        for seq, key in seq_to_h5key.items()
        if key in h5_keys
    }

    keys_to_load: dict[str, tuple[str | None, str | None]] = {}
    for id_key in normalized_ids:
        entry = keys_to_load.get(id_key, (None, None))
        keys_to_load[id_key] = (id_key, entry[1])
    for seq, h5_key in seq_to_h5key.items():
        entry = keys_to_load.get(h5_key, (None, None))
        keys_to_load[h5_key] = (entry[0], seq)

    log(f"      seq_index mapped {len(seq_to_h5key)}/{len(normalized_sequences)} seqs → H5 keys")
    log(
        f"      will load {len(keys_to_load)} unique H5 keys "
        f"({len(normalized_ids)} by id, {len(seq_to_h5key)} by seq)"
    )

    by_id: dict[str, np.ndarray] = {}
    by_seq: dict[str, np.ndarray] = {}
    missing = 0
    total = len(keys_to_load)
    log_interval = max(1, total // 20)
    t_load_start = time.time()
    with h5py.File(str(PROJECT_ROOT / h5_path), "r") as handle:
        log(f"      H5 file opened. Reading {total:,} embeddings...")
        for i, (h5_key, (id_key, seq)) in enumerate(keys_to_load.items(), 1):
            if h5_key not in handle:
                missing += 1
                continue
            arr = np.asarray(handle[h5_key], dtype=np.float32)
            if arr.ndim == 2:
                pooled = arr.mean(axis=0)
            elif arr.ndim == 1:
                pooled = arr
            else:
                raise ValueError(
                    f"Expected 1D or 2D embedding array for key '{h5_key}', got shape {arr.shape}."
                )
            if id_key is not None:
                by_id[id_key] = pooled
            if seq is not None:
                by_seq[seq] = pooled
            if i % log_interval == 0 or i == total:
                elapsed = time.time() - t_load_start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                log(
                    f"      ... {i:,}/{total:,} ({100*i/total:.0f}%)  "
                    f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s"
                )

    if missing:
        log(f"      {missing} keys not found in H5 (will be skipped)")
    return by_id, by_seq


def build_Xy_ligand_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled protein-plus-ligand features without constructing dataset objects."""
    from data_loading.features import _default_rdkit_featurize

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    skipped = 0

    for row in rows:
        emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        if emb is None:
            skipped += 1
            continue

        smiles = (row.get(config.ligand_column) or "").strip()
        try:
            ligand_feat = _default_rdkit_featurize(smiles)
        except (ValueError, ImportError):
            skipped += 1
            continue

        target_str = (row.get(config.target_column) or "").strip()
        if not target_str:
            skipped += 1
            continue

        X_list.append(np.concatenate([emb, ligand_feat]))
        y_list.append(float(target_str))

    if skipped:
        print(f"      Skipped {skipped} rows (missing embedding, ligand, or target)")
    if not X_list:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def _lookup_pooled_embedding(
    row: dict,
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
) -> np.ndarray | None:
    """Resolve a pooled embedding from sequence first, then fallback identifiers."""
    from data_loading.splits import clean_text, normalize_sequence as _norm_seq

    seq = _norm_seq(row.get(config.sequence_column, ""))
    if seq:
        emb = by_seq.get(seq)
        if emb is not None:
            return emb

    for col in config.sample_id_candidates:
        key = clean_text(row.get(col, ""))
        if key and key in by_id:
            return by_id[key]
    return None


def build_multilabel_ec_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
    label_builder: HierarchicalECLabelBuilder,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build pooled EC features plus per-level targets and masks."""
    level_names = label_builder.level_names()
    X_list: list[np.ndarray] = []
    targets = {level_name: [] for level_name in level_names}
    target_masks = {level_name: [] for level_name in level_names}
    skipped = 0

    for row in rows:
        emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        if emb is None:
            skipped += 1
            continue

        label_payload = label_builder.build(row)
        X_list.append(emb)
        for level_name in level_names:
            targets[level_name].append(
                np.asarray(label_payload["target"][level_name], dtype=np.float32)
            )
            target_masks[level_name].append(
                bool(label_payload["target_mask"][level_name])
            )

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings)")

    if not X_list:
        empty_targets = {
            level_name: np.zeros(
                (0, len(label_builder.vocabulary_by_level[level_name])),
                dtype=np.float32,
            )
            for level_name in level_names
        }
        empty_target_masks = {
            level_name: np.zeros((0,), dtype=bool)
            for level_name in level_names
        }
        return np.empty((0, 0), dtype=np.float32), empty_targets, empty_target_masks

    return (
        np.stack(X_list).astype(np.float32, copy=False),
        {
            level_name: np.stack(level_targets).astype(np.float32, copy=False)
            for level_name, level_targets in targets.items()
        },
        {
            level_name: np.asarray(level_mask, dtype=bool)
            for level_name, level_mask in target_masks.items()
        },
    )


def build_flat_multilabel_ec_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
    label_builder: FlatECLabelBuilder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pooled EC features using flattened targets for XGBoost models."""
    label_builder._ensure_flat_index()
    X_list: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    skipped = 0

    for row in rows:
        emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        if emb is None:
            skipped += 1
            continue

        label_payload = label_builder.build(row)
        X_list.append(emb)
        targets.append(np.asarray(label_payload["target"], dtype=np.float32))
        target_masks.append(np.asarray(label_payload["target_mask"], dtype=bool))

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings)")

    if not X_list:
        empty_dim = len(label_builder.flat_vocabulary or ())
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, empty_dim), dtype=np.float32),
            np.empty((0, empty_dim), dtype=bool),
        )

    return (
        np.stack(X_list).astype(np.float32, copy=False),
        np.stack(targets).astype(np.float32, copy=False),
        np.stack(target_masks).astype(bool, copy=False),
    )


def build_protein_global_Xy_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled protein-only features for scalar or binary tasks."""
    if config.label_kind == "scalar_regression":
        label_builder = ScalarRegressionLabelBuilder(target_column=str(config.target_column))
    elif config.label_kind == "binary":
        label_builder = BinaryLabelBuilder(target_column=str(config.target_column))
    else:
        raise ValueError(
            f"build_protein_global_Xy_direct does not support label_kind '{config.label_kind}'."
        )

    X_list: list[np.ndarray] = []
    y_list: list[np.float32] = []
    skipped = 0

    for row in rows:
        emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        if emb is None:
            skipped += 1
            continue

        try:
            target = label_builder.build(row)
        except ValueError:
            skipped += 1
            continue

        X_list.append(emb)
        y_list.append(np.float32(target))

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings or invalid targets)")

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return (
        np.stack(X_list).astype(np.float32, copy=False),
        np.asarray(y_list, dtype=np.float32),
    )


def build_mutation_delta_Xy_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled mutant-minus-wildtype features for mutation-effect tasks."""
    if config.label_kind == "scalar_regression":
        label_builder = ScalarRegressionLabelBuilder(target_column=str(config.target_column))
    elif config.label_kind == "binary":
        label_builder = BinaryLabelBuilder(target_column=str(config.target_column))
    else:
        raise ValueError(
            f"build_mutation_delta_Xy_direct does not support label_kind '{config.label_kind}'."
        )

    if not config.wildtype_sequence_column:
        raise ValueError("Mutation-delta tasks require 'wildtype_sequence_column'.")

    from data_loading.splits import normalize_sequence as _norm_seq

    X_list: list[np.ndarray] = []
    y_list: list[np.float32] = []
    skipped = 0

    for row in rows:
        mutant_emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        wildtype_seq = _norm_seq(row.get(config.wildtype_sequence_column, ""))
        wildtype_emb = by_seq.get(wildtype_seq) if wildtype_seq else None
        if mutant_emb is None or wildtype_emb is None:
            skipped += 1
            continue
        if mutant_emb.shape != wildtype_emb.shape:
            skipped += 1
            continue

        try:
            target = label_builder.build(row)
        except ValueError:
            skipped += 1
            continue

        X_list.append((mutant_emb - wildtype_emb).astype(np.float32, copy=False))
        y_list.append(np.float32(target))

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings or invalid targets)")

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return (
        np.stack(X_list).astype(np.float32, copy=False),
        np.asarray(y_list, dtype=np.float32),
    )


def build_Xy_pooled(bundle, batch_kind: str):
    """Build mean-pooled X (N, emb_dim) and y by iterating samples one at a time."""
    X_list, y_list = [], []
    for i in range(len(bundle.dataset)):
        sample = bundle.dataset[i]
        if batch_kind == "protein_global":
            x = sample["protein_emb"].mean(axis=0)
        elif batch_kind == "mutation_delta_global":
            x = sample["protein_delta_emb"].mean(axis=0)
        elif batch_kind == "protein_ligand_global":
            x = np.concatenate([
                sample["protein_emb"].mean(axis=0),
                sample["ligand_feat"],
            ])
        X_list.append(x)
        y_list.append(sample["target"])
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def get_dataloader(bundle, batch_size: int = 32, collate_fn=None) -> DataLoader:
    """Build a deterministic DataLoader for one dataset bundle."""
    return DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        collate_fn=collate_fn if collate_fn is not None else bundle.collator,
        shuffle=False,
        num_workers=0,
    )


def _pad_1d_array_list(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    """Pad variable-length 1D arrays into a rectangular batch."""
    if not arrays:
        return np.zeros((0, 0), dtype=dtype)
    max_length = max(array.shape[0] for array in arrays)
    batch = np.zeros((len(arrays), max_length), dtype=dtype)
    for index, array in enumerate(arrays):
        batch[index, : array.shape[0]] = array.astype(dtype, copy=False)
    return batch


def _build_label_only_payload(bundle):
    """Build targets from validated rows without materializing feature embeddings."""
    label_kind = bundle.config.label_kind
    rows = bundle.dataset.rows
    label_builder = bundle.label_builder

    if label_kind in {"scalar_regression", "binary"}:
        return (
            np.asarray(
                [label_builder.build(row) for row in rows],
                dtype=np.float32,
            ),
            None,
        )

    if label_kind == "multilabel_ec":
        level_names = label_builder.level_names()
        label_payloads = [label_builder.build(row) for row in rows]
        targets = {
            level_name: np.stack(
                [
                    np.asarray(payload["target"][level_name], dtype=np.float32)
                    for payload in label_payloads
                ]
            ).astype(np.float32, copy=False)
            if label_payloads
            else np.zeros(
                (0, len(label_builder.vocabulary_by_level[level_name])),
                dtype=np.float32,
            )
            for level_name in level_names
        }
        target_masks = {
            level_name: np.asarray(
                [
                    bool(payload["target_mask"][level_name])
                    for payload in label_payloads
                ],
                dtype=bool,
            )
            for level_name in level_names
        }
        return targets, target_masks

    if label_kind == "residue_binary_vector":
        label_payloads = [label_builder.build(row) for row in rows]
        targets = _pad_1d_array_list(
            [
                np.asarray(target_per_residue, dtype=np.float32)
                for target_per_residue, _ in label_payloads
            ],
            dtype=np.float32,
        )
        target_masks = _pad_1d_array_list(
            [
                np.asarray(target_mask, dtype=np.int8)
                for _, target_mask in label_payloads
            ],
            dtype=np.int8,
        ).astype(bool, copy=False)
        return targets, target_masks

    raise ValueError(f"Unsupported label_kind '{label_kind}'.")


def get_target_payload(bundle):
    """Collect task targets and optional masks without reloading embeddings."""
    return _build_label_only_payload(bundle)


__all__ = [
    "LazyResidueEmbeddingStore",
    "LazySafetensorsResidueEmbeddingStore",
    "build_Xy_ligand_direct",
    "build_Xy_pooled",
    "build_flat_multilabel_ec_direct",
    "build_multilabel_ec_direct",
    "build_mutation_delta_Xy_direct",
    "build_protein_global_Xy_direct",
    "collect_needed_lookup_keys",
    "get_dataloader",
    "get_target_payload",
    "get_task_rows",
    "load_embedding_store_subset",
    "load_lazy_embedding_store",
    "load_lazy_safetensors_embedding_store",
    "load_pooled_h5_subset",
    "load_pooled_embedding_subset",
    "load_pooled_safetensors_subset",
    "load_safetensors_embedding_store_subset",
    "resolve_embedding_source",
]
