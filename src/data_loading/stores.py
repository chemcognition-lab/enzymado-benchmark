"""Embedding store loaders: resolves safetensors paths, builds residue and pooled stores."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loading.task_csv import TaskCSVLoader, clean_text, normalize_sequence
from data_loading.features import PooledEmbeddingStore, ResidueEmbeddingStore
from data_loading.labels import FlatECLabelBuilder, HierarchicalECLabelBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PREFERRED_SEQ_INDEX = PROJECT_ROOT / "data/processed/protein_sequences_unique.csv"
_LEGACY_SEQ_INDEX = PROJECT_ROOT / "data/processed/protein_sequences_unique_no_fluorescence.csv"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_sequence_index_path() -> Path | None:
    if _PREFERRED_SEQ_INDEX.exists():
        return _PREFERRED_SEQ_INDEX
    if _LEGACY_SEQ_INDEX.exists():
        return _LEGACY_SEQ_INDEX
    return None


def get_task_rows(config, split_name: str, split_variant: str) -> list[dict]:
    """Load CSV rows for a task split without touching any embeddings."""
    if split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        raise KeyError(
            f"Unknown split variant '{split_variant}' for task '{config.name}'. "
            f"Available variants: {available}"
        )

    return TaskCSVLoader(
        dataset_path=config.dataset_paths[split_variant],
        index_columns=config.index_columns,
        split_name=split_name,
        split_column=config.split_column,
        sequence_column=config.sequence_column,
    ).load_rows()


def collect_needed_lookup_keys(rows: list[dict], config) -> tuple[set[str], set[str]]:
    """Return (id keys, sequence keys) needed to build a partial embedding store."""
    ids = set()
    sequences = set()
    for row in rows:
        for col in config.index_columns:
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


def _project_data_equivalent(path: Path) -> Path | None:
    """Return the repo-local equivalent for an absolute path containing a data segment."""
    parts = list(path.parts)
    try:
        data_index = parts.index("data")
    except ValueError:
        return None
    return PROJECT_ROOT / Path(*parts[data_index:])


def _candidate_safetensor_sources(path: str | Path) -> list[Path]:
    """Return plausible safetensors sources for a manifest path.

    Manifests may reference a single mean-pooled file or a per-residue shard
    directory named ``<model>_shards``.
    """
    full_path = _as_project_path(path)
    candidates: list[Path] = []
    base_paths = [full_path]
    project_equivalent = _project_data_equivalent(full_path)
    if project_equivalent is not None:
        base_paths.append(project_equivalent)

    for base_path in base_paths:
        if base_path.suffix == ".safetensors" or base_path.is_dir():
            candidates.append(base_path)

    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_embedding_source(path: str | Path) -> Path:
    """Resolve an embedding manifest path to the on-disk safetensors source."""
    for candidate in _candidate_safetensor_sources(path):
        if candidate.is_file() or candidate.is_dir():
            return candidate
    candidates = ", ".join(str(candidate) for candidate in _candidate_safetensor_sources(path))
    raise FileNotFoundError(
        f"No safetensors embedding source found for '{path}'. Tried: {candidates}"
    )


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


def _load_sequence_key_map(
    needed_sequences: set[str],
    available_keys: set[str],
) -> dict[str, str]:
    """Build normalized_sequence -> tensor key mapping via sequence index or direct key match."""
    normalized_sequences = {normalize_sequence(s) for s in needed_sequences if normalize_sequence(s)}
    seq_to_key: dict[str, str] = {}

    seq_index_path = resolve_sequence_index_path()
    if seq_index_path is not None and normalized_sequences:
        import csv

        with open(seq_index_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = normalize_sequence(row.get("protein_sequence", ""))
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

    key_to_file: dict[str, Path] = {}
    for file_path in _safetensor_files(source):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = clean_text(str(raw_key))
                if key and key not in key_to_file:
                    key_to_file[key] = file_path
    return key_to_file


class LazySafetensorsResidueEmbeddingStore(ResidueEmbeddingStore):
    """Reads residue embeddings from safetensors shards on-demand as torch.bfloat16 tensors."""

    def __init__(
        self,
        key_to_file: dict[str, Path],
        available_ids: set[str],
        seq_to_key: dict[str, str],
    ):
        super().__init__(by_id={}, by_sequence={})
        self._key_to_file = key_to_file
        self._available_ids = available_ids
        self._seq_to_key = seq_to_key
        self._handles = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}  # safe_open handles can't be pickled; workers re-open lazily
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _handle_for_key(self, key: str):
        from safetensors import safe_open

        file_path = self._key_to_file[key]
        handle = self._handles.get(file_path)
        if handle is None:
            handle = safe_open(str(file_path), framework="pt", device="cpu")
            self._handles[file_path] = handle
        return handle

    def _lookup_key(self, key: str) -> torch.Tensor:
        from data_loading.features import EmbeddingLookupError

        if key not in self._key_to_file:
            raise EmbeddingLookupError(f"Missing residue embedding for key '{key}'.")
        tensor = self._handle_for_key(key).get_tensor(key)
        if tensor.ndim != 2:
            raise ValueError(f"Embedding for key '{key}' must be 2D. Received shape {tuple(tensor.shape)}.")
        return tensor

    def has_id(self, identifier: str) -> bool:
        return clean_text(identifier) in self._available_ids

    def has_sequence(self, sequence: str) -> bool:
        return normalize_sequence(sequence) in self._seq_to_key

    def lookup_id(self, identifier: str) -> torch.Tensor:
        from data_loading.features import EmbeddingLookupError

        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        return self._lookup_key(key)

    def _key_shape(self, key: str) -> int:
        """Read sequence length from safetensors header without loading tensor data."""
        return self._handle_for_key(key).get_slice(key).get_shape()[0]

    def lookup_id_length(self, identifier: str) -> int:
        from data_loading.features import EmbeddingLookupError

        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        return self._key_shape(key)

    def lookup_sequence(self, sequence: str) -> torch.Tensor:
        from data_loading.features import EmbeddingLookupError

        norm_seq = normalize_sequence(sequence)
        key = self._seq_to_key.get(norm_seq)
        if key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        return self._lookup_key(key)

    def lookup_sequence_length(self, sequence: str) -> int:
        from data_loading.features import EmbeddingLookupError

        norm_seq = normalize_sequence(sequence)
        key = self._seq_to_key.get(norm_seq)
        if key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        return self._key_shape(key)


def load_lazy_safetensors_embedding_store(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> LazySafetensorsResidueEmbeddingStore:
    """Index safetensors keys needed for lazy residue-embedding training."""
    source = resolve_embedding_source(source_path)
    key_to_file = _index_safetensors(source)
    available_keys = set(key_to_file)
    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & available_keys
    seq_to_key = _load_sequence_key_map(needed_sequences, available_keys)

    return LazySafetensorsResidueEmbeddingStore(
        key_to_file=key_to_file,
        available_ids=normalized_ids,
        seq_to_key=seq_to_key,
    )


def load_safetensors_embedding_store_subset(
    source_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> ResidueEmbeddingStore:
    """Materialize only the required residue embeddings from safetensors."""
    source = resolve_embedding_source(source_path)
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
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load needed safetensors embeddings and mean-pool per-residue tensors if needed."""
    from safetensors import safe_open

    source = resolve_embedding_source(source_path)
    key_to_file = _index_safetensors(source)
    available_keys = set(key_to_file)

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & available_keys
    normalized_sequences = {normalize_sequence(v) for v in needed_sequences if normalize_sequence(v)}
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

    by_id: dict[str, torch.Tensor] = {}
    by_seq: dict[str, torch.Tensor] = {}
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
        t = handle.get_tensor(tensor_key).float()
        if t.ndim == 2:
            pooled = t.mean(dim=0)
        elif t.ndim == 1:
            pooled = t
        else:
            raise ValueError(
                f"Expected 1D or 2D embedding tensor for key '{tensor_key}', got shape {tuple(t.shape)}."
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
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load mean-pooled embeddings from safetensors."""
    return load_pooled_safetensors_subset(source_path, needed_ids, needed_sequences)


def build_pooled_embedding_store(
    by_id: dict[str, torch.Tensor],
    by_seq: dict[str, torch.Tensor],
) -> PooledEmbeddingStore:
    """Wrap pre-loaded pooled embedding dicts into a PooledEmbeddingStore."""
    return PooledEmbeddingStore(by_sequence=by_seq, by_id=by_id)




def _lookup_pooled_embedding(
    row: dict,
    by_id: dict[str, torch.Tensor],
    by_seq: dict[str, torch.Tensor],
    config,
) -> torch.Tensor | None:
    """Resolve a pooled embedding from sequence first, then fallback identifiers."""
    seq = normalize_sequence(row.get(config.sequence_column, ""))
    if seq:
        emb = by_seq.get(seq)
        if emb is not None:
            return emb

    for col in config.index_columns:
        key = clean_text(row.get(col, ""))
        if key and key in by_id:
            return by_id[key]
    return None


def build_ec_arrays(
    rows: list[dict],
    by_id: dict[str, torch.Tensor],
    by_seq: dict[str, torch.Tensor],
    config,
    label_builder: HierarchicalECLabelBuilder,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build pooled EC features plus per-level targets and masks."""
    level_names = label_builder.level_names()
    X_list: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {level_name: [] for level_name in level_names}
    target_masks: dict[str, list[torch.Tensor]] = {level_name: [] for level_name in level_names}
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
                torch.tensor(label_payload["target"][level_name], dtype=torch.float32)
            )
            target_masks[level_name].append(
                torch.tensor(label_payload["target_mask"][level_name], dtype=torch.bool)
            )

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings)")

    if not X_list:
        empty_targets = {
            level_name: torch.zeros(
                (0, len(label_builder.vocabulary_by_level[level_name])),
                dtype=torch.float32,
            )
            for level_name in level_names
        }
        empty_target_masks = {
            level_name: torch.zeros(
                (0, len(label_builder.vocabulary_by_level[level_name])),
                dtype=torch.bool,
            )
            for level_name in level_names
        }
        return torch.empty((0, 0), dtype=torch.float32), empty_targets, empty_target_masks

    return (
        torch.stack(X_list),
        {level_name: torch.stack(level_targets) for level_name, level_targets in targets.items()},
        {level_name: torch.stack(level_masks) for level_name, level_masks in target_masks.items()},
    )


def build_flat_ec_arrays(
    rows: list[dict],
    by_id: dict[str, torch.Tensor],
    by_seq: dict[str, torch.Tensor],
    config,
    label_builder: FlatECLabelBuilder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pooled EC features using flattened targets for XGBoost models."""
    label_builder._ensure_flat_index()
    X_list: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    skipped = 0

    for row in rows:
        emb = _lookup_pooled_embedding(row, by_id, by_seq, config)
        if emb is None:
            skipped += 1
            continue

        label_payload = label_builder.build(row)
        X_list.append(emb)
        targets.append(torch.tensor(label_payload["target"], dtype=torch.float32))
        target_masks.append(torch.tensor(label_payload["target_mask"], dtype=torch.bool))

    if skipped:
        log(f"      Skipped {skipped} rows (missing embeddings)")

    if not X_list:
        empty_dim = len(label_builder.flat_vocabulary or ())
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, empty_dim), dtype=torch.float32),
            torch.empty((0, empty_dim), dtype=torch.bool),
        )

    return (
        torch.stack(X_list),
        torch.stack(targets),
        torch.stack(target_masks),
    )




def build_xy_pooled(bundle, batch_kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build X (N, emb_dim) and y tensors from a dataset bundle.

    Embeddings that are already 1D (pre-pooled) are used directly.
    Embeddings that are 2D (residue-level) are mean-pooled on the fly.
    """
    x_list: list[torch.Tensor] = []
    y_list: list[float] = []
    for i in range(len(bundle.dataset)):
        sample = bundle.dataset[i]
        if batch_kind == "protein_global":
            x = sample["protein_emb"].float()
            if x.ndim == 2:
                x = x.mean(dim=0)
        elif batch_kind == "mutation_delta_global":
            x = sample["protein_delta_emb"].float()
            if x.ndim == 2:
                x = x.mean(dim=0)
        elif batch_kind == "protein_ligand_global":
            protein = sample["protein_emb"].float()
            if protein.ndim == 2:
                protein = protein.mean(dim=0)
            ligand = sample["ligand_feat"].float()
            x = torch.cat([protein, ligand])
        x_list.append(x)
        y_list.append(sample["target"])
    return torch.stack(x_list), torch.tensor(y_list, dtype=torch.float32)


def get_dataloader(
    bundle,
    batch_size: int = 32,
    collate_fn=None,
    *,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader for one dataset bundle."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        collate_fn=collate_fn if collate_fn is not None else bundle.collator,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
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
            level_name: np.stack(
                [
                    np.asarray(payload["target_mask"][level_name], dtype=bool)
                    for payload in label_payloads
                ]
            ).astype(bool, copy=False)
            if label_payloads
            else np.zeros(
                (0, len(label_builder.vocabulary_by_level[level_name])),
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
                for target_per_residue in label_payloads
            ],
            dtype=np.float32,
        )
        target_masks = _pad_1d_array_list(
            [
                np.ones_like(np.asarray(target_per_residue, dtype=np.int8))
                for target_per_residue in label_payloads
            ],
            dtype=np.int8,
        ).astype(bool, copy=False)
        return targets, target_masks

    raise ValueError(f"Unsupported label_kind '{label_kind}'.")


def get_target_payload(bundle):
    """Collect task targets and optional masks without reloading embeddings."""
    return _build_label_only_payload(bundle)


def load_embeddings(embeddings_json: str) -> dict[str, dict]:
    """Load embedding configs from JSON, resolving relative paths with EMBEDDINGS_ROOT if set."""
    with open(embeddings_json, "r", encoding="utf-8") as fh:
        embeddings = json.load(fh)
    root = os.environ.get("EMBEDDINGS_ROOT")
    if not root:
        return embeddings
    root_path = Path(root)
    resolved = {}
    for name, info in embeddings.items():
        info = dict(info)
        path = Path(info["path"])
        if not path.is_absolute():
            info["path"] = str(root_path / path)
        resolved[name] = info
    return resolved


def load_allowed_ids(path: str | Path) -> set[str]:
    """Load the optional allowlist used to restrict training to shared embeddings."""
    allowed_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                allowed_ids.add(value)
    return allowed_ids


def load_sequence_to_id_index(index_path: Path) -> dict[str, str]:
    """Load a normalized sequence-to-sequence-id mapping for embedding filters."""
    mapping: dict[str, str] = {}
    with open(index_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = normalize_sequence(row.get("protein_sequence", ""))
            sid = clean_text(row.get("sequence_id", ""))
            if seq and sid:
                mapping[seq] = sid
    return mapping


def _candidate_embedding_keys(
    row: dict[str, str],
    sequence_column: str,
    index_columns: tuple[str, ...],
    sequence_to_id: dict[str, str],
) -> set[str]:
    candidates: set[str] = set()
    seq = normalize_sequence(row.get(sequence_column, ""))
    if seq:
        sid = sequence_to_id.get(seq)
        if sid:
            candidates.add(sid)
        candidates.add(seq)
    for column in index_columns:
        value = clean_text(row.get(column, ""))
        if value:
            candidates.add(value)
    return candidates


def filter_rows_to_allowed_ids(
    rows: list[dict[str, str]],
    config,
    allowed_ids: set[str],
    sequence_to_id: dict[str, str],
) -> list[dict[str, str]]:
    """Keep only rows whose embedding ids survive the optional allowlist filter."""
    def _matches(row: dict[str, str]) -> bool:
        main = _candidate_embedding_keys(
            row, config.sequence_column, config.index_columns, sequence_to_id
        )
        if not main or not any(k in allowed_ids for k in main):
            return False
        if config.batch_kind == "mutation_delta_global" and config.wildtype_sequence_column:
            wt = _candidate_embedding_keys(
                row, config.wildtype_sequence_column, (), sequence_to_id
            )
            if not wt or not any(k in allowed_ids for k in wt):
                return False
        return True

    return [row for row in rows if _matches(row)]


__all__ = [
    "LazySafetensorsResidueEmbeddingStore",
    "build_ec_arrays",
    "build_flat_ec_arrays",
    "build_pooled_embedding_store",
    "build_xy_pooled",
    "collect_needed_lookup_keys",
    "filter_rows_to_allowed_ids",
    "get_dataloader",
    "get_target_payload",
    "get_task_rows",
    "load_allowed_ids",
    "load_embeddings",
    "load_lazy_safetensors_embedding_store",
    "load_pooled_embedding_subset",
    "load_pooled_safetensors_subset",
    "load_safetensors_embedding_store_subset",
    "load_sequence_to_id_index",
    "log",
    "resolve_embedding_source",
    "resolve_sequence_index_path",
]
