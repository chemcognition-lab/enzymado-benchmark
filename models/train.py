import sys
import json
import fcntl
import argparse
import inspect
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from models import MODEL_MAP, STRUCTURED_MODEL_MAP
except ImportError:
    from models.models import MODEL_MAP, STRUCTURED_MODEL_MAP
from data_loading import (
    BinaryLabelBuilder,
    FlatECLabelBuilder,
    HierarchicalECLabelBuilder,
    ResidueEmbeddingStore,
    ScalarRegressionLabelBuilder,
    build_task_dataset,
    get_task_config,
    TASK_REGISTRY,
)
from data_loading.adapters import (
    ProteinGlobalAdapter,
    ProteinLigandAdapter,
    MutationDeltaAdapter,
    ProteinResidueAdapter,
)

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_run_complete(run_dir: Path) -> bool:
    """A run is complete when test predictions and a saved model both exist."""
    has_preds = (run_dir / "test_predictions.npz").exists()
    has_model = (run_dir / "model.pt").exists() or (run_dir / "model.joblib").exists()
    return has_preds and has_model


SKIP_LABEL_KINDS: set[str] = set()

CARD_TO_REGISTRY: dict[str, str] = {
    "phopt": "ph_opt",
    "binding_site": "binding_sites",
}
SEQUENCE_INDEX_PATH = PROJECT_ROOT / "data/processed/protein_sequences_unique_no_fluorescence.csv"


def resolve_registry_name(card_task_name: str) -> str | None:
    name = CARD_TO_REGISTRY.get(card_task_name, card_task_name)
    return name if name in TASK_REGISTRY else None


def resolve_split_variant(default_split: str | None) -> str | None:
    if not default_split:
        return None
    if "random" in default_split:
        return "random"
    if "similarity" in default_split:
        return "similarity"
    return None


# ---------------------------------------------------------------------------
# H5 loading — per task, only needed lookup keys
# ---------------------------------------------------------------------------

def get_task_rows(config, split_name: str, split_variant: str) -> list[dict]:
    """Load CSV rows for a task split without touching any embeddings."""
    split_path = None
    if split_name is not None and config.split_column is None:
        split_path = config.split_files[split_variant]

    adapter_cls = {
        "protein_global": ProteinGlobalAdapter,
        "protein_ligand_global": ProteinLigandAdapter,
        "mutation_delta_global": MutationDeltaAdapter,
        "protein_residue": ProteinResidueAdapter,
    }[config.batch_kind]

    return adapter_cls(
        dataset_path=config.dataset_path,
        sample_id_candidates=config.sample_id_candidates,
        split_name=split_name,
        split_column=config.split_column,
        split_path=split_path,
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


class LazyResidueEmbeddingStore(ResidueEmbeddingStore):
    """Reads embeddings from H5 on-demand instead of loading all into RAM.

    Overrides lookup_id / lookup_sequence to read directly from H5.
    Use with DataLoader num_workers=0 (H5 file handle is not fork-safe).
    """

    def __init__(
        self,
        h5_path: str | Path,
        available_ids: set[str],
        seq_to_h5key: dict[str, str],
    ):
        # Initialize parent with empty dicts — no data loaded into RAM
        super().__init__(by_id={}, by_sequence={})
        self._h5_path = str(h5_path)
        self._h5 = None
        self._available_ids: set[str] = available_ids          # normalized ids in H5
        self._seq_to_h5key: dict[str, str] = seq_to_h5key     # norm_seq → H5 key

    def _open(self):
        import h5py
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self._h5_path, "r")

    def has_id(self, identifier: str) -> bool:
        from data_loading.splits import clean_text
        return clean_text(identifier) in self._available_ids

    def has_sequence(self, sequence: str) -> bool:
        from data_loading.splits import normalize_sequence as _ns
        return _ns(sequence) in self._seq_to_h5key

    def lookup_id(self, identifier: str) -> np.ndarray:
        from data_loading.splits import clean_text
        from data_loading.features import EmbeddingLookupError
        key = clean_text(identifier)
        if key not in self._available_ids:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")
        self._open()
        try:
            return np.asarray(self._h5[key], dtype=np.float32)
        except KeyError:
            raise EmbeddingLookupError(f"Missing residue embedding for id '{key}'.")

    def lookup_sequence(self, sequence: str) -> np.ndarray:
        from data_loading.splits import normalize_sequence as _ns
        from data_loading.features import EmbeddingLookupError
        norm_seq = _ns(sequence)
        h5_key = self._seq_to_h5key.get(norm_seq)
        if h5_key is None:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")
        self._open()
        try:
            return np.asarray(self._h5[h5_key], dtype=np.float32)
        except KeyError:
            raise EmbeddingLookupError("Missing residue embedding for sequence.")

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def _build_seq_to_h5key(
    h5_path: str | Path,
    needed_sequences: set[str],
) -> dict[str, str]:
    """Build normalized_sequence → H5_key mapping via sequence index or direct key match."""
    import h5py
    from data_loading.splits import clean_text, normalize_sequence as _ns

    normalized_sequences = {_ns(s) for s in needed_sequences if _ns(s)}
    seq_to_h5key: dict[str, str] = {}

    # Try sequence index first (maps sequence_id → sequence)
    seq_index_path = SEQUENCE_INDEX_PATH if SEQUENCE_INDEX_PATH.exists() else None
    if seq_index_path is not None and normalized_sequences:
        import csv
        with open(seq_index_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq = _ns(row.get("protein_sequence", ""))
                sid = clean_text(row.get("sequence_id", ""))
                if seq in normalized_sequences and sid:
                    seq_to_h5key[seq] = sid

    # Fall back: if the sequence itself is an H5 key
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
    """Return a LazyResidueEmbeddingStore — scans H5 keys but loads no data."""
    import h5py
    from data_loading.splits import clean_text

    full_path = PROJECT_ROOT / h5_path

    # Collect which needed_ids actually exist in H5
    with h5py.File(str(full_path), "r") as handle:
        h5_keys = {clean_text(k) for k in handle.keys()}

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)} & h5_keys
    # Filter seq_to_h5key so only mappings whose H5 key actually exists are kept
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


def load_embedding_store_subset(
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> ResidueEmbeddingStore:
    """Load only the embeddings needed for a task from an H5 file."""
    sequence_index_path = SEQUENCE_INDEX_PATH if SEQUENCE_INDEX_PATH.exists() else None
    return ResidueEmbeddingStore.from_h5_subset(
        PROJECT_ROOT / h5_path,
        needed_ids=needed_ids,
        needed_sequences=needed_sequences,
        sequence_index_path=sequence_index_path,
    )


def load_pooled_h5_subset(
    h5_path: str | Path,
    needed_ids: set[str],
    needed_sequences: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load only the needed H5 embeddings by key and mean-pool each one immediately.

    Reads only the specific keys required rather than scanning the whole file.
    Returns (by_id, by_seq) dicts of (emb_dim,) arrays instead of (seq_len, emb_dim).
    """
    import h5py
    from data_loading.splits import clean_text, normalize_sequence as _norm_seq

    normalized_ids = {clean_text(v) for v in needed_ids if clean_text(v)}
    normalized_sequences = {_norm_seq(v) for v in needed_sequences if _norm_seq(v)}

    # Build seq → H5 key mapping via sequence index, then fall back to direct key match
    seq_to_h5key = _build_seq_to_h5key(PROJECT_ROOT / h5_path, normalized_sequences)

    with h5py.File(str(PROJECT_ROOT / h5_path), "r") as handle:
        h5_keys = {clean_text(key) for key in handle.keys()}

    normalized_ids &= h5_keys
    seq_to_h5key = {
        seq: key
        for seq, key in seq_to_h5key.items()
        if key in h5_keys
    }

    # Collect the exact set of H5 keys to load and what they map to
    # key → (id_key | None, seq | None)
    keys_to_load: dict[str, tuple[str | None, str | None]] = {}
    for id_key in normalized_ids:
        entry = keys_to_load.get(id_key, (None, None))
        keys_to_load[id_key] = (id_key, entry[1])
    for seq, h5_key in seq_to_h5key.items():
        entry = keys_to_load.get(h5_key, (None, None))
        keys_to_load[h5_key] = (entry[0], seq)

    log(f"      seq_index mapped {len(seq_to_h5key)}/{len(normalized_sequences)} seqs → H5 keys")
    log(f"      will load {len(keys_to_load)} unique H5 keys ({len(normalized_ids)} by id, {len(seq_to_h5key)} by seq)")

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
            arr = np.asarray(handle[h5_key], dtype=np.float32).mean(axis=0)
            if id_key is not None:
                by_id[id_key] = arr
            if seq is not None:
                by_seq[seq] = arr
            if i % log_interval == 0 or i == total:
                elapsed = time.time() - t_load_start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                log(f"      ... {i:,}/{total:,} ({100*i/total:.0f}%)  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    if missing:
        log(f"      {missing} keys not found in H5 (will be skipped)")
    return by_id, by_seq


def build_Xy_ligand_direct(
    rows: list[dict],
    by_id: dict[str, np.ndarray],
    by_seq: dict[str, np.ndarray],
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X/y for protein_ligand_global from pre-pooled embeddings + rdkit ligand features."""
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

        # With mean pooling, mean(mut - wt) == mean(mut) - mean(wt).
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


def get_model_map(task_type: str, label_kind: str) -> dict[str, dict]:
    if label_kind in STRUCTURED_MODEL_MAP:
        return STRUCTURED_MODEL_MAP[label_kind]
    return MODEL_MAP[task_type]


# ---------------------------------------------------------------------------
# Data extraction — memory efficient
# ---------------------------------------------------------------------------

def build_Xy_pooled(bundle, batch_kind: str):
    """Build mean-pooled X (N, emb_dim) and y by iterating samples one at a time.
    Avoids loading a padded 3D array into memory.
    """
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


def get_dataloader(bundle, batch_size: int = 32) -> DataLoader:
    return DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        collate_fn=bundle.collator,
        shuffle=False,
        num_workers=0,
    )


def _pad_1d_array_list(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=dtype)
    max_length = max(array.shape[0] for array in arrays)
    batch = np.zeros((len(arrays), max_length), dtype=dtype)
    for index, array in enumerate(arrays):
        batch[index, : array.shape[0]] = array.astype(dtype, copy=False)
    return batch


def get_target_payload(bundle):
    """Collect task targets and optional masks without reloading embeddings."""
    label_kind = bundle.config.label_kind

    if label_kind in {"scalar_regression", "binary"}:
        return (
            np.array(
                [bundle.dataset[i]["target"] for i in range(len(bundle.dataset))],
                dtype=np.float32,
            ),
            None,
        )

    if label_kind == "multilabel_ec":
        level_names = bundle.label_builder.level_names()
        targets = {
            level_name: np.stack(
                [
                    np.asarray(bundle.dataset[i]["target"][level_name], dtype=np.float32)
                    for i in range(len(bundle.dataset))
                ]
            ).astype(np.float32, copy=False)
            for level_name in level_names
        }
        target_masks = {
            level_name: np.asarray(
                [
                    bool(bundle.dataset[i]["target_mask"][level_name])
                    for i in range(len(bundle.dataset))
                ],
                dtype=bool,
            )
            for level_name in level_names
        }
        return targets, target_masks

    if label_kind == "residue_binary_vector":
        targets = _pad_1d_array_list(
            [
                np.asarray(bundle.dataset[i]["target_per_residue"], dtype=np.float32)
                for i in range(len(bundle.dataset))
            ],
            dtype=np.float32,
        )
        target_masks = _pad_1d_array_list(
            [
                np.asarray(bundle.dataset[i]["target_mask"], dtype=np.int8)
                for i in range(len(bundle.dataset))
            ],
            dtype=np.int8,
        ).astype(bool, copy=False)
        return targets, target_masks

    raise ValueError(f"Unsupported label_kind '{label_kind}'.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_info: dict, X_train, y_train, X_val=None, y_val=None, **extra_fit_kwargs):
    model = model_info["wrap"](**model_info["params"])
    fit_signature = inspect.signature(model.fit)
    fit_kwargs = {}
    if X_val is not None and y_val is not None and "X_val" in fit_signature.parameters:
        fit_kwargs["X_val"] = X_val
        fit_kwargs["y_val"] = y_val
    for key, value in extra_fit_kwargs.items():
        if value is not None and key in fit_signature.parameters:
            fit_kwargs[key] = value
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_model_loader(model_info: dict, train_loader: DataLoader, batch_kind: str, val_loader: DataLoader | None = None):
    model = model_info["wrap"](**model_info["params"])
    fit_loader_signature = inspect.signature(model.fit_loader)
    fit_kwargs = {}
    if val_loader is not None and "val_dataloader" in fit_loader_signature.parameters:
        fit_kwargs["val_dataloader"] = val_loader
    model.fit_loader(train_loader, batch_kind=batch_kind, **fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _flatten_npz_payload(prefix: str, payload) -> dict[str, np.ndarray]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        flattened = {}
        for key, value in payload.items():
            flattened.update(_flatten_npz_payload(f"{prefix}_{key}", value))
        return flattened
    return {prefix: np.asarray(payload)}


def make_predictions_file(y_true, pred, run_dir: Path, split: str, target_mask=None):
    payload = {}
    payload.update(_flatten_npz_payload("y_true", y_true))
    payload.update(_flatten_npz_payload("predictions", pred))
    payload.update(_flatten_npz_payload("target_mask", target_mask))
    np.savez(str(run_dir / f"{split}_predictions.npz"), **payload)


def save_model(trained, run_dir: Path) -> Path:
    if hasattr(trained, "model") and isinstance(trained.model, torch.nn.Module):
        path = run_dir / "model.pt"
        torch.save(trained.model.state_dict(), path)
    else:
        path = run_dir / "model.joblib"
        joblib.dump(trained, path)
    return path


def make_result_entry(emb_name, model_name, model_info, run_dir: Path, model_file: Path) -> dict:
    return {
        "embedding_model": emb_name,
        "name": model_name,
        "params": model_info["params"],
        "train_pred_file": str(run_dir / "train_predictions.npz"),
        "test_pred_file": str(run_dir / "test_predictions.npz"),
        "model_file": str(model_file),
    }


def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(data: dict, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_card_locked(card_path: Path, task_name: str, result_key: str, result_entry: dict):
    """Thread/process-safe update of a single result entry in the benchmark card."""
    lock_path = card_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            card = read_json(card_path)
            card["properties"][task_name].setdefault("training_results", {})
            card["properties"][task_name]["training_results"][result_key] = result_entry
            write_json(card, card_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(card_path: str, embeddings_json: str, output_dir: str, emb_name_filter: str | None = None):
    log(f"[startup] Python started. emb_name={emb_name_filter}")
    card = read_json(card_path)
    embeddings = read_json(embeddings_json)
    log(f"[startup] JSON loaded. {len(embeddings)} embedding models, {len(card['properties'])} tasks.")
    card_path = Path(card_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if emb_name_filter is not None and emb_name_filter not in embeddings:
        raise ValueError(f"Unknown embedding '{emb_name_filter}'. Available: {', '.join(embeddings)}")

    for emb_name, emb_info in embeddings.items():
        if emb_name_filter is not None and emb_name != emb_name_filter:
            continue

        log(f"\n=== Embedding: {emb_name} (dim={emb_info['dim']}) ===")

        for task_name, prop in card["properties"].items():
            task_type = prop.get("task_type")
            if not task_type:
                raise ValueError(f"'{task_name}' is missing 'task_type' in card")

            registry_name = resolve_registry_name(task_name)
            if registry_name is None:
                log(f"  Skipping '{task_name}' (not in TASK_REGISTRY)")
                continue

            config = get_task_config(registry_name)

            if config.label_kind in SKIP_LABEL_KINDS:
                log(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' not yet supported)")
                continue

            dataset_info = prop["datasets"][0]
            default_split = dataset_info.get("splits", {}).get("default_split")
            split_variant = resolve_split_variant(default_split)
            if split_variant is None:
                log(f"  Skipping '{task_name}' (no usable split defined in card)")
                continue

            log(f"\n  Task: {task_name} | type: {task_type} | split: {default_split}")
            try:

                # Check whether all models for this task are already trained — skip H5 loading if so
                prospective_model_map = get_model_map(task_type, config.label_kind)
                already_done = [
                    m for m in prospective_model_map
                    if is_run_complete(output_dir / f"{emb_name}__{task_name}__{m}")
                ]
                if len(already_done) == len(prospective_model_map):
                    log(f"    All {len(already_done)} models already trained — skipping task.")
                    continue
                if already_done:
                    log(f"    {len(already_done)}/{len(prospective_model_map)} models already done: {already_done}")

                # Load only the embeddings needed for this task (not the entire H5)
                log(f"    Loading CSV rows...")
                train_rows = get_task_rows(config, "train", split_variant)
                val_rows = get_task_rows(config, "val", split_variant)
                test_rows = get_task_rows(config, "test", split_variant)
                needed_ids, needed_sequences = collect_needed_lookup_keys(train_rows + val_rows + test_rows, config)
                log(f"    Rows loaded: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} | need {len(needed_ids)} ids, {len(needed_sequences)} seqs")

                is_ligand_task = config.batch_kind == "protein_ligand_global"
                is_pooled_ec_task = config.label_kind == "multilabel_ec"
                is_pooled_protein_global_task = (
                    config.batch_kind == "protein_global"
                    and config.label_kind in {"scalar_regression", "binary"}
                )
                is_pooled_mutation_task = (
                    config.batch_kind == "mutation_delta_global"
                    and config.label_kind in {"scalar_regression", "binary"}
                )
                flat_ec_label_builder = None
                X_train_flat_ec = y_train_flat_ec = target_mask_train_flat_ec = None
                X_val_flat_ec = y_val_flat_ec = target_mask_val_flat_ec = None
                X_test_flat_ec = y_test_flat_ec = target_mask_test_flat_ec = None

                if is_ligand_task:
                    # Pool on H5 load — avoids storing (seq_len, emb_dim) arrays for large datasets
                    log(f"    Loading pooled embeddings from H5 (ligand task)...")
                    t0 = time.time()
                    by_id, by_seq = load_pooled_h5_subset(emb_info["path"], needed_ids, needed_sequences)
                    log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

                    log(f"    Building X/y arrays...")
                    X_train, y_train = build_Xy_ligand_direct(train_rows, by_id, by_seq, config)
                    X_val, y_val = build_Xy_ligand_direct(val_rows, by_id, by_seq, config)
                    X_test, y_test = build_Xy_ligand_direct(test_rows, by_id, by_seq, config)
                    store = None
                elif is_pooled_ec_task:
                    log(f"    Loading pooled embeddings from H5 (EC task)...")
                    t0 = time.time()
                    by_id, by_seq = load_pooled_h5_subset(emb_info["path"], needed_ids, needed_sequences)
                    log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

                    log(f"    Fitting EC vocabulary on train rows...")
                    ec_label_builder = HierarchicalECLabelBuilder(
                        target_column=str(config.target_column),
                        separator=config.ec_separator,
                    ).fit(train_rows)
                    flat_ec_label_builder = FlatECLabelBuilder(
                        target_column=str(config.target_column),
                        separator=config.ec_separator,
                        vocabulary_by_level=ec_label_builder.vocabulary_by_level,
                    )

                    log(f"    Building pooled EC arrays...")
                    X_train, y_train, target_mask_train = build_multilabel_ec_direct(
                        train_rows, by_id, by_seq, config, ec_label_builder
                    )
                    X_val, y_val, target_mask_val = build_multilabel_ec_direct(
                        val_rows, by_id, by_seq, config, ec_label_builder
                    )
                    X_test, y_test, target_mask_test = build_multilabel_ec_direct(
                        test_rows, by_id, by_seq, config, ec_label_builder
                    )
                    log(f"    Building pooled flat EC arrays for XGBoost...")
                    X_train_flat_ec, y_train_flat_ec, target_mask_train_flat_ec = build_flat_multilabel_ec_direct(
                        train_rows, by_id, by_seq, config, flat_ec_label_builder
                    )
                    X_val_flat_ec, y_val_flat_ec, target_mask_val_flat_ec = build_flat_multilabel_ec_direct(
                        val_rows, by_id, by_seq, config, flat_ec_label_builder
                    )
                    X_test_flat_ec, y_test_flat_ec, target_mask_test_flat_ec = build_flat_multilabel_ec_direct(
                        test_rows, by_id, by_seq, config, flat_ec_label_builder
                    )
                    store = None
                elif is_pooled_protein_global_task:
                    log(f"    Loading pooled embeddings from H5 (protein-global task)...")
                    t0 = time.time()
                    by_id, by_seq = load_pooled_h5_subset(emb_info["path"], needed_ids, needed_sequences)
                    log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

                    log(f"    Building pooled X/y arrays...")
                    X_train, y_train = build_protein_global_Xy_direct(train_rows, by_id, by_seq, config)
                    X_val, y_val = build_protein_global_Xy_direct(val_rows, by_id, by_seq, config)
                    X_test, y_test = build_protein_global_Xy_direct(test_rows, by_id, by_seq, config)
                    store = None
                elif is_pooled_mutation_task:
                    log(f"    Loading pooled embeddings from H5 (mutation-delta task)...")
                    t0 = time.time()
                    by_id, by_seq = load_pooled_h5_subset(emb_info["path"], needed_ids, needed_sequences)
                    log(f"    Loaded {len(by_id)} id + {len(by_seq)} seq pooled embeddings ({time.time()-t0:.1f}s)")

                    log(f"    Building pooled mutation-delta X/y arrays...")
                    X_train, y_train = build_mutation_delta_Xy_direct(train_rows, by_id, by_seq, config)
                    X_val, y_val = build_mutation_delta_Xy_direct(val_rows, by_id, by_seq, config)
                    X_test, y_test = build_mutation_delta_Xy_direct(test_rows, by_id, by_seq, config)
                    store = None
                else:
                    # Lazy store — scans H5 keys but reads embeddings on-demand per mini-batch
                    log(f"    Scanning H5 keys for lazy store...")
                    t0 = time.time()
                    store = load_lazy_embedding_store(emb_info["path"], needed_ids, needed_sequences)
                    log(f"    Lazy store ready ({time.time()-t0:.1f}s) — {len(store._available_ids)} ids, {len(store._seq_to_h5key)} seqs indexed")

                    log(f"    Building task datasets...")
                    t0 = time.time()
                    train_bundle = build_task_dataset(
                        registry_name, store,
                        split_name="train", split_variant=split_variant, missing_policy="skip",
                    )
                    ec_vocabulary = None
                    if config.label_kind == "multilabel_ec":
                        ec_vocabulary = train_bundle.label_builder.vocabulary_by_level
                    val_bundle = build_task_dataset(
                        registry_name, store,
                        split_name="val", split_variant=split_variant, missing_policy="skip",
                        ec_vocabulary=ec_vocabulary,
                    )
                    test_bundle = build_task_dataset(
                        registry_name, store,
                        split_name="test", split_variant=split_variant, missing_policy="skip",
                        ec_vocabulary=ec_vocabulary,
                    )
                    log(f"    Datasets built ({time.time()-t0:.1f}s)")

                if is_ligand_task or is_pooled_ec_task or is_pooled_protein_global_task or is_pooled_mutation_task:
                    n_train = len(X_train)
                    n_val   = len(X_val)
                    n_test  = len(X_test)
                else:
                    n_train = len(train_bundle.dataset)
                    n_val   = len(val_bundle.dataset)
                    n_test  = len(test_bundle.dataset)

                orig_train = len(train_rows)
                orig_val   = len(val_rows)
                orig_test  = len(test_rows)
                orig_total = orig_train + orig_val + orig_test
                kept_total = n_train + n_val + n_test
                dropped    = orig_total - kept_total

                log(
                    f"    Data summary:"
                    f"\n      original : train={orig_train:,}  val={orig_val:,}  test={orig_test:,}  total={orig_total:,}"
                    f"\n      kept     : train={n_train:,}  val={n_val:,}  test={n_test:,}  total={kept_total:,}"
                    f"\n      dropped  : {dropped:,} ({100*dropped/orig_total:.1f}% of original)"
                    + (f"  ← missing embeddings/ligand features" if dropped > 0 else "")
                )

                if n_train == 0 or n_test == 0:
                    log(f"    Skipping '{task_name}': empty dataset after filtering.")
                    del store
                    continue

                use_validation = n_val > 0
                if use_validation:
                    log(f"    Validation enabled: n_val={n_val:,}")
                else:
                    log("    Validation disabled: empty val split after filtering")

                model_map = get_model_map(task_type, config.label_kind)

                for model_name, model_info in model_map.items():
                    run_dir = output_dir / f"{emb_name}__{task_name}__{model_name}"

                    if is_run_complete(run_dir):
                        log(f"    [{model_name}] Already trained — skipping.")
                        continue

                    residue_aware = model_info.get("residue_aware", False) and not (
                        is_ligand_task or is_pooled_ec_task or is_pooled_protein_global_task or is_pooled_mutation_task
                    )
                    log(f"    [{model_name}] n_train={n_train}, n_test={n_test} | residue_aware={residue_aware}")

                    run_dir.mkdir(parents=True, exist_ok=True)

                    t0 = time.time()
                    y_train_to_save = None
                    y_val_to_save = None
                    y_test_to_save = None
                    target_mask_train_to_save = None
                    target_mask_val_to_save = None
                    target_mask_test_to_save = None
                    if is_ligand_task:
                        log(f"      Training...")
                        trained = train_model(model_info, X_train, y_train,
                                             X_val=X_val if use_validation else None,
                                             y_val=y_val if use_validation else None)
                        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                        pred_train = trained.predict(X_train)
                        pred_val = trained.predict(X_val) if use_validation else None
                        pred_test = trained.predict(X_test)
                        y_train_to_save = y_train
                        y_val_to_save = y_val if use_validation else None
                        y_test_to_save = y_test
                    elif is_pooled_protein_global_task:
                        log(f"      Training on pooled protein-global embeddings...")
                        trained = train_model(
                            model_info,
                            X_train,
                            y_train,
                            X_val=X_val if use_validation else None,
                            y_val=y_val if use_validation else None,
                        )
                        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                        pred_train = trained.predict(X_train)
                        pred_val = trained.predict(X_val) if use_validation else None
                        pred_test = trained.predict(X_test)
                        y_train_to_save = y_train
                        y_val_to_save = y_val if use_validation else None
                        y_test_to_save = y_test
                    elif is_pooled_mutation_task:
                        log(f"      Training on pooled mutation-delta embeddings...")
                        trained = train_model(
                            model_info,
                            X_train,
                            y_train,
                            X_val=X_val if use_validation else None,
                            y_val=y_val if use_validation else None,
                        )
                        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                        pred_train = trained.predict(X_train)
                        pred_val = trained.predict(X_val) if use_validation else None
                        pred_test = trained.predict(X_test)
                        y_train_to_save = y_train
                        y_val_to_save = y_val if use_validation else None
                        y_test_to_save = y_test
                    elif is_pooled_ec_task:
                        if model_info.get("flat_ec_labels", False):
                            log(f"      Training on pooled flat EC embeddings for XGBoost...")
                            trained = train_model(
                                model_info,
                                X_train_flat_ec,
                                y_train_flat_ec,
                                X_val=X_val_flat_ec if use_validation else None,
                                y_val=y_val_flat_ec if use_validation else None,
                                target_mask=target_mask_train_flat_ec,
                                target_mask_val=target_mask_val_flat_ec if use_validation else None,
                            )
                            log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                            pred_train = trained.predict(X_train_flat_ec)
                            pred_val = trained.predict(X_val_flat_ec) if use_validation else None
                            pred_test = trained.predict(X_test_flat_ec)
                            y_train_to_save = y_train_flat_ec
                            y_val_to_save = y_val_flat_ec if use_validation else None
                            y_test_to_save = y_test_flat_ec
                            target_mask_train_to_save = target_mask_train_flat_ec
                            target_mask_val_to_save = target_mask_val_flat_ec if use_validation else None
                            target_mask_test_to_save = target_mask_test_flat_ec
                        else:
                            log(f"      Training on pooled EC embeddings...")
                            trained = train_model(
                                model_info,
                                X_train,
                                y_train,
                                X_val=X_val if use_validation else None,
                                y_val=y_val if use_validation else None,
                                target_mask=target_mask_train,
                                target_mask_val=target_mask_val if use_validation else None,
                            )
                            log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                            pred_train = trained.predict(X_train)
                            pred_val = trained.predict(X_val) if use_validation else None
                            pred_test = trained.predict(X_test)
                            y_train_to_save = y_train
                            y_val_to_save = y_val if use_validation else None
                            y_test_to_save = y_test
                            target_mask_train_to_save = target_mask_train
                            target_mask_val_to_save = target_mask_val if use_validation else None
                            target_mask_test_to_save = target_mask_test
                    elif residue_aware:
                        train_loader = get_dataloader(train_bundle, batch_size=64)
                        val_loader = get_dataloader(val_bundle, batch_size=64) if use_validation else None
                        test_loader = get_dataloader(test_bundle, batch_size=64)
                        log(f"      Training (residue-aware)...")
                        trained = train_model_loader(model_info, train_loader, config.batch_kind, val_loader=val_loader)
                        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                        pred_val = trained.predict_loader(val_loader, batch_kind=config.batch_kind) if use_validation else None
                        pred_test = trained.predict_loader(test_loader, batch_kind=config.batch_kind)
                        pred_train = trained.predict_loader(train_loader, batch_kind=config.batch_kind)
                        y_val_to_save, target_mask_val_to_save = get_target_payload(val_bundle) if use_validation else (None, None)
                        y_test_to_save, target_mask_test_to_save = get_target_payload(test_bundle)
                        y_train_to_save, target_mask_train_to_save = get_target_payload(train_bundle)
                    else:
                        log(f"      Pooling embeddings...")
                        X_train_p, y_train_p = build_Xy_pooled(train_bundle, config.batch_kind)
                        X_val_p, y_val_p = build_Xy_pooled(val_bundle, config.batch_kind) if use_validation else (None, None)
                        X_test_p, y_test_p = build_Xy_pooled(test_bundle, config.batch_kind)
                        log(f"      Pooled shapes: train={X_train_p.shape}, test={X_test_p.shape}. Training...")
                        trained = train_model(model_info, X_train_p, y_train_p, X_val=X_val_p, y_val=y_val_p)
                        log(f"      Training done ({time.time()-t0:.1f}s). Predicting...")
                        pred_val = trained.predict(X_val_p) if use_validation else None
                        pred_test = trained.predict(X_test_p)
                        pred_train = trained.predict(X_train_p)
                        y_train_to_save = y_train_p
                        y_val_to_save = y_val_p if use_validation else None
                        y_test_to_save = y_test_p

                    make_predictions_file(y_train_to_save, pred_train, run_dir, "train", target_mask_train_to_save)
                    if use_validation:
                        make_predictions_file(y_val_to_save, pred_val, run_dir, "val", target_mask_val_to_save)
                    make_predictions_file(y_test_to_save, pred_test, run_dir, "test", target_mask_test_to_save)
                    model_file = save_model(trained, run_dir)

                    if getattr(trained, "best_epoch_", None) is not None:
                        log(
                            "      Early stopping: "
                            f"best_epoch={trained.best_epoch_}, "
                            f"best_val_loss={trained.best_val_loss_:.6f}"
                        )

                    result_key = f"{emb_name}__{model_name}"
                    result_entry = make_result_entry(emb_name, model_name, model_info, run_dir, model_file)
                    update_card_locked(card_path, task_name, result_key, result_entry)
                    log(f"      Saved → {run_dir.name}")

                del store  # free task embeddings before loading next task
            except Exception as _task_exc:
                import traceback
                log(f"  [ERROR] Task '{task_name}' failed: {_task_exc}")
                log(traceback.format_exc())
                continue

    log(f"\nDone. Results written to {card_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to embeddings_models_output.json")
    parser.add_argument("output_dir", help="Directory to write prediction and model files")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding (for SLURM arrays)")
    args = parser.parse_args()
    main(args.card_path, args.embeddings_json, args.output_dir, args.emb_name)
