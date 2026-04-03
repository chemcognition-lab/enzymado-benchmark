import sys
import json
import fcntl
import h5py
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import MODEL_MAP
from data_loading import (
    ResidueEmbeddingStore,
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

SKIP_LABEL_KINDS = {"multilabel_ec", "residue_binary_vector"}

CARD_TO_REGISTRY: dict[str, str] = {
    "phopt": "ph_opt",
    "binding_site": "binding_sites",
}


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
# H5 loading — per task, only needed IDs
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


def get_needed_ids(rows: list[dict], config) -> set[str]:
    """Extract all column values that could be used as H5 lookup keys."""
    lookup_cols = list(config.sample_id_candidates) + [config.sequence_column]
    if config.wildtype_sequence_column:
        lookup_cols.append(config.wildtype_sequence_column)
    ids = set()
    for row in rows:
        for col in lookup_cols:
            val = row.get(col, "").strip()
            if val:
                ids.add(val)
    return ids


def load_embedding_store_subset(h5_path: str | Path, needed_ids: set[str]) -> ResidueEmbeddingStore:
    """Load only the embeddings matching needed_ids from an H5 file."""
    by_id = {}
    with h5py.File(PROJECT_ROOT / h5_path, "r") as f:
        for key in f.keys():
            if key in needed_ids:
                by_id[key] = f[key][:]
    return ResidueEmbeddingStore(by_id=by_id)


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


def get_y_from_bundle(bundle) -> np.ndarray:
    """Collect y values from a dataset bundle without loading embeddings again."""
    return np.array(
        [bundle.dataset[i]["target"] for i in range(len(bundle.dataset))],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_info: dict, X_train, y_train):
    model = model_info["wrap"](**model_info["params"])
    model.fit(X_train, y_train)
    return model


def train_model_loader(model_info: dict, train_loader: DataLoader, batch_kind: str):
    model = model_info["wrap"](**model_info["params"])
    model.fit_loader(train_loader, batch_kind=batch_kind)
    return model


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_predictions_file(y_true, pred, run_dir: Path, split: str):
    np.savez(str(run_dir / f"{split}_predictions.npz"), y_true=y_true, predictions=pred)


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
    card = read_json(card_path)
    embeddings = read_json(embeddings_json)
    card_path = Path(card_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if emb_name_filter is not None and emb_name_filter not in embeddings:
        raise ValueError(f"Unknown embedding '{emb_name_filter}'. Available: {', '.join(embeddings)}")

    for emb_name, emb_info in embeddings.items():
        if emb_name_filter is not None and emb_name != emb_name_filter:
            continue

        print(f"\n=== Embedding: {emb_name} (dim={emb_info['dim']}) ===")

        for task_name, prop in card["properties"].items():
            task_type = prop.get("task_type")
            if not task_type:
                raise ValueError(f"'{task_name}' is missing 'task_type' in card")

            registry_name = resolve_registry_name(task_name)
            if registry_name is None:
                print(f"  Skipping '{task_name}' (not in TASK_REGISTRY)")
                continue

            config = get_task_config(registry_name)

            if config.label_kind in SKIP_LABEL_KINDS:
                print(f"  Skipping '{task_name}' (label_kind='{config.label_kind}' not yet supported)")
                continue

            dataset_info = prop["datasets"][0]
            default_split = dataset_info.get("splits", {}).get("default_split")
            split_variant = resolve_split_variant(default_split)
            if split_variant is None:
                print(f"  Skipping '{task_name}' (no usable split defined in card)")
                continue

            print(f"\n  Task: {task_name} | type: {task_type} | split: {default_split}")

            # Load only the embeddings needed for this task (not the entire H5)
            train_rows = get_task_rows(config, "train", split_variant)
            test_rows = get_task_rows(config, "test", split_variant)
            needed_ids = get_needed_ids(train_rows + test_rows, config)
            store = load_embedding_store_subset(emb_info["path"], needed_ids)
            print(f"    Loaded {len(store.by_id)} embeddings for this task")

            train_bundle = build_task_dataset(
                registry_name, store,
                split_name="train", split_variant=split_variant, missing_policy="skip",
            )
            test_bundle = build_task_dataset(
                registry_name, store,
                split_name="test", split_variant=split_variant, missing_policy="skip",
            )

            if len(train_bundle.dataset) == 0 or len(test_bundle.dataset) == 0:
                print(f"    Skipping '{task_name}': empty dataset (train={len(train_bundle.dataset)}, test={len(test_bundle.dataset)}). Check embeddings and rdkit.")
                del store
                continue

            # Ligand tasks must always pool — can't concat 3D protein + 1D ligand
            use_residue_aware = config.batch_kind != "protein_ligand_global"

            for model_name, model_info in MODEL_MAP[task_type].items():
                residue_aware = model_info.get("residue_aware", False) and use_residue_aware
                print(f"    [{model_name}] n_train={len(train_bundle.dataset)}, n_test={len(test_bundle.dataset)}")

                run_dir = output_dir / f"{emb_name}__{task_name}__{model_name}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if residue_aware:
                    train_loader = get_dataloader(train_bundle, batch_size=64)
                    test_loader = get_dataloader(test_bundle, batch_size=64)
                    trained = train_model_loader(model_info, train_loader, config.batch_kind)
                    pred_test = trained.predict_loader(test_loader, batch_kind=config.batch_kind)
                    pred_train = trained.predict_loader(train_loader, batch_kind=config.batch_kind)
                    y_test = get_y_from_bundle(test_bundle)
                    y_train = get_y_from_bundle(train_bundle)
                else:
                    X_train, y_train = build_Xy_pooled(train_bundle, config.batch_kind)
                    X_test, y_test = build_Xy_pooled(test_bundle, config.batch_kind)
                    trained = train_model(model_info, X_train, y_train)
                    pred_test = trained.predict(X_test)
                    pred_train = trained.predict(X_train)

                make_predictions_file(y_train, pred_train, run_dir, "train")
                make_predictions_file(y_test, pred_test, run_dir, "test")
                model_file = save_model(trained, run_dir)

                result_key = f"{emb_name}__{model_name}"
                result_entry = make_result_entry(emb_name, model_name, model_info, run_dir, model_file)
                update_card_locked(card_path, task_name, result_key, result_entry)

            del store  # free task embeddings before loading next task

    print(f"\nDone. Results written to {card_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_path", help="Path to the dataset card JSON")
    parser.add_argument("embeddings_json", help="Path to embeddings_models_output.json")
    parser.add_argument("output_dir", help="Directory to write prediction and model files")
    parser.add_argument("--emb_name", default=None, help="Train only this embedding (for SLURM arrays)")
    args = parser.parse_args()
    main(args.card_path, args.embeddings_json, args.output_dir, args.emb_name)
