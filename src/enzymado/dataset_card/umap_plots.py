"""Load protein embeddings and render dataset-card UMAP figures."""

import collections.abc
from pathlib import Path
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors

try:
    import umap as umap_lib
except ImportError:
    umap_lib = None


def _candidate_keys(
    sequence: typing.Any,
    uniprot_id: typing.Any,
) -> list[str]:
    """Return supported safetensors keys for one dataset row."""
    keys = []
    if uniprot_id:
        keys.append(str(uniprot_id).strip())
        keys.append(str(uniprot_id).strip().replace("/", "_").replace(".", "_"))
    if sequence:
        keys.append("".join(str(sequence).strip().split()).upper())
    return keys


def load_embeddings(
    sequences: list[typing.Any],
    uniprot_ids: list[typing.Any],
    emb_path: Path,
) -> tuple[np.ndarray | None, list[int]]:
    """Load and mean-pool embeddings, returning their source row indices."""
    if not emb_path.exists():
        print(f"  [UMAP] safetensors file not found: {emb_path}")
        return None, []

    embeddings = []
    valid_indices = []
    n_missing = 0
    with safetensors.safe_open(
        str(emb_path),
        framework="pt",
        device="cpu",
    ) as handle:
        available_keys = set(handle.keys())
        for index, (sequence, uniprot_id) in enumerate(zip(sequences, uniprot_ids)):
            embedding = None
            for key in _candidate_keys(sequence, uniprot_id):
                if key in available_keys:
                    tensor = handle.get_tensor(key)
                    raw = tensor.float().cpu().numpy().astype(np.float32, copy=False)
                    embedding = raw.mean(axis=0) if raw.ndim == 2 else raw
                    break
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(index)
            else:
                n_missing += 1

    n_found = len(embeddings)
    if n_missing > 0:
        print(
            f"  [UMAP] {n_found} embeddings found, {n_missing} missing in safetensors."
        )
    if n_found < 5:
        print(f"  [UMAP] Too few embeddings ({n_found}) — skipping UMAP.")
        return None, []

    return np.stack(embeddings, axis=0), valid_indices


def _plot_labels(
    coords: np.ndarray,
    valid_df: pd.DataFrame,
    target_col: str | None,
    prop_name: str,
    emb_model: str,
    output_dir: Path,
) -> Path:
    """Render and save the target-colored UMAP panel."""
    figure, axis = plt.subplots(figsize=(8, 6))
    if target_col and target_col in valid_df.columns:
        labels = valid_df[target_col]
        numeric = pd.api.types.is_numeric_dtype(labels) and labels.nunique() >= 15
        if numeric:
            scatter = axis.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels.values,
                cmap="viridis",
                s=8,
                alpha=0.7,
                linewidths=0,
            )
            plt.colorbar(scatter, ax=axis, label=target_col)
        else:
            unique_labels = labels.dropna().unique()
            color_map = plt.get_cmap("tab20", len(unique_labels))
            label_colors = {
                label: color_map(index) for index, label in enumerate(unique_labels)
            }
            colors = [label_colors.get(label, (0.5, 0.5, 0.5, 1.0)) for label in labels]
            axis.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                s=8,
                alpha=0.7,
                linewidths=0,
            )
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_colors[label],
                    markersize=6,
                    label=str(label),
                )
                for label in unique_labels[:20]
            ]
            axis.legend(
                handles=handles,
                title=target_col,
                fontsize=6,
                markerscale=1.5,
                loc="best",
                ncol=max(1, len(unique_labels) // 10),
            )
    else:
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            s=8,
            alpha=0.6,
            linewidths=0,
        )

    axis.set_title(
        f"{prop_name} — UMAP ({emb_model})\nColored by: {target_col or 'none'}",
        fontsize=11,
    )
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.set_xticks([])
    axis.set_yticks([])
    plt.tight_layout()
    label_path = output_dir / f"{prop_name}_{emb_model}_labels.png"
    plt.savefig(label_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  [UMAP] Saved label-colored plot → {label_path}")
    return label_path


def _split_assignments(
    valid_df: pd.DataFrame,
    split_files: dict[str, Path],
    id_col: str | None,
    seq_col: str | None,
) -> tuple[pd.Series | None, Path | None]:
    """Align one preferred split file with the rows represented in a UMAP."""
    split_path = next(
        (path for name, path in split_files.items() if "random" in name),
        next(iter(split_files.values()), None),
    )
    if split_path is None:
        return None, None

    split_df = pd.read_csv(split_path).copy()
    split_col = next(
        (column for column in split_df.columns if column.lower() in ["split", "set"]),
        None,
    )
    if not split_col:
        return None, split_path

    merge_col = (
        id_col
        if id_col and id_col in valid_df.columns and id_col in split_df.columns
        else seq_col
        if seq_col and seq_col in split_df.columns
        else None
    )
    if not merge_col:
        return None, split_path

    merged = valid_df[[merge_col]].merge(
        split_df[[merge_col, split_col]],
        on=merge_col,
        how="left",
    )
    return merged[split_col].fillna("unknown"), split_path


def _plot_splits(
    coords: np.ndarray,
    assignments: pd.Series | None,
    split_path: Path | None,
    prop_name: str,
    emb_model: str,
    output_dir: Path,
) -> Path:
    """Render and save the split-colored UMAP panel."""
    figure, axis = plt.subplots(figsize=(8, 6))
    split_colors = {
        "train": "#4C72B0",
        "val": "#DD8452",
        "validation": "#DD8452",
        "test": "#55A868",
        "unknown": "#C0C0C0",
    }
    if assignments is not None:
        for split_name, color in split_colors.items():
            mask = assignments == split_name
            if mask.any():
                axis.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=color,
                    label=split_name,
                    s=8,
                    alpha=0.7,
                    linewidths=0,
                )
        axis.legend(title="Split", fontsize=8, markerscale=2)
    else:
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            s=8,
            alpha=0.6,
            linewidths=0,
        )

    split_name = split_path.stem if split_path else "no split"
    axis.set_title(
        f"{prop_name} — UMAP ({emb_model})\nColored by: split  [{split_name}]",
        fontsize=11,
    )
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.set_xticks([])
    axis.set_yticks([])
    plt.tight_layout()
    split_plot_path = output_dir / f"{prop_name}_{emb_model}_splits.png"
    plt.savefig(split_plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  [UMAP] Saved split-colored plot → {split_plot_path}")
    return split_plot_path


def generate(
    base_df: pd.DataFrame,
    prop_name: str,
    split_files: dict[str, Path],
    emb_path: Path,
    emb_model: str,
    output_dir: Path,
    target_map: dict[str, str],
    load_embeddings_fn: collections.abc.Callable[
        [list[typing.Any], list[typing.Any], Path],
        tuple[np.ndarray | None, list[int]],
    ],
) -> dict[str, str] | None:
    """Generate label-colored and split-colored UMAP plots for one property."""
    if umap_lib is None:
        print("  [UMAP] Missing dependency (umap-learn) — skipping UMAP generation.")
        return None

    matplotlib.use("Agg")
    seq_col = next(
        (
            column
            for column in base_df.columns
            if column in ["protein_sequence", "wildtype_sequence"]
        ),
        None,
    )
    id_col = next(
        (column for column in base_df.columns if column in ["uniprot_id"]),
        None,
    )
    target_col = target_map.get(prop_name)
    sequences = base_df[seq_col].tolist() if seq_col else [None] * len(base_df)
    uniprot_ids = base_df[id_col].tolist() if id_col else [None] * len(base_df)

    print(
        f"  [UMAP] Loading embeddings from {emb_path.name} "
        f"for {len(base_df)} proteins..."
    )
    emb_matrix, valid_indices = load_embeddings_fn(
        sequences,
        uniprot_ids,
        emb_path,
    )
    if emb_matrix is None:
        return None

    valid_df = base_df.iloc[valid_indices].reset_index(drop=True)
    print(f"  [UMAP] Running UMAP on {emb_matrix.shape[0]} embeddings...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
    coords = reducer.fit_transform(emb_matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = _plot_labels(
        coords,
        valid_df,
        target_col,
        prop_name,
        emb_model,
        output_dir,
    )
    assignments, split_path = _split_assignments(
        valid_df,
        split_files,
        id_col,
        seq_col,
    )
    split_plot_path = _plot_splits(
        coords,
        assignments,
        split_path,
        prop_name,
        emb_model,
        output_dir,
    )
    return {
        "embedding_model": emb_model,
        "label_colored": str(label_path),
        "split_colored": str(split_plot_path),
    }
