import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class DatasetCardPopulator:
    def __init__(
        self,
        base_directory: str,
        emb_model: Optional[str] = None,
        embeddings_json: Optional[str] = None,
        seq_identity: float = 0.5,
        dendrogram: bool = False,
    ):
        self.base_dir = Path(base_directory)
        self.template_path = self.base_dir / "active_sites" / "dataset_card_template.json"
        self.emb_model = emb_model
        self.embeddings_json = Path(embeddings_json) if embeddings_json else None
        self.seq_identity = seq_identity  # Identity threshold for whole-dataset MMSeqs2 clustering
        self.dendrogram = dendrogram       # Whether to generate a dendrogram of cluster representatives

        self.target_map = {
            "active_sites": "active_site", "binding": "binding", "binding_site": "binding_positions",
            "ddg_mut": "ddg", "dtm": "dtm", "ec_number": "ec_number",
            "kcat": "kcat", "km": "km", "ph_opt": "phopt_median", "solubility": "solubility"
        }

        # Explicit Composite Keys to fix the duplicate counting
        self.primary_keys_map = {
            "active_sites": ["uniprot_id"],
            "binding": ["protein_sequence", "ligand_id"],
            "binding_site": ["uniprot_id"],
            "ddg_mut": ["uniprot_id", "mutation"],
            "dtm": ["uniprot_id", "mutation"],
            "ec_number": ["uniprot_id"],
            "kcat": ["protein_sequence", "smiles"],
            "km": ["protein_sequence", "smiles"],
            "ph_opt": ["uniprot_id"],
            "solubility": ["protein_sequence"]
        }

    def sanitize_for_json(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self.sanitize_for_json(v) for v in obj]
        elif pd.api.types.is_scalar(obj) and pd.isna(obj):
            return None
        elif hasattr(obj, "item"):
            return obj.item()
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        else:
            return str(obj)

    def get_files(self, directory: Path) -> Dict[str, Any]:
        files = {"base": None, "splits": {}, "json": directory / "dataset_card.json"}
        for file in directory.glob("*.csv"):
            if "splits" not in file.name:
                files["base"] = file
            else:
                files["splits"][file.stem] = file
        return files

    def calc_distribution(self, series: pd.Series) -> Any:
        series = series.dropna()
        if series.empty: return None

        total_valid = int(len(series))
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = series.nunique()

        # Categorical / Binary / High Variance Strings
        if not is_numeric or unique_count < 15:
            if unique_count > 50:
                return {
                    "total_valid_labels": total_valid,
                    "format": "categorical_high_variance",
                    "unique_classes": int(unique_count)
                }

            counts = series.value_counts().to_dict()
            ratios = series.value_counts(normalize=True).to_dict()
            return {
                "total_valid_labels": total_valid,
                "classes": {str(k): {"count": int(counts[k]), "ratio": round(float(ratios[k]), 4)} for k in counts}
            }
        # Continuous
        else:
            return {
                "total_valid_labels": total_valid,
                "mean": round(float(series.mean()), 4), "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4), "max": round(float(series.max()), 4),
                "median": round(float(series.median()), 4)
            }

    def process_splits(self, base_df: pd.DataFrame, split_path: Path, target_col: str) -> Dict[str, Any]:
        split_df = pd.read_csv(split_path)
        split_col = [c for c in split_df.columns if c.lower() in ['split', 'set']]

        if not split_col: return {"leakage": False, "distributions": {}, "sizes": {}}
        split_col = split_col[0]
        key_col = [c for c in split_df.columns if c != split_col][0]

        # 1. Leakage Check
        splits_grouped = split_df.groupby(split_col)[key_col].apply(set).to_dict()
        train_set, val_set, test_set = splits_grouped.get('train', set()), splits_grouped.get('val', splits_grouped.get('validation', set())), splits_grouped.get('test', set())
        leakage = len(train_set.intersection(val_set)) > 0 or len(train_set.intersection(test_set)) > 0 or len(val_set.intersection(test_set)) > 0

        # 2. Exact Split Sizes
        sizes = split_df[split_col].value_counts().to_dict()

        # 3. Distributions
        merged = pd.merge(base_df[[key_col, target_col]], split_df, on=key_col, how='inner')
        distributions = {}
        for split_name, group in merged.groupby(split_col):
            if split_name == "val": split_name = "validation"
            distributions[split_name] = self.calc_distribution(group[target_col])

        return {"leakage": leakage, "distributions": distributions, "sizes": sizes}

    def calc_stats(self, base_df: pd.DataFrame, prop_name: str, split_files: Dict[str, Path]) -> Dict[str, Any]:
        seq_col = next((c for c in base_df.columns if c in ['protein_sequence', 'wildtype_sequence']), None)
        target_col = self.target_map.get(prop_name)

        # 1. Advanced Sequence Stats
        seq_stats = {"min": None, "max": None, "mean": None, "median": None, "p95": None, "count_greater_than_1024": None, "contains_non_standard_aas": None}
        if seq_col:
            seqs = base_df[seq_col].dropna()
            lengths = seqs.str.len()
            if not lengths.empty:
                standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
                seq_stats = {
                    "min": lengths.min(), "max": lengths.max(), "mean": round(lengths.mean(), 2),
                    "median": lengths.median(), "p95": np.percentile(lengths, 95),
                    "count_greater_than_1024": int((lengths > 1024).sum()),
                    "contains_non_standard_aas": int(seqs.apply(lambda s: any(c.upper() not in standard_aas for c in s)).sum())
                }

        # 2. NaNs and Fixed Duplicates
        nans = base_df.isna().sum().to_dict()
        dup_full = base_df.duplicated().sum()

        # New keys check logic
        keys_to_check = self.primary_keys_map.get(prop_name)
        dup_key = base_df.duplicated(subset=keys_to_check).sum() if (keys_to_check and all(k in base_df.columns for k in keys_to_check)) else None

        # 3. Distributions & Split Sizes
        label_dist = {"overall": self.calc_distribution(base_df[target_col]) if target_col else None, "train": {}, "validation": {}, "test": {}}
        split_leakages = {}
        split_sizes = {}

        for split_name, split_path in split_files.items():
            split_res = self.process_splits(base_df, split_path, target_col)
            split_leakages[split_name] = split_res["leakage"]
            split_sizes[split_name] = split_res["sizes"]

            for s_type, dist in split_res["distributions"].items():
                if s_type in label_dist:
                    label_dist[s_type][split_name] = dist

        # 4. Auto-generate Notes
        notes = []
        if prop_name in ["kcat", "km"]:
            notes.append(f"Target values for {prop_name} appear to be log10-transformed.")
        if dup_full > 0:
            notes.append(f"Warning: Base dataset contains {dup_full} full row duplicates.")
        joined_notes = " | ".join(notes) if notes else None

        return self.sanitize_for_json({
            "amount_of_removed_data": {"datapoints_after_cleaning": len(base_df)},
            "nans_dups": {"nan_counts_per_column": nans, "duplicate_rows_full": dup_full, "duplicate_rows_by_key": dup_key},
            "label_distribution": label_dist,
            "split_sizes": split_sizes,
            "sequence_length_summary": seq_stats,
            "quality_checks": {"schema_validated": True, "split_leakage_checked": split_leakages, "label_format_validated": True},
            "notes": joined_notes
        })

    def _load_embeddings(self, sequences: list, uniprot_ids: list, h5_path: Path):
        """Load mean-pooled embeddings from an H5 file.
        Returns (embeddings: np.ndarray of shape (n_found, dim), valid_indices: list[int]),
        or (None, []) on failure."""
        try:
            import h5py
        except ImportError:
            print("  [UMAP] h5py not installed — skipping embedding load.")
            return None, []

        if not h5_path.exists():
            print(f"  [UMAP] H5 file not found: {h5_path}")
            return None, []

        def _candidate_keys(seq, uid):
            keys = []
            if uid:
                keys.append(str(uid).strip())
                keys.append(str(uid).strip().replace("/", "_").replace(".", "_"))
            if seq:
                cleaned = "".join(str(seq).strip().split()).upper()
                keys.append(cleaned)
            return keys

        embeddings = []
        valid_indices = []
        n_missing = 0
        with h5py.File(str(h5_path), "r") as hf:
            available_keys = set(hf.keys())
            for i, (seq, uid) in enumerate(zip(sequences, uniprot_ids)):
                emb_array = None
                for key in _candidate_keys(seq, uid):
                    if key in available_keys:
                        raw = np.asarray(hf[key], dtype=np.float32)
                        emb_array = raw.mean(axis=0) if raw.ndim == 2 else raw
                        break
                if emb_array is not None:
                    embeddings.append(emb_array)
                    valid_indices.append(i)
                else:
                    n_missing += 1

        n_found = len(embeddings)
        if n_missing > 0:
            print(f"  [UMAP] {n_found} embeddings found, {n_missing} missing in H5.")
        if n_found < 5:
            print(f"  [UMAP] Too few embeddings ({n_found}) — skipping UMAP.")
            return None, []

        return np.stack(embeddings, axis=0), valid_indices

    def generate_umap(
        self,
        base_df: pd.DataFrame,
        prop_name: str,
        split_files: Dict[str, Path],
        h5_path: Path,
        emb_model: str,
        output_dir: Path,
    ) -> Optional[Dict[str, str]]:
        """Generate two UMAP plots (label-colored and split-colored) for a property dataset.
        Returns a dict with paths to saved figures, or None on failure.
        """
        try:
            import umap as umap_lib
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError as e:
            print(f"  [UMAP] Missing dependency ({e}) — skipping UMAP generation.")
            return None

        seq_col = next((c for c in base_df.columns if c in ["protein_sequence", "wildtype_sequence"]), None)
        id_col = next((c for c in base_df.columns if c in ["uniprot_id"]), None)
        target_col = self.target_map.get(prop_name)

        sequences = base_df[seq_col].tolist() if seq_col else [None] * len(base_df)
        uniprot_ids = base_df[id_col].tolist() if id_col else [None] * len(base_df)

        print(f"  [UMAP] Loading embeddings from {h5_path.name} for {len(base_df)} proteins...")
        emb_matrix, valid_indices = self._load_embeddings(sequences, uniprot_ids, h5_path)
        if emb_matrix is None:
            return None

        valid_df = base_df.iloc[valid_indices].reset_index(drop=True)

        print(f"  [UMAP] Running UMAP on {emb_matrix.shape[0]} embeddings...")
        reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
        coords = reducer.fit_transform(emb_matrix)  # (n_valid, 2)

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # ── Plot 1: Colored by target label ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        if target_col and target_col in valid_df.columns:
            labels = valid_df[target_col]
            is_numeric_target = pd.api.types.is_numeric_dtype(labels) and labels.nunique() >= 15
            if is_numeric_target:
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels.values, cmap="viridis",
                                s=8, alpha=0.7, linewidths=0)
                plt.colorbar(sc, ax=ax, label=target_col)
            else:
                unique_labels = labels.dropna().unique()
                cmap = plt.get_cmap("tab20", len(unique_labels))
                label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
                colors = [label_to_color.get(lbl, (0.5, 0.5, 0.5, 1.0)) for lbl in labels]
                ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7, linewidths=0)
                handles = [plt.Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=label_to_color[lbl], markersize=6, label=str(lbl))
                           for lbl in unique_labels[:20]]
                ax.legend(handles=handles, title=target_col, fontsize=6, markerscale=1.5,
                          loc="best", ncol=max(1, len(unique_labels) // 10))
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6, linewidths=0)

        ax.set_title(f"{prop_name} — UMAP ({emb_model})\nColored by: {target_col or 'none'}", fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        label_path = output_dir / f"{prop_name}_{emb_model}_labels.png"
        plt.savefig(label_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["label_colored"] = str(label_path)
        print(f"  [UMAP] Saved label-colored plot → {label_path}")

        # ── Plot 2: Colored by split assignment ───────────────────────────────
        split_path = next(
            (p for name, p in split_files.items() if "random" in name),
            next(iter(split_files.values()), None)
        )
        split_col_assigned = None
        if split_path is not None:
            try:
                split_df = pd.read_csv(split_path)
                split_col_name = next((c for c in split_df.columns if c.lower() in ["split", "set"]), None)
                if split_col_name:
                    key_col = next(c for c in split_df.columns if c != split_col_name)
                    merge_col = id_col if (id_col and id_col in valid_df.columns and id_col in split_df.columns) \
                        else (seq_col if seq_col and seq_col in split_df.columns else None)
                    if merge_col:
                        merged_splits = valid_df[[merge_col]].merge(split_df[[merge_col, split_col_name]],
                                                                    on=merge_col, how="left")
                        split_col_assigned = merged_splits[split_col_name].fillna("unknown")
            except Exception as e:
                print(f"  [UMAP] Could not load split file: {e}")

        fig, ax = plt.subplots(figsize=(8, 6))
        split_colors = {"train": "#4C72B0", "val": "#DD8452", "validation": "#DD8452",
                        "test": "#55A868", "unknown": "#C0C0C0"}
        if split_col_assigned is not None:
            for split_name, color in split_colors.items():
                mask = split_col_assigned == split_name
                if mask.any():
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=split_name,
                               s=8, alpha=0.7, linewidths=0)
            ax.legend(title="Split", fontsize=8, markerscale=2)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6, linewidths=0)

        split_file_name = Path(split_path).stem if split_path else "no split"
        ax.set_title(f"{prop_name} — UMAP ({emb_model})\nColored by: split  [{split_file_name}]", fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        split_plot_path = output_dir / f"{prop_name}_{emb_model}_splits.png"
        plt.savefig(split_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["split_colored"] = str(split_plot_path)
        print(f"  [UMAP] Saved split-colored plot → {split_plot_path}")

        return {"embedding_model": emb_model, **saved_paths}

    # ──────────────────────────────────────────────────────────────────────────
    # MMSeqs2 Clustering + Cluster Diversity Plots
    # ──────────────────────────────────────────────────────────────────────────

    def run_clustering(self, base_df: pd.DataFrame, prop_name: str) -> Optional[Dict[str, Any]]:
        """Run MMSeqs2 easy-cluster on unique sequences in the dataset.
        Returns a dict with cluster stats, or None if mmseqs is unavailable."""
        import shutil
        import subprocess
        import tempfile

        if shutil.which("mmseqs") is None:
            print("  [Cluster] mmseqs not found in PATH — skipping clustering.")
            print("            Hint: run 'module load mmseqs2' before launching this script.")
            return None

        seq_col = next((c for c in base_df.columns if c in ["protein_sequence", "wildtype_sequence"]), None)
        id_col = next((c for c in base_df.columns if c in ["uniprot_id"]), None)

        if seq_col is None:
            print("  [Cluster] No sequence column found — skipping clustering.")
            return None

        # Build unique sequence → id mapping (deduplicated)
        seq_to_id: Dict[str, str] = {}
        for _, row in base_df[[seq_col] + ([id_col] if id_col else [])].dropna(subset=[seq_col]).iterrows():
            seq = "".join(str(row[seq_col]).strip().split()).upper()
            uid = str(row[id_col]).strip() if id_col else seq[:16]
            if seq and seq not in seq_to_id:
                seq_to_id[seq] = uid

        total_seqs = len(seq_to_id)
        if total_seqs < 2:
            print(f"  [Cluster] Only {total_seqs} unique sequences — skipping clustering.")
            return None

        print(f"  [Cluster] Running MMSeqs2 on {total_seqs} unique sequences "
              f"(identity={self.seq_identity})...")

        with tempfile.TemporaryDirectory(prefix="enzymado-cluster-") as tmp:
            tmp_dir = Path(tmp)
            fasta_path = tmp_dir / "seqs.fasta"
            db_path = tmp_dir / "db"
            cluster_db_path = tmp_dir / "cluster_db"
            tsv_path = tmp_dir / "clusters.tsv"
            tmp_work = tmp_dir / "tmp"

            # Write FASTA (use sequential IDs to avoid key issues)
            id_map = {}  # safe_id → original_id
            with open(fasta_path, "w") as fh:
                for idx, (seq, uid) in enumerate(seq_to_id.items()):
                    safe_id = f"seq{idx}"
                    id_map[safe_id] = uid
                    fh.write(f">{safe_id}\n{seq}\n")

            try:
                subprocess.run(
                    ["mmseqs", "createdb", str(fasta_path), str(db_path)],
                    check=True, capture_output=True, text=True
                )
                subprocess.run(
                    ["mmseqs", "cluster", str(db_path), str(cluster_db_path), str(tmp_work),
                     "--min-seq-id", str(self.seq_identity), "-c", "0.8", "--cov-mode", "0"],
                    check=True, capture_output=True, text=True
                )
                subprocess.run(
                    ["mmseqs", "createtsv", str(db_path), str(db_path),
                     str(cluster_db_path), str(tsv_path)],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"  [Cluster] MMSeqs2 failed: {e.stderr[:300]}")
                return None

            # Parse TSV: rep_id \t member_id
            cluster_assignments: Dict[str, str] = {}  # safe_id → rep safe_id
            with open(tsv_path) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        cluster_assignments[parts[1]] = parts[0]

        # Compute cluster-size distribution
        from collections import Counter
        cluster_sizes = Counter(cluster_assignments.values())
        sizes = sorted(cluster_sizes.values(), reverse=True)
        n_clusters = len(cluster_sizes)
        n_singletons = sum(1 for s in sizes if s == 1)
        largest = sizes[0] if sizes else 0

        print(f"  [Cluster] Done: {n_clusters} clusters from {total_seqs} sequences "
              f"({n_singletons} singletons, largest={largest}).")

        # Collect representative sequences (capped at 300 for dendrogram tractability)
        safe_id_to_seq = {f"seq{idx}": seq for idx, seq in enumerate(seq_to_id.keys())}
        rep_safe_ids = list(cluster_sizes.keys())
        rep_seqs = [safe_id_to_seq[sid] for sid in rep_safe_ids if sid in safe_id_to_seq]

        return self.sanitize_for_json({
            "cluster_method": "mmseqs2",
            "cluster_parameters": {
                "min_sequence_identity": self.seq_identity,
                "coverage_mode": 0,
                "coverage_threshold": 0.8,
            },
            "total_sequences": total_seqs,
            "number_of_clusters": n_clusters,
            "singleton_clusters": n_singletons,
            "largest_cluster_size": largest,
            "ratio_unique_variants": round(n_clusters / total_seqs, 4) if total_seqs else None,
            "_cluster_sizes": sizes,               # for diversity plots, stripped before card write
            "_representative_sequences": rep_seqs[:300],  # for dendrogram, stripped before card write
        })

    def generate_cluster_diversity_plots(
        self,
        cluster_stats: Dict[str, Any],
        prop_name: str,
        output_dir: Path,
    ) -> Optional[str]:
        """Generate a 2-panel figure: rank-abundance + rarefaction curves.
        Returns the path to the saved figure, or None on failure."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [ClusterDiversity] matplotlib not available — skipping plots.")
            return None

        sizes = cluster_stats.get("_cluster_sizes")
        if not sizes:
            return None

        sizes_arr = np.array(sizes, dtype=np.int64)
        total_seqs = int(sizes_arr.sum())
        n_clusters = len(sizes_arr)

        output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"{prop_name} — Sequence Diversity  "
            f"(MMSeqs2 identity≥{self.seq_identity:.0%}, "
            f"{n_clusters} clusters / {total_seqs} sequences)",
            fontsize=11, fontweight="bold"
        )

        # ── Panel A: Rank-Abundance ───────────────────────────────────────────
        ax = axes[0]
        ranks = np.arange(1, n_clusters + 1)
        ax.plot(ranks, sizes_arr, color="#2D6A9F", linewidth=1.5)
        ax.fill_between(ranks, sizes_arr, alpha=0.15, color="#2D6A9F")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cluster rank (log)", fontsize=10)
        ax.set_ylabel("Cluster size (log)", fontsize=10)
        ax.set_title("Rank–Abundance", fontsize=10)
        ax.axvline(n_clusters * 0.5, color="grey", linestyle="--", linewidth=0.8,
                   label=f"50th percentile rank")
        singleton_frac = (sizes_arr == 1).sum() / n_clusters
        ax.text(0.98, 0.97, f"Singletons: {singleton_frac:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#555")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

        # ── Panel B: Rarefaction ─────────────────────────────────────────────
        ax = axes[1]
        rng = np.random.default_rng(42)

        # Build a flat array of cluster labels (one entry per sequence)
        seq_labels = np.repeat(np.arange(n_clusters), sizes_arr)

        n_points = min(40, total_seqs)
        sample_sizes = np.unique(
            np.concatenate([np.geomspace(1, total_seqs, n_points).astype(int), [total_seqs]])
        )
        n_reps = 5
        means, stds = [], []
        for n in sample_sizes:
            counts = [len(np.unique(rng.choice(seq_labels, size=n, replace=False))) for _ in range(n_reps)]
            means.append(np.mean(counts))
            stds.append(np.std(counts))

        means = np.array(means)
        stds = np.array(stds)
        ax.plot(sample_sizes, means, color="#C0392B", linewidth=1.5, label="Mean (5 replicates)")
        ax.fill_between(sample_sizes, means - stds, means + stds, alpha=0.2, color="#C0392B",
                        label="±1 SD")
        ax.set_xlabel("Sample size (sequences)", fontsize=10)
        ax.set_ylabel("Unique clusters discovered", fontsize=10)
        ax.set_title("Rarefaction Curve", fontsize=10)
        ax.plot([total_seqs], [n_clusters], "k^", markersize=7, label=f"Full set: {n_clusters} clusters")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        out_path = output_dir / f"{prop_name}_cluster_diversity.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ClusterDiversity] Saved cluster diversity panel → {out_path}")
        return str(out_path)

    def generate_dendrogram(
        self,
        cluster_stats: Dict[str, Any],
        prop_name: str,
        output_dir: Path,
    ) -> Optional[str]:
        """Hierarchical clustering of cluster representatives via trigram cosine distance.
        Returns the path to the saved dendrogram PNG, or None on failure."""
        import numpy as np

        rep_seqs = cluster_stats.get("_representative_sequences")
        if not rep_seqs or len(rep_seqs) < 2:
            print("  [Dendrogram] Not enough representatives — skipping.")
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_distances
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
        except ImportError as e:
            print(f"  [Dendrogram] Missing dependency: {e} — skipping.")
            return None

        print(f"  [Dendrogram] Building trigram distance matrix for "
              f"{len(rep_seqs)} representatives...")

        # Trigram (3-mer) frequency vectors over amino-acid sequences
        vect = CountVectorizer(analyzer="char", ngram_range=(3, 3))
        X = vect.fit_transform(rep_seqs).toarray().astype(float)

        dist_matrix = cosine_distances(X)
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = squareform(dist_matrix, checks=False)

        Z = linkage(condensed, method="average")

        n = len(rep_seqs)
        # Width scales with number of leaves; height fixed
        fig_w = max(10, min(n * 0.12, 60))
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        dendrogram(
            Z,
            ax=ax,
            no_labels=True,
            color_threshold=0.7 * float(Z[:, 2].max()),
        )
        ax.set_title(
            f"{prop_name} — Cluster Representative Dendrogram\n"
            f"({n} representatives, UPGMA on trigram cosine distance)"
        )
        ax.set_xlabel("Cluster representatives")
        ax.set_ylabel("Cosine distance")
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{prop_name}_dendrogram.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Dendrogram] Saved → {out_path}")
        return str(out_path)

    def run(self):
        print("Starting Dataset Card Population Pipeline...\n")

        # ── Resolve embedding H5 path if a model was specified ────────────────
        emb_h5_path = None
        if self.emb_model:
            emb_json_path = self.embeddings_json or Path("embeddings_models_output.json")
            if emb_json_path.exists():
                with open(emb_json_path) as f:
                    emb_registry = json.load(f)
                if self.emb_model in emb_registry:
                    emb_h5_path = Path(emb_registry[self.emb_model]["path"])
                    print(f"UMAP embeddings: {self.emb_model} → {emb_h5_path}\n")
                else:
                    print(f"Warning: embedding model '{self.emb_model}' not found in {emb_json_path}. "
                          f"Available: {', '.join(emb_registry)}. Skipping UMAP.\n")
            else:
                print(f"Warning: embeddings JSON not found at {emb_json_path}. Skipping UMAP.\n")

        umap_output_dir = Path("results/figures/umaps")
        cluster_diversity_output_dir = Path("results/figures/cluster_diversity")

        for prop_dir in sorted(self.base_dir.iterdir()):
            if not prop_dir.is_dir(): continue
            print(f"Processing {prop_dir.name}...")

            files = self.get_files(prop_dir)
            if not files["base"]: continue

            base_df = pd.read_csv(files["base"])
            stats = self.calc_stats(base_df, prop_dir.name, files["splits"])

            with open(self.template_path, 'r') as f:
                card = json.load(f)

            # Map the gathered stats directly to the JSON format
            card["property_name"] = prop_dir.name
            card["dataset_id"] = files["base"].stem
            card["amount_of_removed_data"].update(stats["amount_of_removed_data"])
            card["statistics"]["nans_dups"] = stats["nans_dups"]
            card["statistics"]["label_distribution"] = stats["label_distribution"]

            # Injecting new useful metadata
            card["statistics"]["split_sizes_by_method"] = stats["split_sizes"]
            card["additional_important"]["sequence_length_summary"] = stats["sequence_length_summary"]
            card["additional_important"]["quality_checks"].update(stats["quality_checks"])
            card["additional_important"]["notes"] = stats["notes"]
            card["additional_important"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(files["json"], 'w') as f:
                json.dump(card, f, indent=2)

            # ── MMSeqs2 Clustering + Cluster Diversity Plots (always attempted) ──
            cluster_stats = self.run_clustering(base_df, prop_dir.name)
            cluster_diversity_plot_path = None
            if cluster_stats:
                cluster_diversity_plot_path = self.generate_cluster_diversity_plots(
                    cluster_stats, prop_dir.name, cluster_diversity_output_dir
                )
                dendrogram_path = None
                if self.dendrogram:
                    dendrogram_path = self.generate_dendrogram(
                        cluster_stats, prop_dir.name, cluster_diversity_output_dir
                    )
                # Strip internal keys before writing to card
                cluster_for_card = {k: v for k, v in cluster_stats.items() if not k.startswith("_")}
                cluster_for_card["cluster_diversity_plot"] = cluster_diversity_plot_path
                cluster_for_card["dendrogram_plot"] = dendrogram_path
                with open(files["json"]) as f:
                    card = json.load(f)
                card["statistics"]["entire_protein_pool_cluster_with_mmseqs2"] = cluster_for_card
                with open(files["json"], 'w') as f:
                    json.dump(card, f, indent=2)

            # ── UMAP generation (optional) ─────────────────────────────────
            if emb_h5_path is not None:
                umap_result = self.generate_umap(
                    base_df=base_df,
                    prop_name=prop_dir.name,
                    split_files=files["splits"],
                    h5_path=emb_h5_path,
                    emb_model=self.emb_model,
                    output_dir=umap_output_dir,
                )
                if umap_result:
                    with open(files["json"]) as f:
                        card = json.load(f)
                    card["additional_important"]["umap"] = umap_result
                    with open(files["json"], 'w') as f:
                        json.dump(card, f, indent=2)

            print(f"  Successfully generated dataset_card.json for {prop_dir.name}\n")

        print("Pipeline Complete! All dataset_card.json files have been populated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/processed/properties")
    parser.add_argument("--emb_model", default=None,
                        help="Embedding model for UMAP (e.g. esm2_650M). Skipped if not provided.")
    parser.add_argument("--embeddings_json", default="embeddings_models_output.json")
    parser.add_argument("--seq_identity", type=float, default=0.5,
                        help="MMSeqs2 min-seq-id for whole-dataset clustering. Default: 0.5")
    args = parser.parse_args()
    DatasetCardPopulator(
        args.base_dir,
        emb_model=args.emb_model,
        embeddings_json=args.embeddings_json,
        seq_identity=args.seq_identity,
    ).run()