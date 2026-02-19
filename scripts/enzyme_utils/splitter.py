# splits.py

import dataclasses
from pathlib import Path
import shutil
import subprocess
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import sklearn.metrics
import scipy.stats


# =========================
#   SplitTuple
# =========================

@dataclasses.dataclass
class SplitTuple:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        """Raises error if the split parts overlap."""
        for a, b in [('train', 'val'), ('val', 'test'), ('test', 'train')]:
            ai = np.asarray(getattr(self, a))
            bi = np.asarray(getattr(self, b))
            overlap = np.intersect1d(ai, bi)
            if overlap.size:
                raise ValueError(f"Found {overlap.size} overlapping items between {a} and {b}.")

        self.train = np.asarray(self.train, dtype=int)
        self.val   = np.asarray(self.val,   dtype=int)
        self.test  = np.asarray(self.test,  dtype=int)

    def __len__(self) -> int:
        return int(self.train.size + self.val.size + self.test.size)

    def trainval(self) -> np.ndarray:
        return np.concatenate([self.train, self.val], axis=0)

    def indices(self) -> np.ndarray:
        return np.concatenate([self.train, self.val, self.test], axis=0)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_lists(cls, train, val, test) -> "SplitTuple":
        return cls(train=np.asarray(train), val=np.asarray(val), test=np.asarray(test))


# =========================
#   Evaluation Metrics
# =========================

def evaluate(y_true, y_pred):
    """
    Calculates Pearson r, R^2, Kendall tau, and MAE.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    pearson_r, _   = scipy.stats.pearsonr(y_true, y_pred)
    kendall_tau, _ = scipy.stats.kendalltau(y_true, y_pred)
    r_squared      = sklearn.metrics.r2_score(y_true, y_pred)
    mae            = sklearn.metrics.mean_absolute_error(y_true, y_pred)

    return {
        'pearson_r': pearson_r,
        'R^2': r_squared,
        'kendall_tau': kendall_tau,
        'mae': mae,
    }


# =========================
#   Cluster-based splits (MMseqs2)
# =========================

def divide_per_cluster(
    df_complete: pd.DataFrame,
    col_prot: str,
    col_sequence: str,
    directory_out: Path | None = None,
    tresh_identity: float = 0.8,
    n_splits: int = 10,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> dict[str, SplitTuple]:
    """
    Divides a dataframe based on protein clusters using MMseqs2.
    """
    print("Initiating splitting of data based on protein clusters...")

    df_unique = df_complete[[col_prot, col_sequence]].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(df_unique)} sequences for clustering.")

    base_temp_path = directory_out if directory_out is not None else Path("./")
    dir_mmseqs = base_temp_path / "mmseqs_temp"
    dir_mmseqs.mkdir(parents=True, exist_ok=True)

    fasta_path = dir_mmseqs / "sequences.fasta"
    db_path = dir_mmseqs / "DB"
    cluster_path = dir_mmseqs / "ClusterDB"
    results_tsv_path = dir_mmseqs / "clusters.tsv"

    # Write FASTA for MMseqs2
    with open(fasta_path, "w") as f:
        for _, row in df_unique.iterrows():
            f.write(f">{row[col_prot]}\n{row[col_sequence]}\n")

    try:
        subprocess.run(["mmseqs", "createdb", str(fasta_path), str(db_path)],
                       check=True, capture_output=True, text=True)
        subprocess.run([
            "mmseqs", "cluster", str(db_path), str(cluster_path), str(dir_mmseqs / "tmp"),
            "--min-seq-id", str(tresh_identity)
        ], check=True, capture_output=True, text=True)
        subprocess.run(
            ["mmseqs", "createtsv", str(db_path), str(db_path), str(cluster_path), str(results_tsv_path)],
            check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        raise SystemExit("ERROR: Couldn't find MMseqs2. Please make sure it is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ERROR: MMseqs2 Failed.\nOutput error:\n{e.stderr}")

    df_clusters = pd.read_csv(results_tsv_path, sep="\t", header=None, names=['cluster_id', col_prot])
    map_cluster = df_clusters.set_index(col_prot)['cluster_id'].to_dict()
    df_complete['cluster_id'] = df_complete[col_prot].map(map_cluster)

    print(f"Clustering done. Identified {df_complete['cluster_id'].nunique()} clusters.")

    split_dict = {}
    cluster_ids = df_complete['cluster_id'].unique()

    for i in range(n_splits):
        current_seed = seed + i

        train_val_clusters, test_clusters = train_test_split(
            cluster_ids, test_size=frac_test, random_state=current_seed
        )
        adjusted_val_frac = frac_val / (1 - frac_test) if (1 - frac_test) > 0 else 0
        train_clusters, val_clusters = train_test_split(
            train_val_clusters, test_size=adjusted_val_frac, random_state=current_seed
        )

        train_indices = df_complete[df_complete['cluster_id'].isin(train_clusters)].index.to_numpy()
        val_indices   = df_complete[df_complete['cluster_id'].isin(val_clusters)].index.to_numpy()
        test_indices  = df_complete[df_complete['cluster_id'].isin(test_clusters)].index.to_numpy()

        split_key = f'Cluster_{i+1}'
        split_dict[split_key] = SplitTuple(
            train=train_indices,
            val=val_indices,
            test=test_indices
        )
        print(f"  - Generated {split_key}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    if directory_out:
        print("\nSaving example split to CSV...")
        df_complete['split'] = 'unassigned'
        split_0 = list(split_dict.values())[0]
        df_complete.loc[split_0.train, 'split'] = 'train'
        df_complete.loc[split_0.val, 'split']   = 'val'
        df_complete.loc[split_0.test, 'split']  = 'test'

        df_final = df_complete.drop(columns=['cluster_id', col_sequence])
        base_name = f"dataset_split_id{tresh_identity}"
        out_path = directory_out / f"{base_name}_complete_example.csv"
        df_final.to_csv(out_path, index=False)
        print(f"Saved example split file to {out_path}")

    temp_dir_for_mmseqs = directory_out / "mmseqs_temp" if directory_out else Path("./mmseqs_temp")
    if temp_dir_for_mmseqs.exists():
        shutil.rmtree(temp_dir_for_mmseqs)

    print("\nDONE! Returning split dictionary.")
    return split_dict


# =========================
#   Multilabel stratified splits
# =========================

def generate_mls_splits(
    df_complete: pd.DataFrame,
    col_cluster: str,
    col_y: str,
    n_bins_y: int = 5,
    n_splits: int = 10,
    frac_test: float = 0.2,
    frac_val: float = 0.1,
    seed: int = 42,
) -> dict[str, SplitTuple]:
    """
    Splits using MultilabelStratifiedShuffleSplit on clusters + y.
    """
    print(f"Generating {n_splits} splits (Test: {frac_test:.0%}, Val: {frac_val:.0%})...")

    cluster_ids = pd.factorize(df_complete[col_cluster])[0]
    n_clusters = len(np.unique(cluster_ids))
    cluster_one_hot = np.zeros((len(df_complete), n_clusters), dtype=bool)
    cluster_one_hot[np.arange(len(df_complete)), cluster_ids] = 1

    y_binned = pd.qcut(df_complete[col_y], q=n_bins_y, labels=False, duplicates='drop')
    y_one_hot = pd.get_dummies(y_binned).to_numpy(dtype=bool)

    split_labels = np.hstack([cluster_one_hot, y_one_hot])
    print(f"Created stratification matrix with shape {split_labels.shape}")

    split_dict = {}
    indices = np.arange(len(df_complete))

    for i in range(n_splits):
        current_seed = seed + i

        first_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=frac_test,
            random_state=current_seed
        )
        trainval_idx, test_idx = next(first_splitter.split(indices, split_labels))

        relative_val_size = frac_val / (1.0 - frac_test) if (1.0 - frac_test) > 0 else 0

        second_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_val_size,
            random_state=current_seed
        )
        train_idx_rel, val_idx_rel = next(second_splitter.split(trainval_idx, split_labels[trainval_idx]))

        train_idx = trainval_idx[train_idx_rel]
        val_idx   = trainval_idx[val_idx_rel]

        split_key = f'Stratified_{i+1}'
        split_dict[split_key] = SplitTuple(
            train=train_idx,
            val=val_idx,
            test=test_idx
        )
        print(f"  - Generated {split_key}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    print("\nDONE! Returning Multilabel Stratified split dictionary.")
    return split_dict


def loco_specific(
    df: pd.DataFrame,
    *,
    col_cluster: str = "embedding_cluster_id",
    cluster_ids=None,              # list/tuple of cluster ids
    frac_val: float = 0.10,        # fraction of train+val that goes to val
    random_state: int = 0,
    col_y: str | None = None,      # for stratification (binary or continuous)
    n_bins_y: int = 5,
):
    """
    LOCO on a subset of clusters:
      - each cluster from cluster_ids will be a test set in ONE split.
      - the rest of those clusters are split into train/val.
    """
    if not cluster_ids:
        raise ValueError("You must pass cluster_ids (e.g., [4, 5]).")
    cluster_ids = list(cluster_ids)

    # universe = only rows from these clusters
    mask_in = df[col_cluster].isin(cluster_ids)
    idx_in  = pd.Index(df.index[mask_in])

    # prepare y for stratification within this universe
    if col_y is not None:
        y_sub = df.loc[idx_in, col_y]
        if y_sub.dropna().nunique() == 2:
            # binary
            y_strat = y_sub
        else:
            # continuous -> quantiles
            y_strat = pd.qcut(y_sub, q=n_bins_y, labels=False, duplicates="drop")
    else:
        y_strat = None

    splits = {}
    for j, test_c in enumerate(cluster_ids, start=1):
        # test = all rows of that cluster within the universe
        test_idx = pd.Index(df.index[(df[col_cluster] == test_c) & mask_in])
        train_val_idx = idx_in.difference(test_idx)

        # val size within (train+val)
        n_val = max(1, int(round(frac_val * len(train_val_idx))))
        if y_strat is not None:
            _, val_idx = train_test_split(
                train_val_idx,
                test_size=n_val,  # absolute integer
                random_state=random_state + j,
                stratify=y_strat.loc[train_val_idx]
            )
            val_idx = pd.Index(val_idx)
        else:
            rng = np.random.default_rng(random_state + j)
            val_idx = pd.Index(rng.choice(train_val_idx, size=n_val, replace=False))

        train_idx = train_val_idx.difference(val_idx)

        key = f"LOCO_{'-'.join(map(str, cluster_ids))}_test{test_c}"
        splits[key] = SplitTuple(
            train=train_idx.to_numpy(),
            val=val_idx.to_numpy(),
            test=test_idx.to_numpy()
        )

    return splits


# =========================
#   LOPO by protein (Leave-One-Protein-Out)
# =========================

def lopo_splits(df: pd.DataFrame, col_protein: str, n_splits: int, random_state: int = 0):
    """
    Generates n_splits where:
      - 1 protein → test
      - 1 protein → val
      - rest      → train

    Constraints:
      - no protein is repeated in test.
      - no protein is repeated in val.
    """
    proteins = df[col_protein].unique().tolist()
    n_proteins = len(proteins)
    rng = np.random.default_rng(random_state)

    if n_splits is None or n_splits > n_proteins:
        raise ValueError(f"n_splits must be <= number of proteins ({n_proteins}).")

    shuffled = proteins.copy()
    rng.shuffle(shuffled)

    if n_splits == n_proteins:
        # cycle: each protein will be test once, and val is the next one
        pairs = [(shuffled[i], shuffled[(i + 1) % n_proteins]) for i in range(n_proteins)]
    else:
        subset = shuffled[:2 * n_splits]
        test_proteins = subset[:n_splits]
        val_proteins  = subset[n_splits:2 * n_splits]
        pairs = list(zip(test_proteins, val_proteins))

    splits = {}
    for i, (test_p, val_p) in enumerate(pairs, start=1):
        test_idx = df.index[df[col_protein] == test_p].to_numpy()
        val_idx  = df.index[df[col_protein] == val_p].to_numpy()
        train_idx = df.index[~df[col_protein].isin([test_p, val_p])].to_numpy()

        key = f"split{i}_test({test_p})_val({val_p})"
        splits[key] = SplitTuple(train=train_idx, val=val_idx, test=test_idx)

    return splits


# =========================
#   LOCO by cluster pairs (whole cluster for test / val)
# =========================

def loco_clusters_pair(
    df: pd.DataFrame,
    *,
    col_cluster: str = "embedding_cluster_id",
    cluster_pairs=None,  # list of tuples (test_cluster, val_cluster)
):
    """
    LOCO with full clusters:
      - one whole cluster = test
      - another whole cluster = val
      - rest = train
    """
    if not cluster_pairs:
        raise ValueError("Must pass cluster_pairs, e.g.: [(4,5), (2,3)].")

    splits = {}

    for (test_c, val_c) in cluster_pairs:
        test_idx = df.index[df[col_cluster] == test_c].to_numpy()
        val_idx  = df.index[df[col_cluster] == val_c].to_numpy()
        train_idx = df.index[~df[col_cluster].isin([test_c, val_c])].to_numpy()

        key = f"LOCO_test{test_c}_val{val_c}"
        splits[key] = SplitTuple(train=train_idx, val=val_idx, test=test_idx)

    return splits


# =========================
#   Combined LOCO + LOPO
# =========================

def loco_lopo(
    df: pd.DataFrame,
    *,
    col_cluster: str = "embedding_cluster_id",
    col_protein: str = "protein_name",
    max_splits: int | None = None,
    random_state: int = 0,
    unique_across_roles: bool = False,   # if True, an enzyme is not repeated in val/test globally
):
    """
    Generates LOCO+LOPO splits:

      - test:  1 FULL protein (from a cluster X)
      - val:   1 FULL protein (from a different cluster Y)
      - train: everything else

    Guarantees:
      - a protein is not repeated as TEST in different splits.
      - a protein is not repeated as VAL in different splits.
      - if unique_across_roles=True, a protein does not appear in both test and val more than once.
    """
    rng = np.random.default_rng(random_state)

    cluster_to_proteins = df.groupby(col_cluster)[col_protein].unique().to_dict()

    combos = []
    for test_c, val_c in product(cluster_to_proteins.keys(), repeat=2):
        if test_c == val_c:
            continue
        for test_p in cluster_to_proteins[test_c]:
            for val_p in cluster_to_proteins[val_c]:
                if unique_across_roles and test_p == val_p:
                    continue
                combos.append((test_c, test_p, val_c, val_p))

    order = rng.permutation(len(combos))
    used_test: set[str] = set()
    used_val: set[str] = set()

    chosen = []
    for idx in order:
        test_c, test_p, val_c, val_p = combos[idx]

        if test_p in used_test:
            continue
        if val_p in used_val:
            continue

        if unique_across_roles and (test_p in used_val or val_p in used_test):
            continue

        chosen.append((test_c, test_p, val_c, val_p))
        used_test.add(test_p)
        used_val.add(val_p)

        if max_splits is not None and len(chosen) >= max_splits:
            break

    splits = {}
    for i, (test_c, test_p, val_c, val_p) in enumerate(chosen, start=1):
        test_idx = df.index[df[col_protein] == test_p].to_numpy()
        val_idx  = df.index[df[col_protein] == val_p].to_numpy()
        train_idx = df.index[~df[col_protein].isin([test_p, val_p])].to_numpy()

        key = f"split{i}_test({test_p}|c{test_c})_val({val_p}|c{val_c})"
        splits[key] = SplitTuple(train=train_idx, val=val_idx, test=test_idx)

    return splits
