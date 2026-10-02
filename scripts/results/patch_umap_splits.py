#!/usr/bin/env python3
"""
Patch the 'split' column in existing UMAP parquet files without rerunning UMAP.

For each {prop}_{emb}.parquet, reconstructs valid_indices by consulting the
safetensors key index (fast — no tensor data loaded), then re-runs the
split-merge with drop_duplicates to fix composite-key properties like
binding, kcat, km, ddg_mut.
"""

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path("data/processed/properties")
EMB_DIR = Path("data/embeddings_mean_pooled/protein_sequences_unique")
UMAP_DIR = Path("results/figures/umaps")
SEQ_CSV = Path("data/processed/protein_pool.csv")

PROPERTIES = [
    "active_sites",
    "binding",
    "binding_site",
    "ddg_mut",
    "dtm",
    "ec_number",
    "kcat",
    "km",
    "ph_opt",
    "solubility",
]


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(50).startswith(b"version https://git-lfs.github.com")
    except OSError:
        return False


def get_base_and_split_files(prop_dir: Path):
    base_csv = None
    split_files = {}
    for f in sorted(prop_dir.glob("*.csv")):
        if is_lfs_pointer(f):
            continue
        if "standardized" in f.name:
            split_files[f.stem] = f
        elif base_csv is None:
            base_csv = f
    return base_csv, split_files


def get_valid_indices(
    base_df: pd.DataFrame, available_keys: set, seq_col, id_col, seq_to_key: dict = None
) -> list:
    """Mirror _load_embeddings logic: try seq→key map, then direct seq, then uid."""
    sequences = base_df[seq_col].tolist() if seq_col else [None] * len(base_df)
    uniprot_ids = base_df[id_col].tolist() if id_col else [None] * len(base_df)

    valid_indices = []
    for i, (seq, uid) in enumerate(zip(sequences, uniprot_ids)):
        key = None
        if seq is not None:
            cleaned = "".join(str(seq).strip().split()).upper()
            if seq_to_key:
                mapped = seq_to_key.get(cleaned)
                if mapped and mapped in available_keys:
                    key = mapped
            if key is None and cleaned in available_keys:
                key = cleaned
        if key is None and uid is not None and str(uid).strip() in available_keys:
            key = str(uid).strip()
        if key is not None:
            valid_indices.append(i)
    return valid_indices


def compute_split_series(
    valid_df: pd.DataFrame, split_files: dict, seq_col, id_col
) -> pd.Series:
    """Re-run split assignment with drop_duplicates for composite keys."""
    split_path = next(
        (p for name, p in split_files.items() if "random" in name),
        next(iter(split_files.values()), None),
    )
    if split_path is None:
        return pd.Series(["unknown"] * len(valid_df))

    try:
        split_df = pd.read_csv(split_path)
        split_col = next(
            (c for c in split_df.columns if c.lower() in ["split", "set"]), None
        )
        if not split_col:
            return pd.Series(["unknown"] * len(valid_df))

        # Prefer protein_sequence (unique per mutation), then id_col, then seq_col
        merge_col = None
        for cand in ["protein_sequence", id_col, seq_col]:
            if cand and cand in valid_df.columns and cand in split_df.columns:
                merge_col = cand
                break

        if not merge_col:
            return pd.Series(["unknown"] * len(valid_df))

        split_lookup = split_df[[merge_col, split_col]].drop_duplicates(
            subset=[merge_col]
        )
        merged = valid_df[[merge_col]].merge(split_lookup, on=merge_col, how="left")
        return merged[split_col].fillna("unknown").reset_index(drop=True)

    except Exception as e:  # noqa: BLE001 - preserve the legacy unknown fallback.
        print(f"    Warning during split merge: {e}")
        return pd.Series(["unknown"] * len(valid_df))


def patch_parquet(parquet_path: Path, prop: str, emb: str, seq_to_key: dict) -> bool:
    st_path = EMB_DIR / f"{emb}.safetensors"
    prop_dir = BASE_DIR / prop

    if not st_path.exists():
        print(f"  [skip] safetensors not found: {st_path.name}")
        return False
    if not prop_dir.exists():
        print(f"  [skip] property dir not found: {prop_dir}")
        return False

    base_csv, split_files = get_base_and_split_files(prop_dir)
    if base_csv is None:
        print(f"  [skip] no base CSV in {prop_dir}")
        return False

    base_df = pd.read_csv(base_csv)
    # Prefer protein_sequence (handles mutant sequences in dtm/ddg_mut)
    seq_col = next(
        (c for c in ["protein_sequence", "wildtype_sequence"] if c in base_df.columns),
        None,
    )
    id_col = next((c for c in base_df.columns if c == "uniprot_id"), None)

    try:
        import safetensors

        with safetensors.safe_open(str(st_path), framework="pt") as f:
            available_keys = set(f.keys())
    except Exception as e:  # noqa: BLE001 - skip unreadable legacy artifacts.
        print(f"  [skip] cannot read safetensors: {e}")
        return False

    valid_indices = get_valid_indices(
        base_df, available_keys, seq_col, id_col, seq_to_key
    )
    valid_df = base_df.iloc[valid_indices].reset_index(drop=True)

    pq = pd.read_parquet(parquet_path)

    if len(pq) != len(valid_df):
        print(
            f"  [warn] row count mismatch: parquet={len(pq)}, "
            f"reconstructed={len(valid_df)} — skipping"
        )
        return False

    split_series = compute_split_series(valid_df, split_files, seq_col, id_col)

    old_unknown = int((pq["split"] == "unknown").sum())
    pq["split"] = split_series.values
    new_unknown = int((pq["split"] == "unknown").sum())

    pq.to_parquet(parquet_path, index=False)
    delta = old_unknown - new_unknown
    tag = f"+{delta} resolved" if delta > 0 else "no change"
    print(f"  unknown: {old_unknown} → {new_unknown}  ({tag})")
    return True


def parse_prop_emb(stem: str, properties: list):
    """Extract (prop, emb) from a stem like 'active_sites_esm2_3B'."""
    for prop in sorted(properties, key=len, reverse=True):
        if stem.startswith(prop + "_"):
            emb = stem[len(prop) + 1 :]
            return prop, emb
    return None, None


def main():
    parquets = sorted(UMAP_DIR.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files found in {UMAP_DIR}")
        sys.exit(1)

    # Load sequence→key map once for all parquets
    seq_to_key: dict = {}
    if SEQ_CSV.exists():
        seq_df = pd.read_csv(SEQ_CSV)
        seq_to_key = {
            "".join(str(s).strip().split()).upper(): str(k)
            for s, k in zip(seq_df["protein_sequence"], seq_df["sequence_id"])
            if pd.notna(s) and pd.notna(k)
        }
        print(f"Loaded {len(seq_to_key):,} sequence→key mappings\n")

    print(f"Found {len(parquets)} parquet files to patch.\n")
    ok = fail = skipped = 0

    for pq_path in parquets:
        prop, emb = parse_prop_emb(pq_path.stem, PROPERTIES)
        if prop is None:
            print(f"{pq_path.name}  [skip] could not parse prop/emb")
            skipped += 1
            continue

        print(f"{pq_path.name}  (prop={prop}, emb={emb})")
        success = patch_parquet(pq_path, prop, emb, seq_to_key)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone. patched={ok}  skipped/failed={fail + skipped}")


if __name__ == "__main__":
    main()
