"""
Build a JSON with the relative path (from repo root) and embedding dimension
for each model in data/embeddings/protein_sequences_unique/.

Output: data/embeddings/protein_sequences_unique/embeddings_metadata.json

Usage (from repo root or notebooks/):
    python notebooks/from_h5_to_json.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR   = REPO_ROOT / "data/embeddings/protein_sequences_unique"
OUT_JSON  = REPO_ROOT / "embeddings_models_output.json"

# Known embedding dimensions (second axis of per-residue output)
MODEL_DIMS = {
    "prot5_xl_half_uniref50": 1024,
    "prot5_xl_uniref50":      1024,
    "prost5":                 1024,
    "esm2_650M":              1280,
    "esm2_3B":                2560,
    "esm3_open":              1536,
    "esmc_300m":               960,
    "esmc_600m":              1152,
    "ankh_base":               768,
    "ankh_large":             1536,
    "ankh3_large":            1536,
    "ankh3_xl":               2560,
    "profluent_e1_300m":      1024,
    "profluent_e1_600m":      1280,
}

metadata = {}

for h5_path in sorted(EMB_DIR.glob("*.h5")):
    model_key = h5_path.stem
    if model_key not in MODEL_DIMS:
        print(f"[WARN] {h5_path.name}: no known dimension, skipping")
        continue
    metadata[model_key] = {
        "path": str(h5_path.relative_to(REPO_ROOT)),
        "dim":  MODEL_DIMS[model_key],
    }
    print(f"{h5_path.name}: dim={MODEL_DIMS[model_key]}")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nWrote metadata for {len(metadata)} models to {OUT_JSON.relative_to(REPO_ROOT)}")
