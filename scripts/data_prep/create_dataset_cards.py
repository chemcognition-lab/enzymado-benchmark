import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.dataset_card.dataset_card_populator import DatasetCardPopulator

# Run script from project root (enzymado-benchmark/)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate dataset_card.json files for all properties.")
    parser.add_argument(
        "--base_dir", default="data/processed/properties",
        help="Base directory containing one subdirectory per property. Default: data/processed/properties"
    )
    parser.add_argument(
        "--emb_model", default=None,
        help=(
            "Name of the embedding model to use for UMAP generation "
            "(e.g. esm2_650M, ankh_large, prot5_xl_uniref50). "
            "Must match a key in embeddings_models_output.json. "
            "If omitted, UMAP plots are skipped."
        )
    )
    parser.add_argument(
        "--embeddings_json", default="embeddings_models_output.json",
        help="Path to embeddings_models_output.json. Default: embeddings_models_output.json"
    )
    parser.add_argument(
        "--seq_identity", type=float, default=0.3,
        help=(
            "MMSeqs2 min-seq-id threshold for whole-dataset diversity clustering. "
            "Default: 0.5 (50%%). Clustering is skipped if mmseqs is not in PATH."
        )
    )
    parser.add_argument(
        "--dendrogram", action="store_true", default=False,
        help="Generate a dendrogram of cluster representatives (UPGMA, trigram cosine distance). Off by default."
    )
    args = parser.parse_args()

    profiler = DatasetCardPopulator(
        base_directory=args.base_dir,
        emb_model=args.emb_model,
        embeddings_json=args.embeddings_json,
        seq_identity=args.seq_identity,
        dendrogram=args.dendrogram,
    )
    profiler.run()