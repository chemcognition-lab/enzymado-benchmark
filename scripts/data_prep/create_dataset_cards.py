"""Populate dataset-card metadata for all processed property datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import enzymado.dataset_card.dataset_card_populator as dataset_card_populator  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    """Create the dataset-card command-line parser."""
    parser = argparse.ArgumentParser(
        description="Populate dataset_card.json files for all properties."
    )
    parser.add_argument(
        "--base_dir",
        default="data/processed/properties",
        help=(
            "Base directory containing one subdirectory per property. "
            "Default: data/processed/properties"
        ),
    )
    parser.add_argument(
        "--emb_model",
        default=None,
        help=(
            "Name of the embedding model to use for UMAP generation "
            "(e.g. esm2_650M, ankh_large, prot5_xl_uniref50). "
            "Must match a key in embeddings_models_output.json. "
            "If omitted, UMAP plots are skipped."
        ),
    )
    parser.add_argument(
        "--embeddings_json",
        default="embeddings_models_output.json",
        help=(
            "Path to embeddings_models_output.json. "
            "Default: embeddings_models_output.json"
        ),
    )
    parser.add_argument(
        "--seq_identity",
        type=float,
        default=dataset_card_populator.DEFAULT_SEQUENCE_IDENTITY,
        help=(
            "MMSeqs2 min-seq-id threshold for whole-dataset diversity clustering. "
            "Default: 0.5 (50%%). Clustering is skipped if mmseqs is not in PATH."
        ),
    )
    parser.add_argument(
        "--figures_dir",
        default=str(dataset_card_populator.DEFAULT_FIGURES_DIR),
        help=(
            "Root directory for generated figures. Point this elsewhere to "
            "keep a trial run from overwriting the published figures."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write cards even when a section cannot be built (missing mmseqs, "
            "missing umap-learn, unresolved embeddings). Without this the run "
            "stops rather than emitting a card that silently lacks a section."
        ),
    )
    return parser


def main() -> None:
    """Populate all dataset cards using the selected CLI configuration."""
    args = create_parser().parse_args()
    profiler = dataset_card_populator.DatasetCardPopulator(
        base_directory=args.base_dir,
        emb_model=args.emb_model,
        embeddings_json=args.embeddings_json,
        seq_identity=args.seq_identity,
        allow_incomplete=args.allow_incomplete,
        figures_dir=args.figures_dir,
    )
    profiler.run()


if __name__ == "__main__":
    main()
