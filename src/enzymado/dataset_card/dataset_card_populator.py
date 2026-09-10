"""Populate dataset card JSON files with statistics computed from processed CSVs."""

# Keep positional compatibility for the existing constructor API.
# ruff: noqa: FBT001, FBT002

import argparse
import datetime
import json
from pathlib import Path
import shutil
import typing

import pandas as pd

from . import clustering, statistics, umap_plots

DEFAULT_SEQUENCE_IDENTITY = 0.5
DEFAULT_FIGURES_DIR = Path("results/figures")


class DatasetCardPopulator:
    """Compute dataset statistics and write them back into dataset card files."""

    def __init__(
        self,
        base_directory: str | Path,
        emb_model: str | None = None,
        embeddings_json: str | Path | None = None,
        seq_identity: float = DEFAULT_SEQUENCE_IDENTITY,
        dendrogram: bool = False,
        allow_incomplete: bool = False,
        figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    ) -> None:
        self.allow_incomplete = allow_incomplete
        # Figure output was previously hardcoded relative to the working
        # directory, so any sandboxed run silently overwrote the published
        # figures in the repository.
        self.figures_dir = Path(figures_dir)
        self.base_dir = Path(base_directory)
        self.template_path = (
            self.base_dir / "active_sites" / "dataset_card_template.json"
        )
        self.emb_model = emb_model
        self.embeddings_json = Path(embeddings_json) if embeddings_json else None
        self.seq_identity = (
            seq_identity  # Identity threshold for whole-dataset MMSeqs2 clustering
        )
        self.dendrogram = (
            dendrogram  # Whether to generate a dendrogram of cluster representatives
        )

        self.target_map = dict(statistics.TARGET_MAP)
        self.primary_keys_map = {
            property_name: list(keys)
            for property_name, keys in statistics.PRIMARY_KEYS_MAP.items()
        }

    def get_files(self, directory: Path) -> dict[str, typing.Any]:
        """Discover the base CSV, split CSVs, and card path for one property."""
        return statistics.find_files(directory)

    def calc_stats(
        self,
        base_df: pd.DataFrame,
        prop_name: str,
        split_files: dict[str, Path],
    ) -> dict[str, typing.Any]:
        """Collect dataset-wide statistics for one processed property table."""
        return statistics.calculate(
            base_df,
            prop_name,
            split_files,
            target_map=self.target_map,
            primary_keys_map=self.primary_keys_map,
        )

    def _load_embeddings(self, sequences: list, uniprot_ids: list, emb_path: Path):
        """Load mean-pooled embeddings and their source row indices."""
        return umap_plots.load_embeddings(sequences, uniprot_ids, emb_path)

    def generate_umap(
        self,
        base_df: pd.DataFrame,
        prop_name: str,
        split_files: dict[str, Path],
        emb_path: Path,
        emb_model: str,
        output_dir: Path,
    ) -> dict[str, str] | None:
        """Generate label-colored and split-colored UMAP figures."""
        return umap_plots.generate(
            base_df,
            prop_name,
            split_files,
            emb_path,
            emb_model,
            output_dir,
            self.target_map,
            self._load_embeddings,
        )

    def run_clustering(
        self, base_df: pd.DataFrame, prop_name: str
    ) -> dict[str, typing.Any] | None:
        """Run MMseqs2 easy-cluster on the unique dataset sequences."""
        return clustering.run(
            base_df,
            prop_name,
            self.seq_identity,
        )

    def generate_cluster_diversity_plots(
        self,
        cluster_stats: dict[str, typing.Any],
        prop_name: str,
        output_dir: Path,
    ) -> str | None:
        """Generate the rank-abundance and rarefaction figure."""
        return clustering.plot_diversity(
            cluster_stats,
            prop_name,
            output_dir,
            self.seq_identity,
        )

    def generate_dendrogram(
        self,
        cluster_stats: dict[str, typing.Any],
        prop_name: str,
        output_dir: Path,
    ) -> str | None:
        """Generate a dendrogram of cluster representatives."""
        return clustering.plot_dendrogram(
            cluster_stats,
            prop_name,
            output_dir,
        )

    def _resolve_embedding_path(self) -> Path | None:
        """Resolve the selected embedding path from its registry."""
        if not self.emb_model:
            return None

        registry_path = self.embeddings_json or Path("embeddings_models_output.json")
        if not registry_path.exists():
            print(
                f"Warning: embeddings JSON not found at {registry_path}. "
                "Skipping UMAP.\n"
            )
            return None

        with registry_path.open(encoding="utf-8") as handle:
            registry = json.load(handle)
        if self.emb_model not in registry:
            print(
                f"Warning: embedding model '{self.emb_model}' not found in "
                f"{registry_path}. Available: {', '.join(registry)}. "
                "Skipping UMAP.\n"
            )
            return None

        model_entry = registry[self.emb_model]
        pooled_path = model_entry.get("pooled_path")
        if pooled_path and Path(pooled_path).exists():
            emb_path = Path(pooled_path)
            print(f"UMAP embeddings: {self.emb_model} → {emb_path} (pre-pooled)\n")
            return emb_path

        emb_path = Path(model_entry["path"])
        print(
            f"UMAP embeddings: {self.emb_model} → {emb_path} "
            "(per-residue, will pool on-the-fly)\n"
        )
        return emb_path

    def _create_card(
        self,
        base_df: pd.DataFrame,
        prop_name: str,
        files: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Build one populated card from the shared template."""
        stats = self.calc_stats(base_df, prop_name, files["splits"])
        with self.template_path.open(encoding="utf-8") as handle:
            card = json.load(handle)

        card["property_name"] = prop_name
        card["dataset_id"] = files["base"].stem
        card["amount_of_removed_data"].update(stats["amount_of_removed_data"])
        card["statistics"]["nans_dups"] = stats["nans_dups"]
        card["statistics"]["label_distribution"] = stats["label_distribution"]
        card["statistics"]["split_sizes_by_method"] = stats["split_sizes"]
        important = card["additional_important"]
        important["sequence_length_summary"] = stats["sequence_length_summary"]
        important["quality_checks"].update(stats["quality_checks"])
        important["notes"] = stats["notes"]
        important["last_updated"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return card

    @staticmethod
    def _write_card(card: dict[str, typing.Any], path: Path) -> None:
        """Write one card using the existing JSON representation."""
        with path.open("w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=2)

    def _unavailable(self, section: str, reason: str, hint: str = "") -> dict[str, str]:
        """Abort on a missing prerequisite, or describe the gap when allowed.

        A dataset card that quietly omits a section is indistinguishable from
        one where the section legitimately does not apply, so the default is to
        stop. ``--allow-incomplete`` keeps the old behaviour but records the
        omission inside the card, where a reader will actually see it.
        """
        message = f"cannot build the '{section}' section: {reason}"
        if not self.allow_incomplete:
            tail = f" {hint}" if hint else ""
            raise SystemExit(
                f"ERROR: {message}.{tail} "
                "Pass --allow-incomplete to write the card without it."
            )
        print(f"  [card] {message} — recorded as skipped.")
        return {"status": "skipped", "reason": reason}

    def _record_section(
        self,
        card_path: Path,
        top_key: str,
        name: str,
        value: typing.Any,
    ) -> dict[str, typing.Any]:
        """Re-read the card, set one section, and write it back."""
        with card_path.open(encoding="utf-8") as handle:
            card = json.load(handle)
        card.setdefault(top_key, {})[name] = value
        self._write_card(card, card_path)
        return card

    def _add_cluster_outputs(
        self,
        card: dict[str, typing.Any],
        card_path: Path,
        base_df: pd.DataFrame,
        prop_name: str,
        output_dir: Path,
    ) -> dict[str, typing.Any]:
        """Add MMseqs2 statistics and optional cluster figures to one card."""
        if shutil.which("mmseqs") is None:
            return self._record_section(
                card_path,
                "statistics",
                "entire_protein_pool_cluster_with_mmseqs2",
                self._unavailable(
                    "mmseqs2 clustering",
                    "mmseqs is not on PATH",
                    "Hint: run 'module load mmseqs2' before launching this script.",
                ),
            )
        cluster_stats = self.run_clustering(base_df, prop_name)
        if not cluster_stats:
            return card

        diversity_path = self.generate_cluster_diversity_plots(
            cluster_stats,
            prop_name,
            output_dir,
        )
        dendrogram_path = (
            self.generate_dendrogram(cluster_stats, prop_name, output_dir)
            if self.dendrogram
            else None
        )
        cluster_for_card = {
            key: value
            for key, value in cluster_stats.items()
            if not key.startswith("_")
        }
        cluster_for_card["cluster_diversity_plot"] = diversity_path
        cluster_for_card["dendrogram_plot"] = dendrogram_path

        with card_path.open(encoding="utf-8") as handle:
            card = json.load(handle)
        card["statistics"]["entire_protein_pool_cluster_with_mmseqs2"] = (
            cluster_for_card
        )
        self._write_card(card, card_path)
        return card

    def _add_umap_output(
        self,
        card: dict[str, typing.Any],
        card_path: Path,
        base_df: pd.DataFrame,
        prop_name: str,
        split_files: dict[str, Path],
        emb_path: Path | None,
        output_dir: Path,
    ) -> dict[str, typing.Any]:
        """Add UMAP figure paths when an embedding model was resolved."""
        if not self.emb_model:
            return card  # no embedding model requested: UMAPs do not apply
        if emb_path is None:
            return self._record_section(
                card_path,
                "additional_important",
                "umap",
                self._unavailable(
                    "UMAP",
                    f"no embedding source resolved for '{self.emb_model}'",
                ),
            )
        if umap_plots.umap_lib is None:
            return self._record_section(
                card_path,
                "additional_important",
                "umap",
                self._unavailable(
                    "UMAP",
                    "umap-learn is not installed",
                    "Hint: uv pip install umap-learn.",
                ),
            )

        umap_result = self.generate_umap(
            base_df=base_df,
            prop_name=prop_name,
            split_files=split_files,
            emb_path=emb_path,
            emb_model=self.emb_model,
            output_dir=output_dir,
        )
        if not umap_result:
            return card

        with card_path.open(encoding="utf-8") as handle:
            card = json.load(handle)
        card["additional_important"]["umap"] = umap_result
        self._write_card(card, card_path)
        return card

    def run(self) -> None:
        """Populate every dataset card below the configured base directory."""
        print("Starting Dataset Card Population Pipeline...\n")
        emb_path = self._resolve_embedding_path()

        umap_output_dir = self.figures_dir / "umaps"
        cluster_diversity_output_dir = self.figures_dir / "cluster_diversity"

        for prop_dir in sorted(self.base_dir.iterdir()):
            if not prop_dir.is_dir():
                continue
            print(f"Processing {prop_dir.name}...")

            files = self.get_files(prop_dir)
            if not files["base"]:
                continue

            base_df = pd.read_csv(files["base"]).copy()
            card = self._create_card(base_df, prop_dir.name, files)
            self._write_card(card, files["json"])

            card = self._add_cluster_outputs(
                card,
                files["json"],
                base_df,
                prop_dir.name,
                cluster_diversity_output_dir,
            )

            self._add_umap_output(
                card,
                files["json"],
                base_df,
                prop_dir.name,
                files["splits"],
                emb_path,
                umap_output_dir,
            )

            print(f"  Successfully generated dataset_card.json for {prop_dir.name}\n")

        print("Pipeline Complete! All dataset_card.json files have been populated.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone dataset-card command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/processed/properties")
    parser.add_argument(
        "--emb_model",
        default=None,
        help="Embedding model for UMAP (e.g. esm2_650M). Skipped if omitted.",
    )
    parser.add_argument("--embeddings_json", default="embeddings_models_output.json")
    parser.add_argument(
        "--seq_identity",
        type=float,
        default=DEFAULT_SEQUENCE_IDENTITY,
        help="MMseqs2 minimum sequence identity. Default: 0.5.",
    )
    parser.add_argument(
        "--figures_dir",
        default=str(DEFAULT_FIGURES_DIR),
        help=f"Root for generated figures. Default: {DEFAULT_FIGURES_DIR}.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write cards even when a section cannot be built (missing mmseqs, "
            "missing umap-learn, unresolved embeddings). The omission is "
            "recorded inside the card instead of aborting."
        ),
    )
    return parser


def main() -> None:
    """Populate dataset cards from command-line arguments."""
    args = build_argument_parser().parse_args()
    DatasetCardPopulator(
        args.base_dir,
        emb_model=args.emb_model,
        embeddings_json=args.embeddings_json,
        seq_identity=args.seq_identity,
    ).run()


if __name__ == "__main__":
    main()
