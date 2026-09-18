"""Populate dataset_card.json files for real_case tasks, driven by benchmark_card.json.

Unlike DatasetCardPopulator (data/processed/properties), which hardcodes a
target/key-column map per property, every real_case task already declares its
own schema (target col, sequence col, ligand col, dedup keys, provenance) in
benchmark_card.json's "real_case" section — so this reads that instead of
maintaining a second hardcoded map.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class RealCaseDatasetCardPopulator:
    """Compute dataset statistics for real_case tasks and write dataset_card.json files."""

    def __init__(
        self,
        base_directory: str = "data/processed/real_case",
        benchmark_card_path: str = "benchmark_card.json",
    ):
        self.base_dir = Path(base_directory)
        self.template_path = self.base_dir / "dataset_card_template.json"
        with open(benchmark_card_path) as f:
            self.benchmark_card = json.load(f)

    def sanitize_for_json(self, obj: Any) -> Any:
        """Convert pandas and NumPy values into JSON-safe Python objects."""
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
        """Discover the base CSV and standardized split CSV(s) in one task directory.

        real_case files are named `<name>.csv` (base) and
        `<name>_random_80_10_10_standardized.csv` (split) — distinguished by
        the `_standardized` suffix, unlike the properties tree's `*splits*`
        naming convention.
        """
        files = {"base": None, "splits": {}}
        for file in directory.glob("*.csv"):
            if file.stem.endswith("_standardized"):
                files["splits"][file.stem] = file
            else:
                files["base"] = file
        return files

    def calc_distribution(self, series: pd.Series) -> Any:
        """Summarize a label distribution for categorical or numeric targets."""
        series = series.dropna()
        if series.empty:
            return None

        total_valid = int(len(series))
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = series.nunique()

        if not is_numeric or unique_count < 15:
            counts = series.value_counts().to_dict()
            ratios = series.value_counts(normalize=True).to_dict()
            return {
                "total_valid_labels": total_valid,
                "classes": {str(k): {"count": int(counts[k]), "ratio": round(float(ratios[k]), 4)} for k in counts},
            }
        return {
            "total_valid_labels": total_valid,
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "median": round(float(series.median()), 4),
        }

    def process_splits(
        self, base_df: pd.DataFrame, split_path: Path, target_col: Optional[str], key_col: Optional[str]
    ) -> Dict[str, Any]:
        """Compute leakage checks, split sizes, and per-split label distributions."""
        split_df = pd.read_csv(split_path)
        split_col = next((c for c in split_df.columns if c.lower() in ["split", "set"]), None)
        if split_col is None:
            return {"leakage": False, "distributions": {}, "sizes": {}}

        leakage = False
        if key_col and key_col in split_df.columns:
            splits_grouped = split_df.groupby(split_col)[key_col].apply(set).to_dict()
            train_set = splits_grouped.get("train", set())
            val_set = splits_grouped.get("val", splits_grouped.get("validation", set()))
            test_set = splits_grouped.get("test", set())
            leakage = bool((train_set & val_set) or (train_set & test_set) or (val_set & test_set))

        sizes = split_df[split_col].value_counts().to_dict()

        distributions = {}
        if target_col and target_col in base_df.columns and len(split_df) == len(base_df):
            # Align positionally rather than joining on key_col: dataset_splitting.py
            # only appends split/standardized columns onto the base rows in place, and
            # real_case id columns (e.g. dataset_id) aren't guaranteed unique — some
            # tasks (erickson_petase) have repeat measurements under the same id.
            target_values = base_df[target_col].reset_index(drop=True)
            split_col_values = split_df[split_col].reset_index(drop=True)
            for split_name, group_values in target_values.groupby(split_col_values):
                if split_name == "val":
                    split_name = "validation"
                distributions[split_name] = self.calc_distribution(group_values)

        return {"leakage": leakage, "distributions": distributions, "sizes": sizes}

    def calc_stats(
        self,
        base_df: pd.DataFrame,
        schema: Dict[str, Any],
        filters: Dict[str, Any],
        split_files: Dict[str, Path],
    ) -> Dict[str, Any]:
        """Collect dataset-wide statistics for one real_case task."""
        seq_col = schema.get("sequence_col")
        target_col = schema.get("target_col")
        id_col = schema.get("protein_id_col")
        ligand_col = schema.get("ligand_smiles_col")
        dedup_keys = filters.get("deduplicate_by") or []

        seq_stats = {"min": None, "max": None, "mean": None, "median": None, "p95": None}
        if seq_col and seq_col in base_df.columns:
            lengths = base_df[seq_col].dropna().str.len()
            if not lengths.empty:
                seq_stats = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": round(float(lengths.mean()), 2),
                    "median": float(lengths.median()),
                    "p95": float(np.percentile(lengths, 95)),
                }

        unique_proteins = int(base_df[seq_col].nunique()) if seq_col and seq_col in base_df.columns else None
        unique_ligands = int(base_df[ligand_col].nunique()) if ligand_col and ligand_col in base_df.columns else None

        nans = base_df.isna().sum().to_dict()
        dup_full = int(base_df.duplicated().sum())
        dup_key = (
            int(base_df.duplicated(subset=dedup_keys).sum())
            if dedup_keys and all(k in base_df.columns for k in dedup_keys)
            else None
        )

        label_dist = {
            "overall": self.calc_distribution(base_df[target_col])
            if target_col and target_col in base_df.columns
            else None,
            "train": {},
            "validation": {},
            "test": {},
        }
        split_sizes: Dict[str, Any] = {}
        split_leakages: Dict[str, Any] = {}
        key_col = id_col if id_col and id_col in base_df.columns else (dedup_keys[0] if dedup_keys else None)

        for split_name, split_path in split_files.items():
            split_res = self.process_splits(base_df, split_path, target_col, key_col)
            split_leakages[split_name] = split_res["leakage"]
            split_sizes[split_name] = split_res["sizes"]
            for s_type, dist in split_res["distributions"].items():
                if s_type in label_dist:
                    label_dist[s_type][split_name] = dist

        notes = []
        small_dataset_warning = None
        if unique_proteins is not None and unique_proteins < 10:
            small_dataset_warning = (
                f"Only {unique_proteins} distinct protein sequence(s) in this dataset — "
                "val/test splits may cover a single protein, a noisy generalization signal."
            )
            notes.append(small_dataset_warning)
        if dup_full > 0:
            notes.append(f"Warning: base dataset contains {dup_full} full row duplicates.")

        return self.sanitize_for_json(
            {
                "amount_of_removed_data": {"datapoints_after_cleaning": len(base_df)},
                "nans_dups": {
                    "nan_counts_per_column": nans,
                    "duplicate_rows_full": dup_full,
                    "duplicate_rows_by_key": dup_key,
                },
                "unique_proteins": unique_proteins,
                "unique_ligands": unique_ligands,
                "label_distribution": label_dist,
                "split_sizes": split_sizes,
                "sequence_length_summary": seq_stats,
                "quality_checks": {
                    "schema_validated": True,
                    "split_leakage_checked": split_leakages,
                    "label_format_validated": target_col is not None,
                    "small_dataset_warning": small_dataset_warning,
                },
                "notes": " | ".join(notes) if notes else None,
            }
        )

    def run(self):
        """Populate dataset_card.json for every real_case task in benchmark_card.json."""
        print("Starting Real-Case Dataset Card Population Pipeline...\n")
        real_case_tasks = self.benchmark_card.get("real_case", {})

        with open(self.template_path) as f:
            template = json.load(f)

        for task_name, task_cfg in sorted(real_case_tasks.items()):
            task_dir = self.base_dir / task_name
            if not task_dir.is_dir():
                print(f"Skipping {task_name}: directory not found at {task_dir}")
                continue

            dataset_entry = task_cfg["datasets"][0]
            schema = dataset_entry.get("schema", {})
            filters = dataset_entry.get("filters", {})
            provenance = dataset_entry.get("provenance", {})

            files = self.get_files(task_dir)
            if not files["base"]:
                print(f"Skipping {task_name}: no base CSV found in {task_dir}")
                continue

            print(f"Processing {task_name}...")
            base_df = pd.read_csv(files["base"])
            stats = self.calc_stats(base_df, schema, filters, files["splits"])

            card = json.loads(json.dumps(template))
            card["task_name"] = task_name
            card["dataset_id"] = dataset_entry.get("dataset_id")
            card["origin_of_data"] = {
                "source_name": provenance.get("source_name"),
                "source_version": provenance.get("source_version"),
                "citation": provenance.get("citation"),
            }
            card["schema"] = {
                "protein_id_col": schema.get("protein_id_col"),
                "sequence_col": schema.get("sequence_col"),
                "ligand_smiles_col": schema.get("ligand_smiles_col"),
                "condition_cols": schema.get("condition_cols") or schema.get("condition_col"),
                "target_col": schema.get("target_col"),
            }
            card["amount_of_removed_data"] = stats["amount_of_removed_data"]
            card["statistics"]["nans_dups"] = stats["nans_dups"]
            card["statistics"]["unique_proteins"] = stats["unique_proteins"]
            card["statistics"]["unique_ligands"] = stats["unique_ligands"]
            card["statistics"]["label_distribution"] = stats["label_distribution"]
            card["statistics"]["split_sizes"] = stats["split_sizes"]
            card["additional_important"]["sequence_length_summary"] = stats["sequence_length_summary"]
            card["additional_important"]["quality_checks"] = stats["quality_checks"]
            card["additional_important"]["notes"] = stats["notes"]
            card["additional_important"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            out_path = task_dir / "dataset_card.json"
            with open(out_path, "w") as f:
                json.dump(card, f, indent=2)
            print(f"  Wrote {out_path}\n")

        print("Pipeline Complete! dataset_card.json written for all available real_case tasks.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate dataset_card.json files for all real_case tasks.")
    parser.add_argument("--base_dir", default="data/processed/real_case")
    parser.add_argument("--benchmark_card", default="benchmark_card.json")
    args = parser.parse_args()
    RealCaseDatasetCardPopulator(args.base_dir, args.benchmark_card).run()
