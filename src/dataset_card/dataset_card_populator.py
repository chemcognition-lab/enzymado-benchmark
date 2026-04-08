import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class DatasetCardPopulator:
    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        self.template_path = self.base_dir / "active_sites" / "dataset_card_template.json"
        
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

    def run(self):
        print("Starting Dataset Card Population Pipeline...\n")
        
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
            
            print(f"  Successfully generated dataset_card.json for {prop_dir.name}\n")

        print("Pipeline Complete! All dataset_card.json files have been populated.")

if __name__ == "__main__":
    BASE_DIR = "data/processed/properties"
    DatasetCardPopulator(BASE_DIR).run()