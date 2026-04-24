import json
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Aggregate HPO JSON results into a CSV.")
    parser.add_argument("--hpo-dir", type=str, default="results/predictions/hpo_metrics", 
                        help="Directory containing the HPO JSON files.")
    parser.add_argument("--output", type=str, default="hpo_final_results.csv",
                        help="Name of the output CSV file.")
    args = parser.parse_args()

    hpo_dir = Path(args.hpo_dir)
    if not hpo_dir.exists():
        print(f"[ERROR] Directory not found: {hpo_dir.resolve()}")
        return

    results = []
    for file in hpo_dir.glob("*.json"):
        try:
            with open(file, "r") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"[WARNING] Could not read {file.name}: {e}")

    if results:
        df = pd.DataFrame(results)
        
        # Sort by Validation R-Squared (Highest is best)
        if "val_R2" in df.columns:
            df = df.sort_values(by="val_R2", ascending=False)
        
        df.to_csv(args.output, index=False)
        print(f"[SUCCESS] Aggregated {len(results)} HPO runs into {args.output}")
        
        # Print the top 3 best models to the terminal instantly!
        if "val_R2" in df.columns:
            print("\n--- TOP 3 MODELS ---")
            top_3 = df.head(3)[["run_id", "val_R2", "val_MAE", "val_Kendall", "lr", "num_layers", "activation"]]
            print(top_3.to_string(index=False))
            
    else:
        print("[INFO] No JSON files found in the HPO directory.")

if __name__ == "__main__":
    main()