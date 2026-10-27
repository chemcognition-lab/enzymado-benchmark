"""Rank Appendix B configurations by decreasing validation R^2.

Report validation MAE and Kendall's tau alongside all sampled parameters.
Require 400 completed runs by default so a partial array cannot claim a winner
for the full search. Use --expected-count explicitly for a partial study.

Example:
    python scripts/mlp_hpo/aggregate_hpo_results.py --hpo-dir mlp_hpo_runs/hpo_metrics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def aggregate(hpo_dir: Path, *, expected_count: int = 400) -> pd.DataFrame:
    """Read complete metrics, reject duplicate runs, and sort on R^2 only."""
    results = []
    for path in sorted(hpo_dir.glob("mlp_hpo_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for metric in ("val_R2", "val_MAE", "val_Kendall"):
            value = result[metric]
            if metric == "val_Kendall" and value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: {metric} must be a finite number.")
        results.append(result)
    if expected_count <= 0 or len(results) != expected_count:
        raise ValueError(f"Expected {expected_count} runs, found {len(results)}.")
    frame = pd.DataFrame(results)
    if frame["run_id"].duplicated().any():
        raise ValueError("Duplicate HPO run IDs.")
    return frame.sort_values("val_R2", ascending=False, kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    """Write the full ranking and print its top three configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hpo-dir", type=Path, default=Path("mlp_hpo_runs/hpo_metrics")
    )
    parser.add_argument("--output", type=Path, default=Path("hpo_final_results.csv"))
    parser.add_argument("--expected-count", type=int, default=400)
    args = parser.parse_args()
    frame = aggregate(args.hpo_dir, expected_count=args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Ranked {len(frame)} configurations in {args.output}")
    print(
        frame.head(3)[
            ["run_id", "val_R2", "val_MAE", "val_Kendall", "activation", "lr"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
