import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
from enzymado.dataset_card.real_case_dataset_card_populator import RealCaseDatasetCardPopulator

# Run script from project root (enzymado-benchmark/)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate dataset_card.json files for all real_case tasks.")
    parser.add_argument(
        "--base_dir", default="data/processed/real_case",
        help="Base directory containing one subdirectory per real_case task. Default: data/processed/real_case"
    )
    parser.add_argument(
        "--benchmark_card", default="benchmark_card.json",
        help="Path to benchmark_card.json, the source of each task's schema and provenance. Default: benchmark_card.json"
    )
    args = parser.parse_args()

    RealCaseDatasetCardPopulator(
        base_directory=args.base_dir,
        benchmark_card_path=args.benchmark_card,
    ).run()
