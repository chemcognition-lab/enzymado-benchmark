"""Build the expanded CatPred kcat table with products, reactions, pH and T.

Example: python scripts/data_prep/prep_kcat_expanded.py
"""

import json
import time

from enzymado import paths_and_constants
import enzymado.propertyprep.catpred as catpred


def main() -> None:
    """Clean the raw CatPred kcat file into kcat/kcat_expanded.csv."""
    start = time.time()
    stats = catpred.prep_kcat_expanded(
        paths_and_constants.Paths(), paths_and_constants.Constants()
    )
    print(json.dumps(stats, indent=2))
    print(f"kcat_expanded built in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
