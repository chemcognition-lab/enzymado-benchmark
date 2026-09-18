#!/usr/bin/env python3
"""CLI shim for the dataset splitting utilities in :mod:`enzymado.dataset_splitting`."""

from pathlib import Path
import sys

# Direct script execution needs the repository root before the project import.
# ruff: noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import enzymado.dataset_splitting as dataset_splitting

if __name__ == "__main__":
    dataset_splitting.main()
