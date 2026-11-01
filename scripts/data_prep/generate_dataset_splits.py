#!/usr/bin/env python3
"""CLI shim for the dataset splitting utilities in :mod:`enzymado.dataset_splitting`."""

from enzymado import paths_and_constants
import enzymado.dataset_splitting as dataset_splitting

_P = paths_and_constants.Paths
PROJECT_ROOT = _P.project_root

if __name__ == "__main__":
    dataset_splitting.main()
