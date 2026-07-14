#!/usr/bin/env python
"""Wrapper to run the benchmark hypothesis engine with ``src`` on the path.

Usage:
    python scripts/hypothesize.py --pair esm2_650m prost5
    python scripts/hypothesize.py --pair ankh_large esm2_650M --task kcat
    python scripts/hypothesize.py --combo ankh_large,prost5 esm2_650m,esmc_600m
    python scripts/hypothesize.py --pair esm2_650m prost5 --no-critique --json

Requires GEMINI_API_KEY in the environment and the `[llm]` extra installed
(`uv sync --extra llm`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm.hypothesize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())