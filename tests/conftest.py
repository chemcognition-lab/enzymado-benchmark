"""Add the source tree for canonical package imports."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# OpenMM reads this once, at import. One thread keeps the seeded hydrogen
# placement in tests/test_structures.py repeatable (see that test).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
