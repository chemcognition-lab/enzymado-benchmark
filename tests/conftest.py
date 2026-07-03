"""Make src/ importable as a set of top-level packages (data_loading, training, ...).

Production entry points (scripts/run_benchmark.py) add src/ to sys.path directly,
so intra-src modules use bare absolute imports like `from models.common import ...`
rather than `from src.models.common import ...`. Tests need the same path entry to
import those modules, whether they do it via `from training.x import ...` or via
`from src.training.x import ...` (both resolve, as separate module objects).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
