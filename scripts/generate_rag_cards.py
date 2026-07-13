#!/usr/bin/env python
"""Wrapper to generate the RAG knowledge-base cards with ``src`` on the path.

Usage:
    python scripts/generate_rag_cards.py                 # write cards to markdown/
    python scripts/generate_rag_cards.py --output DIR
    python scripts/generate_rag_cards.py --scaffold      # stub missing content/*.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_cards.render import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
