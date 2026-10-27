#!/usr/bin/env python
"""Ask Gemini why one representation, or set of them, beats another.

Needs GEMINI_API_KEY in the environment and the llm extra (`uv sync --extra llm`).

Example:
    python scripts/llm/hypothesize.py --pair prost5 esm2_650M --task kcat
"""

from enzymado.llm import hypothesize

if __name__ == "__main__":
    raise SystemExit(hypothesize.main())
