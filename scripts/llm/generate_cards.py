#!/usr/bin/env python
"""Build the assistant's knowledge-base cards from the published results.

Writes one markdown card per representation, split, baseline, task and
leaderboard. Rebuild after the results change.

Example:
    python scripts/llm/generate_cards.py --output results/llm_cards
"""

from enzymado.llm import render

if __name__ == "__main__":
    raise SystemExit(render.main())
