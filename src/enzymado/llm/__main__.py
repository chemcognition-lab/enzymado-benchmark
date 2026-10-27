"""Build the knowledge-base cards: ``python -m enzymado.llm``.

Example:
    python -m enzymado.llm --output results/llm_cards
"""

from enzymado.llm import render

if __name__ == "__main__":
    raise SystemExit(render.main())
