"""Load the generated markdown cards into one context string for the language model.

The corpus is small (about 40 cards), so the whole of it goes into the prompt
instead of being retrieved by embedding search. Each card is tagged with its id
so the model can cite ``[task_kcat]``-style ids. :mod:`hypothesize` uses it as
background behind its evidence tables and :mod:`chat` as the chatbot's standing
knowledge base.

Example:
    python -c "from enzymado.llm import corpus; print(corpus.card_count())"
"""

from __future__ import annotations

import functools
from pathlib import Path

from enzymado.llm import render

DEFAULT_MARKDOWN_DIR = render.DEFAULT_OUTPUT

#: Prefix of the delimiter introducing each card; also how cards are counted.
CARD_MARKER = "===== CARD:"


@functools.lru_cache(maxsize=4)
def load_corpus(markdown_dir: Path | None = None) -> str:
    """Every ``*.md`` card, tagged with its id and separated by blank lines.

    Cached per directory, since the chatbot rebuilds it for every session; cards
    regenerated afterwards need a fresh process.
    """
    md_dir = markdown_dir or DEFAULT_MARKDOWN_DIR
    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No markdown cards in {md_dir}. Run `python -m enzymado.llm` first."
        )
    return "\n\n".join(
        f"{CARD_MARKER} {path.stem} =====\n{path.read_text().rstrip()}"
        for path in files
    )


def card_count(markdown_dir: Path | None = None) -> int:
    """How many cards :func:`load_corpus` would include; raises if there are none."""
    return load_corpus(markdown_dir).count(CARD_MARKER)
