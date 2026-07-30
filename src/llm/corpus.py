"""Load the generated markdown cards into a single context blob for the LLM.

The corpus is tiny (~40 cards, ~43k tokens), so we stuff all of it into context rather
than doing embedding retrieval. Each card is tagged with its filename so the model can
cite `[task_kcat]`-style ids.

Both consumers stuff the whole thing: :mod:`llm.hypothesize` as background behind its
deterministic evidence table, and :mod:`llm.chat` as the chatbot's standing knowledge
base (see :func:`llm.chat.build_system_instruction`).
"""
from __future__ import annotations

import functools
from pathlib import Path

from . import sources as S

DEFAULT_MARKDOWN_DIR = S.REPO_ROOT / "markdown"

#: Delimiter prefix introducing each card; also how cards are counted.
CARD_MARKER = "===== CARD:"


@functools.lru_cache(maxsize=4)
def load_corpus(markdown_dir: Path | None = None) -> str:
    """Concatenate every ``*.md`` card into one id-tagged, separated string.

    Cached: the cards are read once per process per directory, since the chatbot rebuilds
    this on every new session. Regenerated cards therefore require a fresh process —
    matching ``sources.load_records`` / ``sources.load_models``.
    """
    md_dir = markdown_dir or DEFAULT_MARKDOWN_DIR
    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No markdown cards in {md_dir}. Run `python scripts/generate_cards.py` first."
        )
    blocks = []
    for path in files:
        blocks.append(f"{CARD_MARKER} {path.stem} =====\n{path.read_text().rstrip()}")
    return "\n\n".join(blocks)


def card_count(markdown_dir: Path | None = None) -> int:
    """How many cards :func:`load_corpus` would stuff (raises if there are none)."""
    return load_corpus(markdown_dir).count(CARD_MARKER)