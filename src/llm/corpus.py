"""Load the generated markdown cards into a single context blob for the LLM.

The corpus is tiny (~40 cards, well under 100k tokens), so we stuff all of it into
context rather than doing embedding retrieval. Each card is tagged with its filename so
the model can cite `[task_kcat]`-style ids in its hypotheses.
"""
from __future__ import annotations

from pathlib import Path

from . import sources as S

DEFAULT_MARKDOWN_DIR = S.REPO_ROOT / "markdown"


def load_corpus(markdown_dir: Path | None = None) -> str:
    """Concatenate every ``*.md`` card into one id-tagged, separated string."""
    md_dir = markdown_dir or DEFAULT_MARKDOWN_DIR
    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No markdown cards in {md_dir}. Run `python scripts/generate_cards.py` first."
        )
    blocks = []
    for path in files:
        blocks.append(f"===== CARD: {path.stem} =====\n{path.read_text().rstrip()}")
    return "\n\n".join(blocks)