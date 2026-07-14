"""Enzymado LLM layer: RAG knowledge-base cards + benchmark hypothesis generation.

Card generation (torch-free) sources everything from the repo's live artifacts:
see :mod:`llm.render` for the entry point and :mod:`llm.sources` for the loaders.

On top of those cards, the hypothesis engine (:mod:`llm.hypothesize`) proposes
*why model X beats model Y* — and why one combination of models beats another —
by pairing a deterministic head-to-head table (:mod:`llm.compare`) with a
Gemini generate-then-critique loop.
"""
