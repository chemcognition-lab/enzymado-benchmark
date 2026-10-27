"""Enzymado assistant: knowledge-base cards, exact comparisons and a Gemini chatbot.

:mod:`render` builds markdown cards from the published results (:mod:`sources`)
and the authored prose in ``content/``. :mod:`compare` computes head-to-head and
model-set comparisons exactly, so the language model never adds numbers up itself.
:mod:`hypothesize` turns a comparison into ranked, self-critiqued hypotheses, and
:mod:`chat` exposes all of it as a chatbot, served by ``scripts/llm/chat_app.py``.
Only :mod:`gemini` talks to the model; the rest is offline and needs no API key.

Example:
    python -m enzymado.llm --output results/llm_cards
"""
