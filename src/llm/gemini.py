"""Thin Google Gemini wrapper (lazy-imported SDK, keyed off ``GEMINI_API_KEY``).

Kept deliberately small: one ``generate`` call returning raw text. The SDK is imported
lazily so importing the ``llm`` package for card generation never requires ``google-genai``
(it lives in the optional ``[llm]`` extra).

Install: ``uv sync --extra llm``  (or ``pip install google-genai streamlit``).
"""
from __future__ import annotations

import functools
import os

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set in the environment.")
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover - env-dependent
        raise GeminiError(
            "google-genai is not installed. Run `uv sync --extra llm` "
            "or `pip install google-genai`."
        ) from e
    return genai.Client(api_key=api_key)


def model_name(override: str | None = None) -> str:
    return override or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """Single-shot generation; returns the model's text (JSON string when ``json_mode``).

    Thinking is left at the model default (on for the 2.5 models). ``json_mode`` sets the
    response mime type so the model returns parseable JSON.
    """
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    client = _client()  # raises GeminiError on missing key / uninstalled SDK
    try:
        resp = client.models.generate_content(
            model=model_name(model), contents=prompt, config=config
        )
        # `.text` is a property that can itself raise on a safety/recitation block,
        # so it must be read inside the guard — not after it.
        text = resp.text
    except Exception as e:  # network, rate limit, invalid model id, safety block, …
        raise GeminiError(f"Gemini request failed: {e}") from e
    if not text:
        # Empty text usually means a prompt-level block; surface the reason if present.
        reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        if reason:
            raise GeminiError(f"Gemini blocked the request ({reason}).")
        raise GeminiError("Gemini returned an empty response.")
    return text