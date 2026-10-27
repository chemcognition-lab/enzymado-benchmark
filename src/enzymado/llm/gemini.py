"""Thin Google Gemini wrapper: one ``generate`` call, keyed by ``GEMINI_API_KEY``.

The SDK is imported lazily, so the rest of the package (cards and comparisons)
works without ``google-genai`` installed; it lives in the optional ``llm`` extra
(``uv sync --extra llm``). The key is only ever read from the environment.

Example:
    GEMINI_API_KEY=... python -m enzymado.llm.hypothesize --pair prost5 esm2_650M
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
    except ImportError as exc:  # depends on the environment
        raise GeminiError(
            "google-genai is not installed. Run `uv sync --extra llm`."
        ) from exc
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
    """One generation; returns the model's text, a JSON string when ``json_mode``."""
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    client = _client()
    try:
        response = client.models.generate_content(
            model=model_name(model), contents=prompt, config=config
        )
        # ``.text`` can itself raise on a safety block, so read it inside the guard.
        text = response.text
    except Exception as exc:  # noqa: BLE001 - any SDK failure becomes GeminiError
        raise GeminiError(f"Gemini request failed: {exc}") from exc
    if not text:
        reason = getattr(
            getattr(response, "prompt_feedback", None), "block_reason", None
        )
        if reason:
            raise GeminiError(f"Gemini blocked the request ({reason}).")
        raise GeminiError("Gemini returned an empty response.")
    return text
