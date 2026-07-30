"""The ``Hypothesis`` record and (de)serialization helpers.

A hypothesis is deliberately constrained: it must name the *evidence* (specific numbers)
and the *differentiating attribute* (a field that actually differs between the models) it
rests on. Those two fields are the leash — no number + no field-difference → no claim.

The generator fills every field except ``verdict``/``verdict_reason``; the critic pass
fills those in a second call. We keep this as a plain dataclass (no pydantic) so the base
package stays dependency-light; Gemini is asked for JSON and we parse it here.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Any

VERDICTS = ("supported", "partly-supported", "cherry-picked", "contradicted", "uncertain")


@dataclass
class Hypothesis:
    claim: str
    mechanism: str
    evidence: list[str] = field(default_factory=list)
    differentiating_attribute: str = ""
    confidence: float = 0.0
    falsification_test: str = ""
    # Disclaimer the model adds when a claim mixes configs / rests on excluded tasks:
    caveat: str = ""
    # Filled by the critic pass:
    verdict: str = ""
    verdict_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Shown to the model so it emits exactly these fields (generation step).
GENERATION_SCHEMA_HINT = """Return ONLY a JSON array of hypothesis objects, each:
{
  "claim": "one-sentence causal claim (X beats Y / set A beats set B because ...)",
  "mechanism": "why that attribute would produce the observed performance pattern",
  "evidence": ["numbers copied EXACTLY from the evidence tables, e.g. 'kcat similarity: Y 0.5927 vs X 0.676'"],
  "differentiating_attribute": "the attribute field that differs and drives the claim",
  "confidence": 0.0-1.0,
  "falsification_test": "a task/split where this attribute predicts a result that would DISPROVE the claim",
  "caveat": "empty string, UNLESS the claim mixes model configurations or leans on an excluded task — then note it here"
}"""

# Shown to the critic so it appends a verdict per hypothesis (critique step).
CRITIQUE_SCHEMA_HINT = """Return ONLY a JSON array, one object per hypothesis:
{
  "index": copy the hypothesis's "index" field verbatim so the verdict is matched to it,
  "verdict": one of ["supported","partly-supported","cherry-picked","contradicted","uncertain"],
  "verdict_reason": "does the attribute hold across ALL tasks or only the cited ones? cite the counter-cases",
  "confidence": revised 0.0-1.0
}"""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model sometimes adds around JSON."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _load_array(text: str) -> list[dict]:
    data = json.loads(_strip_fences(text))
    if isinstance(data, dict):  # tolerate {"hypotheses": [...]}
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return data


def _as_str_list(value: Any) -> list[str]:
    """Coerce a model's ``evidence`` field to a list of strings.

    Models often return a lone string instead of a one-element array; ``list()`` on
    that would explode it into characters. A bare string becomes a single bullet.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_hypotheses(text: str) -> list[Hypothesis]:
    """Parse the generation step's JSON into ``Hypothesis`` objects."""
    out = []
    for obj in _load_array(text):
        out.append(Hypothesis(
            claim=obj.get("claim", ""),
            mechanism=obj.get("mechanism", ""),
            evidence=_as_str_list(obj.get("evidence")),
            differentiating_attribute=obj.get("differentiating_attribute", ""),
            confidence=float(obj.get("confidence") or 0.0),
            falsification_test=obj.get("falsification_test", ""),
            caveat=obj.get("caveat", "") or "",
        ))
    return out


def apply_critique(hyps: list[Hypothesis], text: str) -> list[Hypothesis]:
    """Merge the critic step's verdicts back onto the hypotheses.

    Prefer matching on the ``index`` the critic echoes back (robust to a reordered
    or partial critique); fall back to positional pairing only when no indices are
    present. Non-dict / out-of-range entries are skipped rather than misapplied.
    """
    verdicts = _load_array(text)
    by_index = {v["index"]: v for v in verdicts
                if isinstance(v, dict) and isinstance(v.get("index"), int)}
    for i, h in enumerate(hyps):
        v = by_index.get(i) if by_index else (verdicts[i] if i < len(verdicts) else None)
        if not isinstance(v, dict):
            continue
        h.verdict = v.get("verdict", "")
        h.verdict_reason = v.get("verdict_reason", "")
        if v.get("confidence") is not None:
            h.confidence = float(v["confidence"])
    return hyps