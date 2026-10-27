"""The ``Hypothesis`` record and its JSON (de)serialization.

A hypothesis must name its evidence (specific numbers) and its differentiating
attribute (a field that really differs between the models): no number and no
attribute, no claim. The generator fills every field but the verdict; a second,
critic call fills ``verdict`` and ``verdict_reason``.

Example:
    python -c "from enzymado.llm import schema; print(schema.parse_hypotheses('[]'))"
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing

VERDICTS = (
    "supported",
    "partly-supported",
    "cherry-picked",
    "contradicted",
    "uncertain",
)


@dataclasses.dataclass
class Hypothesis:
    claim: str
    mechanism: str
    evidence: list[str] = dataclasses.field(default_factory=list)
    differentiating_attribute: str = ""
    confidence: float = 0.0
    falsification_test: str = ""
    # Set when a claim mixes baselines or leans on a task only one side ran.
    caveat: str = ""
    # Filled by the critic.
    verdict: str = ""
    verdict_reason: str = ""

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


GENERATION_SCHEMA_HINT = """Return ONLY a JSON array of hypothesis objects, each:
{
  "claim": "one-sentence causal claim (X beats Y / set A beats set B because ...)",
  "mechanism": "why that attribute would produce the observed performance pattern",
  "evidence": ["numbers copied EXACTLY from the evidence tables,
               e.g. 'kcat similarity: Y 0.2492 vs X 0.2311'"],
  "differentiating_attribute": "the attribute field that differs and drives the claim",
  "confidence": 0.0-1.0,
  "falsification_test": "a task/split where this attribute predicts a result
                         that would DISPROVE the claim",
  "caveat": "empty string, UNLESS the claim mixes baselines, rests on ties within
             the noise threshold, or leans on a left-out task; then say so here"
}"""

CRITIQUE_SCHEMA_HINT = """Return ONLY a JSON array, one object per hypothesis:
{
  "index": copy the hypothesis's "index" field verbatim so the verdict is matched to it,
  "verdict": one of ["supported", "partly-supported", "cherry-picked",
              "contradicted", "uncertain"],
  "verdict_reason": "does the attribute hold across ALL tasks or only the cited
                     ones? cite the counter-cases",
  "confidence": revised 0.0-1.0
}"""


def _strip_fences(text: str) -> str:
    """Remove the ```json fences a model sometimes wraps around JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _load_array(text: str) -> list[dict]:
    data = json.loads(_strip_fences(text))
    if isinstance(data, dict):  # tolerate {"hypotheses": [...]}
        for value in data.values():
            if isinstance(value, list):
                return value
        return [data]
    return data


def _as_str_list(value: typing.Any) -> list[str]:
    """A lone string becomes one bullet instead of a list of characters."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_hypotheses(text: str) -> list[Hypothesis]:
    """Parse the generation step's JSON into ``Hypothesis`` objects."""
    return [
        Hypothesis(
            claim=obj.get("claim", ""),
            mechanism=obj.get("mechanism", ""),
            evidence=_as_str_list(obj.get("evidence")),
            differentiating_attribute=obj.get("differentiating_attribute", ""),
            confidence=float(obj.get("confidence") or 0.0),
            falsification_test=obj.get("falsification_test", ""),
            caveat=obj.get("caveat", "") or "",
        )
        for obj in _load_array(text)
    ]


def apply_critique(hyps: list[Hypothesis], text: str) -> list[Hypothesis]:
    """Merge the critic's verdicts onto the hypotheses.

    Matches on the ``index`` the critic echoes back, which survives a reordered or
    partial critique, and falls back to position only when no index is present.
    """
    verdicts = _load_array(text)
    by_index = {
        v["index"]: v
        for v in verdicts
        if isinstance(v, dict) and isinstance(v.get("index"), int)
    }
    for i, hyp in enumerate(hyps):
        if by_index:
            verdict = by_index.get(i)
        else:
            verdict = verdicts[i] if i < len(verdicts) else None
        if not isinstance(verdict, dict):
            continue
        hyp.verdict = verdict.get("verdict", "")
        hyp.verdict_reason = verdict.get("verdict_reason", "")
        if verdict.get("confidence") is not None:
            hyp.confidence = float(verdict["confidence"])
    return hyps
