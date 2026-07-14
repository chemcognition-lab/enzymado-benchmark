"""Prompt construction for the generate-then-critique loop.

The deterministic evidence tables (from :mod:`llm.compare`) go *first and prominently*;
the full card corpus follows as background. The grounding rules force every claim to be
tied to a cited number and a real attribute difference, and to respect the cards' own
``needs_verification`` / caveat flags.
"""
from __future__ import annotations

from . import schema

SYSTEM = """You are a protein-ML benchmark analyst. You propose falsifiable hypotheses \
for WHY one model (or set of models) outperforms another on the Enzymado benchmark.

Hard rules:
- Ground every claim in the PRECOMPUTED EVIDENCE tables and the cards. The numbers in the \
evidence tables are authoritative and correct — never invent or recompute numbers. Copy figures \
into `evidence` EXACTLY as they appear in the tables (same digits); do not round or paraphrase them.
- Each hypothesis must cite specific numbers (in `evidence`) AND name a real attribute that \
differs between the models (in `differentiating_attribute`). No number + no attribute → no claim.
- An attribute flagged `needs_verification` is soft evidence — cap confidence ≤ 0.4 for any \
claim that leans on it.
- Respect stated caveats. E.g. a model's retrieval advantage may be DISABLED in this \
single-sequence benchmark; do not attribute a win to a capability the cards say is unused.
- Each score is the model's BEST head/pooling for that task, so a comparison can mix \
configurations, and split×tasks run for only one side are EXCLUDED (see 'Methodology & caveats' \
in the evidence). If a hypothesis leans on a mixed-configuration comparison or an excluded task, \
you MUST state that limitation in its `caveat` field; otherwise leave `caveat` empty.
- Prefer patterns that hold across MANY tasks/splits over one cherry-picked task. The \
random vs similarity (out-of-distribution) split is the most informative contrast.
- Be concise and mechanistic. It is fine to return fewer, stronger hypotheses."""

CRITIQUE_SYSTEM = """You are a skeptical reviewer. For each proposed hypothesis, decide \
whether the claimed attribute actually explains the performance pattern ACROSS ALL tasks in \
the evidence table, or whether it only fits the cherry-picked tasks the author cited. Look \
for counter-cases (tasks where the same attribute predicts the opposite of what happened). \
Down-weight claims resting on `needs_verification` attributes. Be strict."""


def build_generation_prompt(question: str, evidence_md: str, corpus: str, n: int) -> str:
    return f"""# TASK
{question}

Propose up to {n} distinct, falsifiable hypotheses.

# PRECOMPUTED EVIDENCE (authoritative numbers — do not recompute)
{evidence_md}

# KNOWLEDGE BASE (all benchmark cards; cite ids like [task_kcat], [model_prost5])
{corpus}

# OUTPUT FORMAT
{schema.GENERATION_SCHEMA_HINT}"""


def build_critique_prompt(hypotheses_json: str, evidence_md: str) -> str:
    return f"""# PROPOSED HYPOTHESES
{hypotheses_json}

# PRECOMPUTED EVIDENCE (the full per-task table to check each hypothesis against)
{evidence_md}

# YOUR JOB
For each hypothesis, in order, judge whether its attribute explains the pattern across ALL \
tasks/splits above or only the cited ones, and revise confidence accordingly.

# OUTPUT FORMAT
{schema.CRITIQUE_SCHEMA_HINT}"""