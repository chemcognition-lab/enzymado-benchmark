"""Prompts for the generate-then-critique loop.

The evidence tables from :mod:`compare` come first; the card corpus follows as
background. The rules tie every claim to a quoted number and a real attribute
difference, treat gaps within the noise threshold as ties, and respect the cards'
``needs_verification`` flags and caveats.

Example:
    python -c "from enzymado.llm import prompts; print(prompts.SYSTEM)"
"""

from __future__ import annotations

from enzymado.llm import schema

SYSTEM = """You are a protein-ML benchmark analyst. You propose falsifiable \
hypotheses for WHY one model (or set of models) outperforms another on the \
Enzymado benchmark.

Hard rules:
- Ground every claim in the PRECOMPUTED EVIDENCE tables and the cards. The numbers \
in the evidence tables are authoritative; never invent or recompute numbers. Copy \
figures into `evidence` EXACTLY as they appear (same digits).
- Each hypothesis must cite specific numbers (in `evidence`) AND name a real \
attribute that differs between the models (in `differentiating_attribute`). No \
number and no attribute, no claim.
- A task marked `tie` is within the noise threshold: retraining alone could flip \
it. Never explain a tie as a win.
- An attribute flagged `needs_verification` is soft evidence: cap confidence at \
0.4 for any claim that leans on it.
- Respect stated caveats, e.g. a retrieval capability that is unused in this \
single-sequence benchmark must not be credited for a win.
- Each score is the model's BEST baseline (head and pooling) for that task, so a \
comparison can mix baselines, and tasks run for only one side are LEFT OUT (see \
'Methodology & caveats'). If a hypothesis leans on a mixed-baseline comparison, a \
tie or a left-out task, state that in its `caveat`; otherwise leave it empty.
- Prefer patterns that hold across MANY tasks and splits over one cherry-picked \
task. The random against similarity split is the most informative contrast.
- Be concise and mechanistic. Fewer, stronger hypotheses are fine."""

CRITIQUE_SYSTEM = """You are a skeptical reviewer. For each proposed hypothesis, \
decide whether the claimed attribute explains the performance pattern ACROSS ALL \
tasks in the evidence table, or only the cherry-picked tasks the author cited. Look \
for counter-cases (tasks where the same attribute predicts the opposite of what \
happened), treat ties as no evidence either way, and down-weight claims resting on \
`needs_verification` attributes. Be strict."""


def build_generation_prompt(
    question: str, evidence_md: str, corpus: str, n: int
) -> str:
    return f"""# TASK
{question}

Propose up to {n} distinct, falsifiable hypotheses.

# PRECOMPUTED EVIDENCE (authoritative numbers; do not recompute)
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
For each hypothesis, in order, judge whether its attribute explains the pattern \
across ALL tasks and splits above or only the cited ones, and revise its confidence.

# OUTPUT FORMAT
{schema.CRITIQUE_SCHEMA_HINT}"""
