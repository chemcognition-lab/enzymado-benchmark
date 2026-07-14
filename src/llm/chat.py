"""Conversational, grounded chatbot over the Enzymado benchmark.

The form-driven hypothesis engine (see :mod:`llm.hypothesize`) guaranteed correct numbers
by making the user pick the two models and computing the head-to-head deterministically.
This module keeps that guarantee inside a *free-text* interface: Gemini is given a set of
**function-calling tools** that wrap the deterministic layer (:mod:`llm.compare`,
:mod:`llm.sources`) and the authored cards, so the model must *call a tool* to obtain any
benchmark number rather than inventing it.

The four things the assistant is built to do:

1. explain how a given model works (``model_info``),
2. hypothesize why model X beats model Y (``compare_two_models`` → inline; or the opt-in
   rigorous generate-then-critique pipeline via ``rigorous_pairwise_hypotheses``),
3. explain why one *set* of models beats another (``compare_model_sets`` / ``rigorous_combo``),
4. help with experimental-design choices (``task_info`` / ``task_leaderboard`` / ``split_difficulty``).

Public API: the tool functions, :data:`TOOLS`, :data:`CHAT_SYSTEM`, and :class:`ChatSession`.
Importing this module needs neither the Gemini SDK nor an API key — the client is created
lazily on the first :meth:`ChatSession.send`, mirroring :mod:`llm.gemini`.

NOTE: unlike the rest of the package, this module deliberately does NOT use
``from __future__ import annotations``. The Gemini SDK builds each tool's JSON schema by
reading the function's raw ``__annotations__`` (it calls ``isinstance(value, annotation)``),
so PEP 563 stringized annotations would break automatic function calling for every tool that
takes an argument. Native ``str`` / ``list[...]`` / ``str | None`` hints work at our 3.12 floor.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import compare, gemini, hypothesize, schema
from . import sources as S

MARKDOWN_DIR = S.REPO_ROOT / "markdown"


# ---------------------------------------------------------------------------
# Small helpers shared by the tools
# ---------------------------------------------------------------------------

def _records() -> list[dict]:
    """The cached tidy per-run records (``sources.load_records`` is ``lru_cache``d)."""
    return S.load_records()


def _split_csv(value: str) -> list[str]:
    """Parse a comma-separated model list, trimming whitespace and empty items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _known_models_str() -> str:
    return ", ".join(f"`{k}`" for k in sorted(S.load_models()))


# ---------------------------------------------------------------------------
# Tools — each returns markdown/plain text and never raises out of the chat loop.
# The docstrings ARE the tool descriptions the model sees, so keep them purposeful.
# ---------------------------------------------------------------------------

def list_models() -> str:
    """List every foundation model available on this benchmark, with its embedding dimension.

    Call this first whenever you are unsure which model key the user means, so you can map a
    loose name (e.g. "ESM2") to a canonical key (e.g. `esm2_650M`) before other tools.
    """
    models = S.load_models()
    lines = ["Available models (canonical key — embedding dim):"]
    for key in sorted(models):
        lines.append(f"- `{key}` — dim {models[key]['dim']}")
    return "\n".join(lines)


def list_tasks() -> str:
    """List every benchmark task and which data splits (random / similarity) it has.

    Use this to resolve a task name before calling `task_info` or `task_leaderboard`, or to
    answer "what can this benchmark measure?" questions.
    """
    by_task: dict[str, set[str]] = {}
    for r in _records():
        by_task.setdefault(r["task"], set()).add(r["split_family"])
    lines = ["Benchmark tasks (name — available splits):"]
    for task in sorted(by_task):
        splits = ", ".join(
            compare.SPLIT_DISPLAY.get(s, s)
            for s in sorted(by_task[task], key=lambda s: 0 if s == "random" else 1)
        )
        lines.append(f"- `{S._display_task(task)}` — {splits}")
    return "\n".join(lines)


def model_info(model: str) -> str:
    """Return the knowledge-base card for one model: architecture, pretraining, size, use.

    Use this to explain how a model works, or to gather a model's attributes when reasoning
    about an experimental-design question. `model` accepts any casing or the `model_` prefix.
    """
    try:
        key = compare.normalize_key(model)
    except KeyError:
        return f"Unknown model {model!r}. Known models: {_known_models_str()}."
    path = MARKDOWN_DIR / f"model_{key.lower()}.md"
    if not path.exists():
        return f"No card found for `{key}` (expected {path.name}). Known models: {_known_models_str()}."
    return path.read_text().rstrip()


def task_info(task: str) -> str:
    """Return the knowledge-base card for one task: what it predicts, metric, data, difficulty.

    Use this for "what is task X / how is it evaluated" questions and to inform experimental
    design. `task` accepts the display name (`ph_opt`), the results name (`phopt`), or a
    `task_` prefix.
    """
    cand = task.strip().lower().removeprefix("task_")
    cand = S.RESULTS_TO_REGISTRY_TASK.get(cand, cand)  # phopt -> ph_opt
    path = MARKDOWN_DIR / f"task_{cand}.md"
    if not path.exists():
        available = ", ".join(f"`{t}`" for t in sorted(S.load_task_registry()))
        return f"Unknown task {task!r}. Available tasks: {available}."
    return path.read_text().rstrip()


def compare_two_models(model_x: str, model_y: str) -> str:
    """Exact head-to-head benchmark comparison of two models (numbers correct by construction).

    Returns the per-task/per-split table (with sign-aware deltas and a winner tally), the
    overall mean-rank standing, and the models' differentiating attributes. This is the
    grounded evidence for a "why does X beat Y" answer — quote its numbers verbatim.
    """
    try:
        h2h = compare.head_to_head(_records(), model_x, model_y)
        diff = compare.attribute_diff(model_x, model_y)
    except KeyError as e:
        return f"{e} (known models: {_known_models_str()})."
    return compare.render_head_to_head_md(h2h) + "\n\n" + compare.render_attribute_diff_md(diff)


def compare_model_sets(models_a: str, models_b: str) -> str:
    """Compare two *sets* of models by task coverage and attribute diversity.

    `models_a` and `models_b` are comma-separated model keys (e.g. "ankh_large, prost5").
    Returns which set covers each task best, a win tally, and each set's diversity profile —
    the grounded evidence for "why does combination A beat combination B" (reason about
    complementarity, not just individual strength).
    """
    a, b = _split_csv(models_a), _split_csv(models_b)
    if not a or not b:
        return "Provide at least one model in each set (comma-separated)."
    try:
        combo = compare.combo_coverage(_records(), a, b)
    except KeyError as e:
        return f"{e} (known models: {_known_models_str()})."
    return compare.render_combo_md(combo)


def task_leaderboard(task: str) -> str:
    """Rank all models by their best score on one task, per split (random & similarity).

    Use this to recommend which model to use for a task, or to ground a claim about who wins
    a specific task. `task` accepts the display, results, or `task_`-prefixed name.
    """
    cand = task.strip().lower().removeprefix("task_")
    registry_name = S.RESULTS_TO_REGISTRY_TASK.get(cand, cand)
    results_name = S.results_task_name(registry_name)
    board = S.task_leaderboard(_records(), results_name)
    if not board:
        available = ", ".join(f"`{t}`" for t in sorted(S.load_task_registry()))
        return f"No leaderboard for task {task!r}. Available tasks: {available}."
    lines = [f"### Leaderboard: `{S._display_task(results_name)}`", ""]
    for sfam in sorted(board, key=lambda s: 0 if s == "random" else 1):
        lines.append(f"**{compare.SPLIT_DISPLAY.get(sfam, sfam)} split** (best config per model):")
        lines.append("")
        lines.append("| Rank | Model | Metric | Best value | Head |")
        lines.append("|---|---|---|---|---|")
        for row in board[sfam]:
            lines.append(
                f"| {row['rank']} | `{row['emb']}` | {row['metric']} | "
                f"{row['value']} | {row['head']} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def split_difficulty() -> str:
    """Show how much harder the similarity (out-of-distribution) split is than the random split.

    Returns, per task, the mean score under each split and the drop — useful for advising
    which split to evaluate on and for reasoning about generalization.
    """
    rows = S.split_gap_table(_records())
    if not rows:
        return "No split-gap data available."
    lines = [
        "### Random vs similarity split (mean over all models/heads)",
        "",
        "`harder_by` > 0 means the similarity (out-of-distribution) split is worse.",
        "",
        "| Task | Metric | Random | Similarity | Harder by |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['metric']} | {r['random']} | "
            f"{r['similarity']} | {r['harder_by']:+} |")
    return "\n".join(lines)


def rigorous_pairwise_hypotheses(model_x: str, model_y: str, task_focus: str = "") -> str:
    """Run the rigorous generate-then-critique pipeline for "why X beats Y" (SLOW, opt-in).

    Only call this when the user explicitly asks for *ranked*, *vetted*, *critiqued*, or
    *rigorous* hypotheses — it makes extra model calls. Returns hypotheses each with a
    skeptic's verdict and a confidence, best-supported first. Leave `task_focus` empty for
    an all-tasks analysis, or set it to one task name to focus.
    """
    try:
        result = hypothesize.explain_pairwise(
            model_x, model_y, task_focus=task_focus or None)
    except KeyError as e:
        return f"{e} (known models: {_known_models_str()})."
    except gemini.GeminiError as e:
        return f"The rigorous pipeline failed: {e}"
    return render_result_md(result)


def rigorous_combo_hypotheses(models_a: str, models_b: str) -> str:
    """Run the rigorous generate-then-critique pipeline for "why set A beats set B" (SLOW, opt-in).

    Only call this when the user explicitly asks for ranked/vetted/critiqued hypotheses about
    two model *sets*. `models_a`/`models_b` are comma-separated model keys. Returns ranked,
    critic-vetted hypotheses.
    """
    a, b = _split_csv(models_a), _split_csv(models_b)
    if not a or not b:
        return "Provide at least one model in each set (comma-separated)."
    try:
        result = hypothesize.explain_combo(a, b)
    except KeyError as e:
        return f"{e} (known models: {_known_models_str()})."
    except gemini.GeminiError as e:
        return f"The rigorous pipeline failed: {e}"
    return render_result_md(result)


def render_result_md(result: hypothesize.Result) -> str:
    """Render a :class:`hypothesize.Result` (ranked hypotheses) to markdown.

    The string counterpart to ``app._render_hypotheses`` — used so the rigorous pipeline's
    output can be handed back to the chat model as a tool result.
    """
    if not result.hypotheses:
        return "The rigorous pipeline returned no hypotheses."
    lines = ["### Rigorous hypotheses (best-supported first)", ""]
    for i, h in enumerate(result.hypotheses, 1):
        verdict = f" _[{h.verdict}]_" if h.verdict else ""
        lines.append(f"**{i}. {h.claim}**{verdict} · confidence {h.confidence:.2f}")
        lines.append(f"- *Mechanism.* {h.mechanism}")
        lines.append(f"- *Differentiating attribute.* `{h.differentiating_attribute}`")
        for e in h.evidence:
            lines.append(f"- *Evidence.* {e}")
        if h.caveat:
            lines.append(f"- *Caveat.* {h.caveat}")
        if h.falsification_test:
            lines.append(f"- *Falsification test.* {h.falsification_test}")
        if h.unverified_numbers:
            nums = ", ".join(str(n) for n in h.unverified_numbers)
            lines.append(f"- ⚠ Cites numbers not in the evidence table: {nums}")
        if h.verdict_reason:
            lines.append(f"- *Critic.* {h.verdict_reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


# The order here is the order the model sees the tools in.
TOOLS = [
    list_models,
    list_tasks,
    model_info,
    task_info,
    compare_two_models,
    compare_model_sets,
    task_leaderboard,
    split_difficulty,
    rigorous_pairwise_hypotheses,
    rigorous_combo_hypotheses,
]


# ---------------------------------------------------------------------------
# System prompt — capabilities, grounding rules, tool routing
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """You are a protein-ML benchmark analyst for the Enzymado benchmark. You help \
the user through a chat: you explain how models work, propose falsifiable hypotheses for why \
one model (or set of models) outperforms another, and advise on experimental-design choices \
(which model / prediction head / data split to use).

HARD GROUNDING RULES — these are non-negotiable:
- NEVER invent, recompute, or estimate a benchmark number. Any figure you state MUST come \
from a tool result, quoted verbatim (same digits). If you do not have a number from a tool, \
call the tool first; do not guess.
- Every performance claim must rest on BOTH a real number (from a comparison/leaderboard tool) \
AND a real attribute that differs between the models (from `model_info` / the attribute table). \
No number + no attribute → no claim.
- Respect the cards' flags: an attribute marked `needs_verification` is soft evidence (cap \
confidence); honor stated caveats (e.g. a retrieval capability that is DISABLED in this \
single-sequence benchmark must not be credited for a win).
- Prefer patterns that hold across MANY tasks/splits over a single cherry-picked task. The \
random vs similarity (out-of-distribution) split is the most informative contrast.

TOOL ROUTING:
- "How does model X work?" → `model_info`. "What is task T?" → `task_info`.
- "Why does X beat Y?" → `compare_two_models`, then reason inline from that evidence.
- "Why does set A beat set B?" → `compare_model_sets` (reason about complementarity/diversity).
- "Which model/head/split should I use for task T?" → `task_leaderboard`, `task_info`, \
`split_difficulty`, and `model_info` as needed.
- Call `rigorous_pairwise_hypotheses` / `rigorous_combo_hypotheses` ONLY when the user \
explicitly asks for ranked, vetted, critiqued, or rigorous hypotheses — they are slow.
- If you are unsure of a model or task name, call `list_models` / `list_tasks` first.

Be concise and mechanistic. It is fine to give fewer, stronger points. When a comparison mixes \
model configurations or leans on a task run for only one side, say so."""


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result_preview: str


@dataclass
class ChatReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)


class ChatSession:
    """A multi-turn Gemini chat wired to the grounded benchmark tools.

    The underlying ``genai`` chat is created lazily on the first :meth:`send`, so constructing
    a session (and importing this module) requires neither the SDK nor ``GEMINI_API_KEY``.
    Automatic function calling runs the tool loop; the model's curated history — including the
    tool round-trips — is the conversation's memory.
    """

    def __init__(self, *, model: str | None = None, temperature: float = 0.3):
        self._model = model
        self._temperature = temperature
        self._chat = None

    def _ensure_chat(self):
        if self._chat is not None:
            return self._chat
        client = gemini._client()  # raises GeminiError on missing key / uninstalled SDK
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM,
            temperature=self._temperature,
            tools=TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8),
        )
        self._chat = client.chats.create(
            model=gemini.model_name(self._model), config=config)
        return self._chat

    def send(self, user_text: str) -> ChatReply:
        """Send one user turn; returns the final grounded reply plus tool metadata."""
        chat = self._ensure_chat()
        # `automatic_function_calling_history` is cumulative (prior curated history + this
        # turn's tool round-trips), so remember where the prior history ends to slice out
        # only this turn's calls — mirroring the SDK's own record-history slice.
        prior_len = len(getattr(chat, "_curated_history", None) or [])
        try:
            resp = chat.send_message(user_text)
        except gemini.GeminiError:
            raise
        except Exception as e:  # network, rate limit, safety block, tool loop overflow, …
            raise gemini.GeminiError(f"Gemini chat request failed: {e}") from e

        text = resp.text
        if not text:
            reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            if reason:
                raise gemini.GeminiError(f"Gemini blocked the request ({reason}).")
            raise gemini.GeminiError("Gemini returned an empty response.")

        full_history = getattr(resp, "automatic_function_calling_history", None) or []
        tool_calls, tool_outputs = _parse_afc_history(full_history[prior_len:])
        ungrounded = _ungrounded_numbers(text, tool_outputs)
        return ChatReply(text=text, tool_calls=tool_calls, ungrounded_numbers=ungrounded)


def _parse_afc_history(history) -> tuple[list[ToolCall], list[str]]:
    """Extract (tool calls for display, tool-output strings for grounding) from AFC history.

    ``automatic_function_calling_history`` is a ``list[Content]`` whose parts carry either a
    ``function_call`` (name + args) or a ``function_response`` (name + wrapped result). AFC
    wraps a tool's return as ``{'result': ...}`` and a raised exception as ``{'error': ...}``.
    """
    calls: list[dict[str, Any]] = []
    outputs: list[str] = []
    if not history:
        return [], []
    for content in history:
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                calls.append({"name": fc.name, "args": dict(fc.args or {}), "result": ""})
            fr = getattr(part, "function_response", None)
            if fr is not None:
                response = fr.response or {}
                value = response.get("result", response.get("error", ""))
                text = value if isinstance(value, str) else str(value)
                outputs.append(text)
                # Attach the response to the most recent matching, still-unfilled call.
                for entry in reversed(calls):
                    if entry["name"] == fr.name and not entry["result"]:
                        entry["result"] = text
                        break
    tool_calls = [
        ToolCall(name=c["name"], args=c["args"],
                 result_preview=_truncate(c["result"], 600))
        for c in calls
    ]
    return tool_calls, outputs


def _ungrounded_numbers(reply_text: str, tool_outputs: list[str]) -> list[float]:
    """Numbers stated in the reply that don't appear in any tool output (possible fabrication).

    Compensates for the loss of ``schema.verify_evidence`` in a free-text answer: the reply's
    numbers are matched (tolerantly, via ``schema._is_grounded``) against the numbers the tools
    actually returned.
    """
    allowed = set()
    for out in tool_outputs:
        allowed.update(schema.extract_numbers(out))
    flagged: list[float] = []
    for num in schema.extract_numbers(reply_text):
        if not schema._is_grounded(num, allowed) and num not in flagged:
            flagged.append(num)
    return flagged


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"
