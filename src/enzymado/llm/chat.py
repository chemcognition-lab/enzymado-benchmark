"""Grounded, multi-turn chatbot over the Enzymado benchmark.

Gemini gets function-calling tools wrapping :mod:`compare` and :mod:`sources`, so
every cross-model number (a head-to-head, a ranking, a split gap) is computed, not
read off by eye. The whole card corpus goes into the system instruction of each
session (:func:`build_system_instruction`), so descriptive questions are answered
from the cards; the card readers are registered as tools only when the corpus
cannot be loaded. The assistant can:

1. explain how a model works,
2. propose hypotheses for why model X beats model Y (``compare_two_models``, or
   the slower critiqued pipeline ``rigorous_pairwise_hypotheses``),
3. explain why one set of models beats another (``compare_model_sets``),
4. advise on experimental design (``task_leaderboard``, ``leaderboard``,
   ``split_difficulty``).

Importing this module needs neither the Gemini SDK nor an API key: the client is
created on the first :meth:`ChatSession.send`.

This module deliberately does not use ``from __future__ import annotations``: the
SDK builds each tool's schema from the raw annotations, which must be real types.

Example:
    python -c "from enzymado.llm import chat; print(chat.task_leaderboard('kcat'))"
"""

import dataclasses
from pathlib import Path
import typing

from enzymado.llm import compare, corpus, gemini, hypothesize, sources


def _records() -> list[dict]:
    """The cached per-run records."""
    return sources.load_records()


def _split_csv(value: str) -> list[str]:
    """Parse a comma-separated model list, dropping whitespace and empty items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _known_models() -> str:
    return ", ".join(f"`{k}`" for k in sorted(sources.load_models()))


def _known_tasks() -> str:
    return ", ".join(f"`{t}`" for t in sorted(sources.load_task_registry()))


def _task_name(task: str) -> str:
    return task.strip().lower().removeprefix("task_")


# Tools: each returns markdown or plain text and never raises out of the chat loop.
# Their docstrings are the tool descriptions the model sees.


def list_models() -> str:
    """List every representation on the benchmark with its width and parameters.

    Call this whenever you are unsure which key the user means, to map a loose name
    (e.g. "ESM2") to a canonical key (e.g. `esm2_650M`) before other tools. `noplm`
    is the reference without a language model.
    """
    lines = ["Representations (key: embedding width, parameters):"]
    for key, facts in sorted(sources.load_models().items()):
        lines.append(
            f"- `{key}`: width {facts['dim']}, "
            f"{sources.format_params(facts['params'])} parameters"
        )
    return "\n".join(lines)


def list_tasks() -> str:
    """List every benchmark task with its scope and its splits (random, similarity).

    Use this to resolve a task name before `task_info` or `task_leaderboard`, or to
    answer what the benchmark measures.
    """
    by_task: dict[str, set[str]] = {}
    scope: dict[str, str] = {}
    for r in _records():
        by_task.setdefault(r["task"], set()).add(r["split"])
        scope[r["task"]] = r["scope"]
    lines = ["Benchmark tasks (name: scope, splits):"]
    for task in sorted(by_task):
        splits = ", ".join(sorted(by_task[task], key=sources.split_order))
        lines.append(f"- `{task}`: {scope[task]}, {splits}")
    return "\n".join(lines)


def model_info(model: str) -> str:
    """Return one model's card: architecture, pretraining, size, standing, use.

    `model` accepts any casing or the `model_` prefix.
    """
    try:
        key = compare.normalize_key(model)
    except KeyError:
        return f"Unknown model {model!r}. Known models: {_known_models()}."
    path = corpus.DEFAULT_MARKDOWN_DIR / f"model_{key.lower()}.md"
    if not path.exists():
        return f"No card for `{key}`. Run `python -m enzymado.llm` to build them."
    return path.read_text().rstrip()


def task_info(task: str) -> str:
    """Return one task's card: what it predicts, its metric, data and difficulty.

    `task` accepts the registry name (`ph_opt`) or the `task_` prefix.
    """
    path = corpus.DEFAULT_MARKDOWN_DIR / f"task_{_task_name(task)}.md"
    if not path.exists():
        return f"Unknown task {task!r}. Available tasks: {_known_tasks()}."
    return path.read_text().rstrip()


def compare_two_models(model_x: str, model_y: str) -> str:
    """Exact head-to-head of two models, task by task, with ties and leaderboard ranks.

    Returns each model's leaderboard standing, a per-task and per-split table with
    signed gaps, noise thresholds and winners (a gap within the threshold is a tie),
    a win tally and the attributes that differ. This is the evidence for a "why does
    X beat Y" answer; quote its numbers verbatim.
    """
    try:
        h2h = compare.head_to_head(_records(), model_x, model_y)
        diff = compare.attribute_diff(model_x, model_y)
    except KeyError as exc:
        return f"{exc} (known models: {_known_models()})."
    return (
        compare.render_head_to_head_md(h2h)
        + "\n\n"
        + compare.render_attribute_diff_md(diff)
    )


def compare_model_sets(models_a: str, models_b: str) -> str:
    """Compare two sets of models by task coverage and attribute diversity.

    `models_a` and `models_b` are comma-separated keys (e.g. "ankh_large, prost5").
    Returns which set covers each task best, a win tally with ties, and each set's
    diversity: the evidence for "why does combination A beat combination B". Reason
    about complementarity, not just individual strength.
    """
    a, b = _split_csv(models_a), _split_csv(models_b)
    if not a or not b:
        return "Provide at least one model in each set (comma-separated)."
    try:
        combo = compare.combo_coverage(_records(), a, b)
    except KeyError as exc:
        return f"{exc} (known models: {_known_models()})."
    return compare.render_combo_md(combo)


def task_leaderboard(task: str) -> str:
    """Rank every representation by its best score on one task, per split.

    Rows marked within noise are statistically tied with the leader. Use this to
    recommend a model for a task or to check who wins it. `task` accepts the
    registry name or the `task_` prefix.
    """
    name = _task_name(task)
    board = sources.task_leaderboard(_records(), name)
    if not board:
        return f"No results for task {task!r}. Available tasks: {_known_tasks()}."
    lines = [f"### Best score per representation: `{name}`", ""]
    for split, rows in board.items():
        lines += [
            f"**{split} split**:",
            "",
            "| Rank | Representation | Metric | Best | Gap to leader | Baseline |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            gap = "leader" if r["rank"] == 1 else f"{r['gap']}"
            if r["rank"] > 1 and r["within_noise"]:
                gap += " (within noise)"
            lines.append(
                f"| {r['rank']} | `{r['emb']}` | {r['metric']} | {r['value']} | "
                f"{gap} | {r['config']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def leaderboard(name: str = "properties", top: int = 10) -> str:
    """Show the published Bradley-Terry leaderboard, best first.

    `name` is `properties` (the 10 property tasks) or `real_case` (the 8 protein
    engineering datasets); `top` limits the rows. Each competitor is a
    representation with one baseline, with its rating and 95% bootstrap interval.
    """
    if name not in sources.LEADERBOARDS:
        return f"Unknown leaderboard {name!r}; use one of {sources.LEADERBOARDS}."
    ratings = sources.load_ratings(name)
    lines = [
        f"### {name} leaderboard (top {min(top, len(ratings))} of {len(ratings)})",
        "",
        "| Rank | Representation | Baseline | Rating | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(ratings.head(top).itertuples(), start=1):
        lines.append(
            f"| {i} | `{r.embedding_model}` | "
            f"{sources.config_label(r.model_family, r.pool_type)} | "
            f"{r.rating:.1f} | [{r.ci_lo:.1f}, {r.ci_hi:.1f}] |"
        )
    return "\n".join(lines)


def split_difficulty() -> str:
    """Show how much lower the PLMs score on the similarity split than the random one.

    Per property task, the mean score under each split and the drop. Use it to
    advise which split to evaluate on and to reason about generalization.
    """
    rows = sources.split_gap_table(_records())
    lines = [
        "### Random against similarity split (mean over PLMs, heads and poolings)",
        "",
        "A positive drop means the similarity split scores lower.",
        "",
        "| Task | Metric | Random | Similarity | Drop |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['task']} | {r['metric']} | {r['random']} | {r['similarity']} | "
        f"{r['harder_by']:+} |"
        for r in rows
    ]
    return "\n".join(lines)


def rigorous_pairwise_hypotheses(
    model_x: str, model_y: str, task_focus: str = ""
) -> str:
    """Run the slow generate-then-critique pipeline for why X beats Y.

    Call this only when the user explicitly asks for ranked, vetted, critiqued or
    rigorous hypotheses: it makes extra model calls. Leave `task_focus` empty for an
    all-task analysis, or set it to one task name.
    """
    try:
        result = hypothesize.explain_pairwise(
            model_x, model_y, task_focus=task_focus or None
        )
    except KeyError as exc:
        return f"{exc} (known models: {_known_models()})."
    except gemini.GeminiError as exc:
        return f"The rigorous pipeline failed: {exc}"
    return render_result_md(result)


def rigorous_combo_hypotheses(models_a: str, models_b: str) -> str:
    """Run the slow generate-then-critique pipeline for why set A beats set B.

    Call this only when the user explicitly asks for ranked, vetted or critiqued
    hypotheses about two model sets, given as comma-separated keys.
    """
    a, b = _split_csv(models_a), _split_csv(models_b)
    if not a or not b:
        return "Provide at least one model in each set (comma-separated)."
    try:
        result = hypothesize.explain_combo(a, b)
    except KeyError as exc:
        return f"{exc} (known models: {_known_models()})."
    except gemini.GeminiError as exc:
        return f"The rigorous pipeline failed: {exc}"
    return render_result_md(result)


def render_result_md(result: hypothesize.Result) -> str:
    """Ranked hypotheses as markdown, to hand back to the chat model."""
    if not result.hypotheses:
        return "The rigorous pipeline returned no hypotheses."
    lines = ["### Rigorous hypotheses (best supported first)", ""]
    for i, h in enumerate(result.hypotheses, start=1):
        verdict = f" _[{h.verdict}]_" if h.verdict else ""
        lines.append(f"**{i}. {h.claim}**{verdict}, confidence {h.confidence:.2f}")
        lines.append(f"- *Mechanism.* {h.mechanism}")
        lines.append(f"- *Differentiating attribute.* `{h.differentiating_attribute}`")
        lines += [f"- *Evidence.* {item}" for item in h.evidence]
        if h.caveat:
            lines.append(f"- *Caveat.* {h.caveat}")
        if h.falsification_test:
            lines.append(f"- *Falsification test.* {h.falsification_test}")
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
    leaderboard,
    split_difficulty,
    rigorous_pairwise_hypotheses,
    rigorous_combo_hypotheses,
]
#: Card readers, redundant when the corpus is in context; registered only without it.
CARD_TOOLS = [model_info, task_info]
#: The tools registered when the corpus is in the prompt.
CORPUS_TOOLS = [t for t in TOOLS if t not in CARD_TOOLS]

CHAT_SYSTEM = """You are a protein-ML benchmark analyst for the Enzymado benchmark. \
In a chat, you explain how models work, propose falsifiable hypotheses for why one \
model (or set of models) outperforms another, and advise on experimental design \
(which model, prediction head and data split to use).

The COMPLETE knowledge base is appended below: every model, task, baseline, split \
and leaderboard card. Answer descriptive questions straight from the cards instead \
of calling a tool or asking the user for context.

USING NUMBERS:
- Two sources are legitimate: a figure copied verbatim from a card below, or a \
figure returned by a tool. Never invent, recompute, estimate or interpolate one.
- For anything spanning models or tasks (a head-to-head, a gap, a win tally, a \
ranking, a split drop) CALL THE TOOL instead of reading tables by eye: \
`compare_two_models`, `compare_model_sets`, `task_leaderboard`, `leaderboard` and \
`split_difficulty` compute it exactly.
- Overall "who is best" questions are answered by the published Bradley-Terry \
leaderboards (`leaderboard`), not by averaging task scores.
- A gap within the noise threshold is a TIE: retraining alone could reverse it. \
Never present a tie as a win.
- A card's figures describe that card's model or task only; never carry a number \
over to another task or split.

MAKING CLAIMS:
- Every performance claim must rest on BOTH a real number AND a real attribute that \
differs between the models.
- Respect the cards' flags: an attribute marked `needs_verification` is soft \
evidence (cap confidence), and stated caveats hold (e.g. a capability unused in \
this single-sequence benchmark cannot explain a win).
- Prefer patterns that hold across MANY tasks and splits over one task. The random \
against similarity split is the most informative contrast.

TOOL ROUTING:
- "How does model X work?" / "What is task T?": answer from the cards.
- "Why does X beat Y?": `compare_two_models`, then reason from its table and the cards.
- "Why does set A beat set B?": `compare_model_sets` (complementarity, diversity).
- "Which model, head or split should I use for task T?": `task_leaderboard` and \
`split_difficulty`, with the task and model cards for the mechanism.
- "Which model is best overall?": `leaderboard`.
- Call `rigorous_pairwise_hypotheses` or `rigorous_combo_hypotheses` ONLY when the \
user explicitly asks for ranked, vetted, critiqued or rigorous hypotheses.
- `list_models` and `list_tasks` resolve loose names to canonical keys.

Be concise and mechanistic. Say so when a comparison mixes baselines, rests on \
ties, or leans on a task run for only one side."""

#: Introduces the corpus; ``{n}`` is the card count.
CORPUS_PREAMBLE = """

================================ KNOWLEDGE BASE ================================
All {n} benchmark cards follow, each introduced by a `===== CARD: <id> =====` marker.
Cite ids inline (e.g. [task_kcat], [model_prost5]) when a claim leans on a card.
"""

#: Appended instead of the corpus when the cards cannot be loaded.
NO_CORPUS_NOTE = """

KNOWLEDGE BASE UNAVAILABLE: the cards could not be loaded and are NOT in this prompt.
Ignore the instructions above about reading cards inline: call `model_info` or \
`task_info` to fetch a card before using it, and tell the user to run \
`python -m enzymado.llm` to build the knowledge base."""


def build_system_instruction(
    markdown_dir: Path | None = None,
) -> tuple[str, list, int]:
    """``(system_instruction, tools, n_cards)`` for a session.

    Normally :data:`CHAT_SYSTEM` plus every card and :data:`CORPUS_TOOLS`. Without
    cards the session falls back to the card-reading tools instead of failing.
    ``n_cards`` is counted on the corpus, since the preamble quotes the marker too.
    """
    try:
        knowledge = corpus.load_corpus(markdown_dir)
    except (FileNotFoundError, OSError):
        return CHAT_SYSTEM + NO_CORPUS_NOTE, list(TOOLS), 0
    n_cards = knowledge.count(corpus.CARD_MARKER)
    system = CHAT_SYSTEM + CORPUS_PREAMBLE.format(n=n_cards) + "\n" + knowledge
    return system, list(CORPUS_TOOLS), n_cards


@dataclasses.dataclass
class ToolCall:
    name: str
    args: dict[str, typing.Any]
    result_preview: str


@dataclasses.dataclass
class ChatReply:
    text: str
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)


class ChatSession:
    """A multi-turn Gemini chat holding the card corpus, wired to the tools.

    The underlying chat is created on the first :meth:`send`, so building a session
    needs neither the SDK nor ``GEMINI_API_KEY``. Automatic function calling runs
    the tool loop, and the chat's history, tool round-trips included, is the
    conversation's memory.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        markdown_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._markdown_dir = markdown_dir
        self._chat = None
        #: Cards in this session's prompt; ``None`` until the first :meth:`send`.
        self.cards_in_context: int | None = None

    def _ensure_chat(self):
        if self._chat is not None:
            return self._chat
        client = gemini._client()  # raises GeminiError without key or SDK
        from google.genai import types

        system, tools, self.cards_in_context = build_system_instruction(
            self._markdown_dir
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._temperature,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8
            ),
        )
        self._chat = client.chats.create(
            model=gemini.model_name(self._model), config=config
        )
        return self._chat

    def send(self, user_text: str) -> ChatReply:
        """Send one user turn; returns the reply and the tools it called."""
        chat = self._ensure_chat()
        # The function-calling history is cumulative, so note where this turn starts.
        prior_len = len(getattr(chat, "_curated_history", None) or [])
        try:
            response = chat.send_message(user_text)
        except gemini.GeminiError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK failure becomes GeminiError
            raise gemini.GeminiError(f"Gemini chat request failed: {exc}") from exc
        text = response.text
        if not text:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            if reason:
                raise gemini.GeminiError(f"Gemini blocked the request ({reason}).")
            raise gemini.GeminiError("Gemini returned an empty response.")
        history = getattr(response, "automatic_function_calling_history", None) or []
        return ChatReply(text=text, tool_calls=_parse_afc_history(history[prior_len:]))


def _parse_afc_history(history: list) -> list[ToolCall]:
    """This turn's tool calls (name, arguments, result preview).

    Each part of the history carries a ``function_call`` or a ``function_response``
    (the tool's return wrapped as ``{'result': ...}``, or ``{'error': ...}``).
    Responses are matched to calls first in, first out: the model can call one tool
    twice in a turn and the SDK answers in order, so matching the latest call would
    pair each result with the other call's arguments.
    """
    calls: list[dict[str, typing.Any]] = []
    for content in history or []:
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None:
                calls.append(
                    {
                        "name": call.name,
                        "args": dict(call.args or {}),
                        "result": "",
                        "answered": False,
                    }
                )
            reply = getattr(part, "function_response", None)
            if reply is None:
                continue
            payload = reply.response or {}
            value = payload.get("result", payload.get("error", ""))
            text = value if isinstance(value, str) else str(value)
            # The oldest unanswered call of that name; an empty result still counts.
            for entry in calls:
                if entry["name"] == reply.name and not entry["answered"]:
                    entry["result"], entry["answered"] = text, True
                    break
    return [
        ToolCall(
            name=c["name"], args=c["args"], result_preview=_truncate(c["result"], 600)
        )
        for c in calls
    ]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."
