"""Hypotheses for why one representation (or set) beats another, ranked and critiqued.

Exact evidence from :mod:`compare`, the card corpus from :mod:`corpus` and the
prompts from :mod:`prompts` go to Gemini, which proposes hypotheses; a second,
critic call judges each one against the whole evidence table, and the result is
ranked by verdict and confidence. Needs ``GEMINI_API_KEY`` and the ``llm`` extra.

Example:
    python -m enzymado.llm.hypothesize --pair prost5 esm2_650M
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import typing

from enzymado.llm import compare, corpus, gemini, prompts, schema, sources

# Best verdict first after sorting.
_VERDICT_RANK = {
    "supported": 4,
    "partly-supported": 3,
    "uncertain": 2,
    "cherry-picked": 1,
    "contradicted": 0,
    "": 2,
}


@dataclasses.dataclass
class Result:
    question: str
    evidence_md: str
    hypotheses: list[schema.Hypothesis] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "question": self.question,
            "evidence_md": self.evidence_md,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


def _rank(hyps: list[schema.Hypothesis]) -> list[schema.Hypothesis]:
    return sorted(
        hyps,
        key=lambda h: (_VERDICT_RANK.get(h.verdict, 2), h.confidence),
        reverse=True,
    )


def _run(
    question: str,
    evidence_md: str,
    n: int,
    model: str | None,
    *,
    critique: bool,
) -> Result:
    knowledge = corpus.load_corpus()
    generated = gemini.generate(
        prompts.build_generation_prompt(question, evidence_md, knowledge, n),
        system=prompts.SYSTEM,
        model=model,
        temperature=0.7,
        json_mode=True,
    )
    try:
        hyps = schema.parse_hypotheses(generated)[:n]
    except (json.JSONDecodeError, ValueError) as exc:
        raise gemini.GeminiError(
            f"Could not parse the model's hypotheses: {exc}"
        ) from exc
    if critique and hyps:
        payload = json.dumps(
            [{"index": i, **h.to_dict()} for i, h in enumerate(hyps)], indent=2
        )
        critiqued = gemini.generate(
            prompts.build_critique_prompt(payload, evidence_md),
            system=prompts.CRITIQUE_SYSTEM,
            model=model,
            temperature=0.2,
            json_mode=True,
        )
        try:
            schema.apply_critique(hyps, critiqued)
        except (json.JSONDecodeError, ValueError):
            pass  # keep the uncritiqued hypotheses rather than fail the run
    return Result(question=question, evidence_md=evidence_md, hypotheses=_rank(hyps))


def explain_pairwise(
    x: str,
    y: str,
    *,
    task_focus: str | None = None,
    n: int = 5,
    model: str | None = None,
    critique: bool = True,
) -> Result:
    """Hypotheses for why representation X beats, or loses to, representation Y."""
    records = sources.load_records()
    h2h = compare.head_to_head(records, x, y)
    evidence_md = (
        compare.render_head_to_head_md(h2h)
        + "\n\n"
        + compare.render_attribute_diff_md(compare.attribute_diff(x, y))
    )
    focus = f" Focus on the task '{task_focus}'." if task_focus else ""
    question = (
        "Propose hypotheses explaining the performance differences between "
        f"`{h2h['x']}` (X) and `{h2h['y']}` (Y) on this benchmark.{focus}"
    )
    return _run(question, evidence_md, n, model, critique=critique)


def explain_combo(
    set_a: list[str],
    set_b: list[str],
    *,
    n: int = 5,
    model: str | None = None,
    critique: bool = True,
) -> Result:
    """Hypotheses for why the set of representations A beats set B."""
    combo = compare.combo_coverage(sources.load_records(), set_a, set_b)
    question = (
        f"Propose hypotheses explaining why the model set A={combo['set_a']} covers "
        f"this benchmark better or worse than set B={combo['set_b']}. Reason about "
        "complementarity and diversity, not just individual strength."
    )
    return _run(question, compare.render_combo_md(combo), n, model, critique=critique)


def _print(result: Result) -> None:
    print(result.evidence_md)
    print("\n" + "=" * 72 + "\nHYPOTHESES (best supported first)\n" + "=" * 72)
    if not result.hypotheses:
        print("(none returned)")
        return
    for i, h in enumerate(result.hypotheses, start=1):
        verdict = f"[{h.verdict}] " if h.verdict else ""
        print(f"\n{i}. {verdict}(confidence {h.confidence:.2f})\n   {h.claim}")
        print(f"   Mechanism: {h.mechanism}")
        print(f"   Attribute: {h.differentiating_attribute}")
        for item in h.evidence:
            print(f"     - {item}")
        if h.caveat:
            print(f"   Caveat: {h.caveat}")
        if h.falsification_test:
            print(f"   Falsify: {h.falsification_test}")
        if h.verdict_reason:
            print(f"   Critic: {h.verdict_reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", nargs=2, metavar=("X", "Y"))
    group.add_argument("--combo", nargs=2, metavar=("A,B", "C,D"))
    parser.add_argument("--task", help="task to focus a pairwise comparison on")
    parser.add_argument("--n", type=int, default=5, help="maximum hypotheses")
    parser.add_argument("--model", help="Gemini model id (default: env GEMINI_MODEL)")
    parser.add_argument("--no-critique", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    critique = not args.no_critique
    if args.pair:
        result = explain_pairwise(
            *args.pair, task_focus=args.task, n=args.n, model=args.model,
            critique=critique,
        )  # fmt: skip
    else:
        set_a = [s for s in args.combo[0].split(",") if s]
        set_b = [s for s in args.combo[1].split(",") if s]
        result = explain_combo(
            set_a, set_b, n=args.n, model=args.model, critique=critique
        )
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        print()
    else:
        _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
