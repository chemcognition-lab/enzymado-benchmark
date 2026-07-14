"""Orchestrator: deterministic evidence → Gemini hypotheses → self-critique → ranked list.

This is the entry point for the hypothesis engine. It wires together the pieces:

    compare.py   builds the precomputed head-to-head / combo evidence tables
    corpus.py    loads all the markdown cards as background context
    prompts.py   assembles the generation + critique prompts
    gemini.py    calls the model
    schema.py    parses hypotheses and merges the critic's verdicts

Public API: :func:`explain_pairwise` and :func:`explain_combo`, plus a CLI :func:`main`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from . import compare, corpus, gemini, prompts, schema
from . import sources as S

# Rank worst→best so the "best" verdict floats to the top after sorting.
_VERDICT_RANK = {
    "supported": 4, "partly-supported": 3, "uncertain": 2,
    "cherry-picked": 1, "contradicted": 0, "": 2,
}

# A hypothesis that cites numbers absent from the authoritative evidence table has its
# self-reported confidence multiplied by this. `verify_evidence` only *flags* fabricated
# numbers; without acting on that flag, the "numbers correct by construction" guarantee
# would stop at display and an ungrounded claim could still rank first.
_UNGROUNDED_CONFIDENCE_FACTOR = 0.5


@dataclass
class Result:
    question: str
    evidence_md: str
    hypotheses: list[schema.Hypothesis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "evidence_md": self.evidence_md,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


def _penalize_ungrounded(hyps: list[schema.Hypothesis]) -> list[schema.Hypothesis]:
    """Discount confidence for hypotheses citing numbers not in the evidence table."""
    for h in hyps:
        if h.unverified_numbers:
            h.confidence *= _UNGROUNDED_CONFIDENCE_FACTOR
    return hyps


def _rank(hyps: list[schema.Hypothesis]) -> list[schema.Hypothesis]:
    return sorted(
        hyps,
        key=lambda h: (_VERDICT_RANK.get(h.verdict, 2), h.confidence),
        reverse=True,
    )


def _run(question: str, evidence_md: str, n: int, model: str | None,
         critique: bool, markdown_dir=None) -> Result:
    kb = corpus.load_corpus(markdown_dir)

    gen_text = gemini.generate(
        prompts.build_generation_prompt(question, evidence_md, kb, n),
        system=prompts.SYSTEM, model=model, temperature=0.7, json_mode=True,
    )
    try:
        hyps = schema.parse_hypotheses(gen_text)[:n]
    except (json.JSONDecodeError, ValueError) as e:
        # Surface as a GeminiError so callers (CLI/app) handle it uniformly.
        raise gemini.GeminiError(f"Could not parse the model's hypotheses: {e}") from e
    schema.verify_evidence(hyps, evidence_md)  # flag cited numbers absent from the table

    if critique and hyps:
        # Tag each hypothesis with its index so the critic's verdicts can be matched
        # back positionally-independently (see schema.apply_critique).
        payload = json.dumps(
            [{"index": i, **h.to_dict()} for i, h in enumerate(hyps)], indent=2)
        crit_text = gemini.generate(
            prompts.build_critique_prompt(payload, evidence_md),
            system=prompts.CRITIQUE_SYSTEM, model=model, temperature=0.2, json_mode=True,
        )
        try:
            schema.apply_critique(hyps, crit_text)
        except (json.JSONDecodeError, ValueError):
            pass  # keep un-critiqued hypotheses rather than failing the whole run

    # Applied after critique, which may overwrite confidence with its own revised value.
    _penalize_ungrounded(hyps)
    return Result(question=question, evidence_md=evidence_md, hypotheses=_rank(hyps))


def explain_pairwise(x: str, y: str, *, task_focus: str | None = None, n: int = 5,
                     model: str | None = None, critique: bool = True) -> Result:
    """Hypotheses for *why model X beats (or loses to) model Y*."""
    records = S.load_records()
    h2h = compare.head_to_head(records, x, y)
    diff = compare.attribute_diff(x, y)
    evidence_md = (compare.render_head_to_head_md(h2h) + "\n\n"
                   + compare.render_attribute_diff_md(diff))
    focus = f" Focus on the task '{task_focus}'." if task_focus else ""
    question = (f"Propose hypotheses explaining the performance differences between "
                f"`{h2h['x']}` (X) and `{h2h['y']}` (Y) on this benchmark.{focus}")
    return _run(question, evidence_md, n, model, critique)


def explain_combo(set_a: list[str], set_b: list[str], *, n: int = 5,
                  model: str | None = None, critique: bool = True) -> Result:
    """Hypotheses for *why the combination of models A beats combination B*."""
    records = S.load_records()
    combo = compare.combo_coverage(records, set_a, set_b)
    evidence_md = compare.render_combo_md(combo)
    question = (f"Propose hypotheses explaining why the model set A={combo['set_a']} "
                f"covers this benchmark better or worse than set B={combo['set_b']}. "
                "Reason about complementarity/diversity, not just individual strength.")
    return _run(question, evidence_md, n, model, critique)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(result: Result) -> None:
    print(result.evidence_md)
    print("\n" + "=" * 72 + "\nHYPOTHESES (best-supported first)\n" + "=" * 72)
    if not result.hypotheses:
        print("(none returned)")
        return
    for i, h in enumerate(result.hypotheses, 1):
        verdict = f"[{h.verdict}] " if h.verdict else ""
        print(f"\n{i}. {verdict}(confidence {h.confidence:.2f})\n   {h.claim}")
        print(f"   Mechanism: {h.mechanism}")
        print(f"   Attribute: {h.differentiating_attribute}")
        for e in h.evidence:
            print(f"     • {e}")
        if h.caveat:
            print(f"   Caveat: {h.caveat}")
        if h.falsification_test:
            print(f"   Falsify: {h.falsification_test}")
        if h.unverified_numbers:
            nums = ", ".join(str(n) for n in h.unverified_numbers)
            print(f"   ⚠ Unverified numbers (not in the evidence table): {nums}")
        if h.verdict_reason:
            print(f"   Critic: {h.verdict_reason}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Propose hypotheses about Enzymado benchmark results.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair", nargs=2, metavar=("X", "Y"), help="two model keys to compare")
    g.add_argument("--combo", nargs=2, metavar=("A,B", "C,D"),
                   help="two comma-separated model sets to compare")
    p.add_argument("--task", help="optional task to focus a pairwise comparison on")
    p.add_argument("--n", type=int, default=5, help="max hypotheses (default 5)")
    p.add_argument("--model", help="Gemini model id (default: env GEMINI_MODEL or gemini-2.5-flash)")
    p.add_argument("--no-critique", action="store_true", help="skip the self-critique pass")
    p.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = p.parse_args(argv)

    critique = not args.no_critique
    if args.pair:
        result = explain_pairwise(*args.pair, task_focus=args.task, n=args.n,
                                  model=args.model, critique=critique)
    else:
        set_a = [s for s in args.combo[0].split(",") if s]
        set_b = [s for s in args.combo[1].split(",") if s]
        result = explain_combo(set_a, set_b, n=args.n, model=args.model, critique=critique)

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        print()
    else:
        _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())