"""Tests for the assistant's exact comparisons and its hypothesis engine.

Torch-free and network-free: the comparisons run on the published results, and
Gemini is replaced by a stub, so no API key or SDK is needed.
"""

from __future__ import annotations

import json

import pytest

from enzymado.llm import compare, hypothesize, prompts, schema, sources


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return sources.load_records()


def test_normalize_key_handles_case_and_card_prefix():
    assert compare.normalize_key("esm2_650m") == "esm2_650M"
    assert compare.normalize_key("model_ankh3_xl") == "ankh3_xl"
    assert compare.normalize_key("NOPLM") == "noplm"
    with pytest.raises(KeyError):
        compare.normalize_key("not_a_model")


def test_head_to_head_winners_follow_gap_and_noise_threshold(records):
    h2h = compare.head_to_head(records, "prost5", "esm2_650m")
    assert (h2h["x"], h2h["y"]) == ("prost5", "esm2_650M")
    assert len(h2h["rows"]) == 28  # 10 property tasks x 2 splits + 8 real case
    for r in h2h["rows"]:
        if r["threshold"] is not None and abs(r["delta"]) <= r["threshold"]:
            assert r["winner"] == "tie"
        elif r["delta"] > 0:
            assert r["winner"] == h2h["x"]
        elif r["delta"] < 0:
            assert r["winner"] == h2h["y"]
    assert sum(h2h["tally"].values()) == len(h2h["rows"])
    assert h2h["dropped"] == []


def test_head_to_head_reports_some_ties(records):
    h2h = compare.head_to_head(records, "prost5", "esm2_650m")
    assert h2h["tally"]["tie"] > 0


def test_standing_comes_from_the_leaderboards(records):
    standing = compare.head_to_head(records, "prost5", "esm2_650m")["standing"]
    ratings = sources.load_ratings("properties")
    assert standing["prost5"]["properties"]["best"]["rating"] == round(
        float(ratings.rating[ratings.embedding_model == "prost5"].max()), 1
    )
    assert set(standing["esm2_650M"]) == set(sources.LEADERBOARDS)


def test_attribute_diff_uses_the_attribute_table():
    diff = compare.attribute_diff("ankh3_xl", "esm2_650m")
    assert diff["flags"]["ankh3_xl"]["needs_verification"] is True
    assert diff["flags"]["esm2_650M"]["needs_verification"] is False
    differing = {d["field"]: d for d in diff["differing"]}
    assert differing["embedding_dim"]["x"] == 2560
    assert differing["parameters"]["x"] == "3.48B"


def test_attribute_diff_shares_width_when_equal():
    diff = compare.attribute_diff("prost5", "prot5_xl_uniref50")
    assert "embedding_dim" in {d["field"] for d in diff["shared"]}


def test_combo_coverage_tally_and_membership(records):
    combo = compare.combo_coverage(
        records, ["ankh_large", "prost5"], ["esm2_650m", "esmc_600m"]
    )
    assert sum(combo["tally"].values()) == len(combo["rows"])
    for r in combo["rows"]:
        assert r["a_member"] in combo["set_a"]
        assert r["b_member"] in combo["set_b"]


def test_dropped_tasks_detects_a_coverage_gap():
    best_x = {("random", "kcat"): {}, ("random", "km"): {}}
    best_y = {("random", "kcat"): {}}
    dropped = compare._dropped_tasks({"esm2_650M": best_x, "prost5": best_y})
    assert dropped == [
        {"task": "km", "split": "random", "present_only_for": "esm2_650M"}
    ]


def test_methodology_block_in_both_views(records):
    h2h = compare.render_head_to_head_md(
        compare.head_to_head(records, "esm2_650m", "prost5")
    )
    combo = compare.render_combo_md(
        compare.combo_coverage(records, ["ankh_large"], ["esm2_650m"])
    )
    for md in (h2h, combo):
        assert "Methodology & caveats" in md
        assert "noise threshold" in md


def test_generation_prompt_embeds_evidence_and_schema():
    prompt = prompts.build_generation_prompt(
        question="why does X beat Y",
        evidence_md="| Task | Split |\n|---|---|\nEVIDENCE_MARKER",
        corpus="===== CARD: task_kcat =====\nbody",
        n=3,
    )
    for marker in ("EVIDENCE_MARKER", "task_kcat", "falsification_test", "caveat"):
        assert marker in prompt


def test_apply_critique_matches_by_index_when_reordered():
    hyps = [schema.Hypothesis(claim="first", mechanism="m"),
            schema.Hypothesis(claim="second", mechanism="m")]  # fmt: skip
    critique = json.dumps(
        [{"index": 1, "verdict": "supported", "confidence": 0.9},
         {"index": 0, "verdict": "contradicted", "confidence": 0.1}]
    )  # fmt: skip
    schema.apply_critique(hyps, critique)
    assert (hyps[0].verdict, hyps[0].confidence) == ("contradicted", 0.1)
    assert (hyps[1].verdict, hyps[1].confidence) == ("supported", 0.9)


def test_parse_keeps_caveat_and_wraps_string_evidence():
    hyps = schema.parse_hypotheses(
        json.dumps([{"claim": "c", "mechanism": "m", "evidence": "kcat 0.6 vs 0.7",
                     "caveat": "mixes baselines"}])
    )  # fmt: skip
    assert hyps[0].evidence == ["kcat 0.6 vs 0.7"]
    assert hyps[0].caveat == "mixes baselines"


_GEN = json.dumps(
    [{"claim": "X wins because bigger", "mechanism": "capacity",
      "evidence": ["kcat 0.6 vs 0.7"], "differentiating_attribute": "parameters",
      "confidence": 0.8, "falsification_test": "km"},
     {"claim": "Y wins on the similarity split", "mechanism": "structure",
      "evidence": ["similarity split"], "differentiating_attribute": "objective",
      "confidence": 0.6, "falsification_test": "random"}]
)  # fmt: skip
_CRIT = json.dumps(
    [{"verdict": "contradicted", "verdict_reason": "fails on half", "confidence": 0.2},
     {"verdict": "supported", "verdict_reason": "holds", "confidence": 0.7}]
)  # fmt: skip


@pytest.fixture
def stub_corpus(monkeypatch):
    monkeypatch.setattr(
        hypothesize.corpus, "load_corpus", lambda *a: "===== CARD: x =====\nbody"
    )


def test_orchestrator_critiques_and_ranks(monkeypatch, stub_corpus):
    calls = {"n": 0}

    def fake_generate(prompt, **kwargs):
        calls["n"] += 1
        return _GEN if calls["n"] == 1 else _CRIT

    monkeypatch.setattr(hypothesize.gemini, "generate", fake_generate)
    result = hypothesize.explain_pairwise("esm2_650m", "prost5", n=5)
    assert calls["n"] == 2
    assert [h.verdict for h in result.hypotheses] == ["supported", "contradicted"]


def test_orchestrator_survives_bad_critique_json(monkeypatch, stub_corpus):
    def fake_generate(prompt, **kwargs):
        return _GEN if "PROPOSED HYPOTHESES" not in prompt else "not json{{"

    monkeypatch.setattr(hypothesize.gemini, "generate", fake_generate)
    assert len(hypothesize.explain_pairwise("esm2_650m", "prost5").hypotheses) == 2
