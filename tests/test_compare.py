"""Tests for the hypothesis engine's deterministic layer and orchestration wiring.

Torch-free and network-free: the ``compare`` math runs on the live ``results_card.json``,
and the orchestrator is exercised with a monkeypatched Gemini call (no API key needed).
"""
from __future__ import annotations

import json

import pytest

from llm import compare, hypothesize, prompts, schema
from llm import sources as S


@pytest.fixture(scope="module")
def records():
    return S.load_records()


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

def test_normalize_key_handles_case_and_card_prefix():
    assert compare.normalize_key("esm2_650m") == "esm2_650M"
    assert compare.normalize_key("esm2_650M") == "esm2_650M"
    assert compare.normalize_key("model_ankh3_xl") == "ankh3_xl"
    with pytest.raises(KeyError):
        compare.normalize_key("not_a_model")


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def test_head_to_head_winner_is_internally_consistent(records):
    h = compare.head_to_head(records, "esm2_650m", "prost5")
    assert h["x"] == "esm2_650M" and h["y"] == "prost5"
    assert h["rows"], "expected shared task rows"

    for r in h["rows"]:
        # winner must agree with the sign of the sign-aware delta
        if r["delta"] > 0:
            assert r["winner"] == h["x"]
        elif r["delta"] < 0:
            assert r["winner"] == h["y"]
        # for a lower-is-better metric (mae), the winner holds the smaller value
        if r["metric"] == "mae" and r["winner"] != "tie":
            win_val = r["x_value"] if r["winner"] == h["x"] else r["y_value"]
            lose_val = r["y_value"] if r["winner"] == h["x"] else r["x_value"]
            assert win_val <= lose_val

    tally = h["tally"]
    assert tally[h["x"]] + tally[h["y"]] + tally["tie"] == len(h["rows"])


def test_head_to_head_standing_covers_four_tracks(records):
    h = compare.head_to_head(records, "esm2_650m", "prost5")
    tracks = {row["track"] for row in h["standing"]}
    assert any("random" in t for t in tracks)
    assert any("similarity" in t for t in tracks)


# ---------------------------------------------------------------------------
# Attribute diff
# ---------------------------------------------------------------------------

def test_attribute_diff_flags_needs_verification_and_dim(records):
    diff = compare.attribute_diff("ankh3_xl", "esm2_650m")
    assert diff["flags"]["ankh3_xl"]["needs_verification"] is True
    assert diff["flags"]["esm2_650M"]["needs_verification"] is False
    fields = {d["field"] for d in diff["differing"]}
    assert "embedding_dim" in fields
    assert "pretraining_objective" in fields


def test_attribute_diff_render_warns_on_soft_attribute(records):
    md = compare.render_attribute_diff_md(compare.attribute_diff("ankh3_xl", "esm2_650m"))
    assert "needs_verification" in md


def test_attribute_diff_shares_embedding_dim_when_equal():
    # prost5 and prot5_xl_uniref50 both embed at dim 1024 — dim is NOT a differentiator.
    diff = compare.attribute_diff("prost5", "prot5_xl_uniref50")
    shared = {d["field"] for d in diff["shared"]}
    differing = {d["field"] for d in diff["differing"]}
    assert "embedding_dim" in shared
    assert "embedding_dim" not in differing


# ---------------------------------------------------------------------------
# Combination coverage
# ---------------------------------------------------------------------------

def test_combo_coverage_tally_and_membership(records):
    combo = compare.combo_coverage(records, ["ankh_large", "prost5"], ["esm2_650m", "esmc_600m"])
    assert combo["set_a"] == ["ankh_large", "prost5"]
    t = combo["tally"]
    assert t["A"] + t["B"] + t["tie"] == len(combo["rows"])
    for r in combo["rows"]:
        assert r["a_member"] in combo["set_a"]
        assert r["b_member"] in combo["set_b"]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def test_generation_prompt_embeds_evidence_and_schema():
    prompt = prompts.build_generation_prompt(
        question="why does X beat Y",
        evidence_md="| Task | Split |\n|---|---|\nEVIDENCE_MARKER",
        corpus="===== CARD: task_kcat =====\nbody",
        n=3,
    )
    assert "EVIDENCE_MARKER" in prompt
    assert "task_kcat" in prompt
    assert "falsification_test" in prompt  # from the schema hint
    assert "caveat" in prompt               # #6: config/dropped-task disclaimer field


# ---------------------------------------------------------------------------
# #6 Disclaimer: dropped tasks + methodology block
# ---------------------------------------------------------------------------

def test_full_coverage_reports_no_dropped(records):
    # All 14 models cover every split×task in this benchmark, so nothing is excluded.
    h = compare.head_to_head(records, "esm2_650m", "prost5")
    assert h["dropped"] == []


def test_dropped_tasks_detects_a_coverage_gap():
    # Synthetic best-dicts: model B is missing one split×task that A has.
    bx = {("random", "kcat"): {}, ("random", "km"): {}}
    by = {("random", "kcat"): {}}
    dropped = compare._dropped_tasks({"esm2_650M": bx, "prost5": by})
    assert len(dropped) == 1
    assert dropped[0]["task"] == "km" and dropped[0]["split"] == "random"
    assert dropped[0]["present_only_for"] == "esm2_650M"


def test_methodology_block_rendered_in_both_views(records):
    h = compare.render_head_to_head_md(compare.head_to_head(records, "esm2_650m", "prost5"))
    c = compare.render_combo_md(
        compare.combo_coverage(records, ["ankh_large", "prost5"], ["esm2_650m", "esmc_600m"]))
    for md in (h, c):
        assert "Methodology & caveats" in md
        assert "Best-config per task" in md


def test_apply_critique_matches_by_index_when_reordered():
    hyps = [schema.Hypothesis(claim="first", mechanism="m"),
            schema.Hypothesis(claim="second", mechanism="m")]
    # Critic returns verdicts out of order, tagged with the hypothesis index.
    crit = json.dumps([
        {"index": 1, "verdict": "supported", "confidence": 0.9},
        {"index": 0, "verdict": "contradicted", "confidence": 0.1},
    ])
    schema.apply_critique(hyps, crit)
    assert hyps[0].verdict == "contradicted" and hyps[0].confidence == 0.1
    assert hyps[1].verdict == "supported" and hyps[1].confidence == 0.9


def test_parse_reads_caveat_field():
    import json
    text = json.dumps([{"claim": "c", "mechanism": "m", "evidence": [],
                        "caveat": "mixes attention and no-attention heads"}])
    hyps = schema.parse_hypotheses(text)
    assert hyps[0].caveat == "mixes attention and no-attention heads"


def test_parse_coerces_string_evidence_to_single_bullet():
    # A model that returns `evidence` as a lone string must not be split into characters.
    text = json.dumps([{"claim": "c", "mechanism": "m", "evidence": "kcat 0.6 vs 0.7"}])
    hyps = schema.parse_hypotheses(text)
    assert hyps[0].evidence == ["kcat 0.6 vs 0.7"]


# ---------------------------------------------------------------------------
# Orchestration (monkeypatched Gemini — no network)
# ---------------------------------------------------------------------------

_GEN_JSON = json.dumps([
    {"claim": "X wins because bigger", "mechanism": "capacity", "evidence": ["kcat 0.6 vs 0.7"],
     "differentiating_attribute": "parameters", "confidence": 0.8, "falsification_test": "km"},
    {"claim": "Y wins on OOD", "mechanism": "structure signal", "evidence": ["similarity split"],
     "differentiating_attribute": "pretraining_objective", "confidence": 0.6, "falsification_test": "random"},
])
_CRIT_JSON = json.dumps([
    {"verdict": "contradicted", "verdict_reason": "fails on half the tasks", "confidence": 0.2},
    {"verdict": "supported", "verdict_reason": "holds across similarity split", "confidence": 0.7},
])


def test_orchestrator_parses_critiques_and_ranks(monkeypatch):
    calls = {"n": 0}

    def fake_generate(prompt, **kwargs):
        calls["n"] += 1
        return _GEN_JSON if calls["n"] == 1 else _CRIT_JSON

    monkeypatch.setattr(hypothesize.gemini, "generate", fake_generate)
    result = hypothesize.explain_pairwise("esm2_650m", "prost5", n=5)

    assert calls["n"] == 2  # generation + critique
    assert len(result.hypotheses) == 2
    # "supported" must rank above "contradicted" after ranking
    assert result.hypotheses[0].verdict == "supported"
    assert result.hypotheses[1].verdict == "contradicted"
    assert result.hypotheses[0].verdict_reason


def test_orchestrator_survives_bad_critique_json(monkeypatch):
    def fake_generate(prompt, **kwargs):
        return _GEN_JSON if "PROPOSED HYPOTHESES" not in prompt else "not json{{"

    monkeypatch.setattr(hypothesize.gemini, "generate", fake_generate)
    result = hypothesize.explain_pairwise("esm2_650m", "prost5", n=5)
    assert len(result.hypotheses) == 2  # kept despite critique parse failure


def test_hypotheses_rank_by_self_reported_confidence_without_critique(monkeypatch):
    # With the critique off, verdicts tie and the model's own confidence alone orders them.
    gen = json.dumps([
        {"claim": "weaker", "mechanism": "m", "evidence": ["Y wins 987654 tasks"],
         "differentiating_attribute": "parameters", "confidence": 0.4},
        {"claim": "stronger", "mechanism": "m", "evidence": [],
         "differentiating_attribute": "family", "confidence": 0.6},
    ])
    monkeypatch.setattr(hypothesize.gemini, "generate", lambda *a, **k: gen)
    result = hypothesize.explain_pairwise("esm2_650m", "prost5", n=5, critique=False)

    by_claim = {h.claim: h for h in result.hypotheses}
    # Confidence is passed through untouched — nothing rewrites it post-generation.
    assert by_claim["weaker"].confidence == pytest.approx(0.4)
    assert by_claim["stronger"].confidence == pytest.approx(0.6)
    assert result.hypotheses[0].claim == "stronger"