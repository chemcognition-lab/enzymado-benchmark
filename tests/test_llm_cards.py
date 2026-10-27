"""Tests for the assistant's knowledge-base cards (``enzymado.llm.render``).

Torch-free and network-free: the cards are built from the published results that
are tracked in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from enzymado.llm import render, sources

LLM_DIR = Path(render.__file__).resolve().parent
EM_DASH = chr(0x2014)  # spelled as a code point so this file never contains one


@pytest.fixture(scope="module")
def cards() -> dict[str, str]:
    return dict(render.build_all())


def test_card_ids_cover_every_representation_split_baseline_and_task():
    ids = render.card_ids()
    assert sum(i.startswith("model_") for i in ids) == 15  # 14 PLMs and no-PLM
    assert sum(i.startswith("split_") for i in ids) == 2
    assert sum(i.startswith("baseline_") for i in ids) == 5
    assert sum(i.startswith("task_") for i in ids) == 18


def test_every_card_has_authored_content_that_parses():
    missing = [
        i for i in render.card_ids() if not (render.CONTENT_DIR / f"{i}.json").exists()
    ]
    assert missing == []
    for card_id in render.card_ids():
        json.loads((render.CONTENT_DIR / f"{card_id}.json").read_text())


def test_render_all_writes_every_card(tmp_path):
    written = render.render_all(tmp_path)
    assert len(written) == len(render.card_ids()) + len(sources.LEADERBOARDS)
    for name in ("model_esm2_650m.md", "task_kcat.md", "baseline_xgboost.md",
                 "split_similarity.md", "leaderboard_properties.md"):  # fmt: skip
        assert (tmp_path / name).exists()


def test_model_card_merges_ratings_facts_and_prose(cards):
    card = cards["model_prost5.md"]
    best = sources.model_standing("prost5")["properties"]["best"]
    assert "embedding_dim: 1024" in card
    assert "1.21B (encoder only)" in card  # from plm_attributes.csv
    assert "Rostlab/ProstT5" in card  # authored
    assert f"{best['rating']} [{best['ci_lo']}, {best['ci_hi']}]" in card


def test_leaderboard_card_lists_every_competitor_best_first(cards):
    card = cards["leaderboard_properties.md"]
    ratings = sources.load_ratings("properties")
    assert f"| 1 | `{ratings.embedding_model[0]}`" in card
    assert card.count("\n| ") - 1 == len(ratings)  # header row excluded


def test_baseline_settings_come_from_the_runs(cards):
    card = cards["baseline_xgboost.md"]
    assert '"n_estimators": 100' in card
    assert '"max_depth": 6' in card


def test_task_cards_use_the_papers_primary_metrics(cards):
    assert "task_type: regression" in cards["task_kcat.md"]
    assert "`kendall_tau`" in cards["task_kcat.md"]
    assert "`auprc`" in cards["task_ec_number.md"]
    assert "CatPred" in cards["task_kcat.md"]


def test_real_case_task_card_has_a_random_split_ranking(cards):
    card = cards["task_erickson_petase.md"]
    assert "task_scope: real_case" in card
    assert "### random split" in card
    assert "### similarity split" not in card


def test_authored_cards_have_frontmatter_and_references(cards):
    for name, text in cards.items():
        assert text.startswith("---\n"), name
        if not name.startswith("leaderboard_"):
            assert "## References" in text, name


def test_cards_and_sources_never_claim_truncation(cards):
    for name, text in cards.items():
        assert "truncat" not in text.replace("never truncated", ""), name


def test_no_em_dash_in_cards_content_or_templates(cards):
    files = [*LLM_DIR.glob("content/*.json"), *LLM_DIR.glob("templates/*.j2")]
    for path in files:
        assert EM_DASH not in path.read_text(), path.name
    for name, text in cards.items():
        assert EM_DASH not in text, name


def test_render_is_deterministic(cards):
    assert dict(render.build_all()) == cards


def test_scaffold_content_never_overwrites():
    assert render.scaffold_content() == []


def test_split_gap_averages_the_plms_only():
    records = sources.load_records()
    rows = {r["task"]: r for r in sources.split_gap_table(records)}
    assert set(rows) == {
        t
        for t, c in sources.load_task_registry().items()
        if c.task_scope == "properties"
    }
    ec = [
        r["primary_value"]
        for r in records
        if r["task"] == "ec_number" and r["split"] == "random" and r["emb"] != "noplm"
    ]
    assert rows["ec_number"]["random"] == round(sum(ec) / len(ec), 4)


def test_sources_and_cards_do_not_import_torch():
    code = "import sys; import enzymado.llm.render; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
