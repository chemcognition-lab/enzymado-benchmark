"""Tests for the RAG knowledge-base card generator (`llm` package).

These are intentionally torch-free (they exercise only the base-env code path),
mirroring how the generator sources everything from data_registry.py,
metrics_registry, and results_card.json.
"""
from __future__ import annotations

import json

from llm import render, sources


def _cards() -> dict[str, str]:
    return dict(render.build_all())


def test_card_id_breakdown():
    ids = render.card_ids()
    assert len(ids) == 39
    assert sum(i.startswith("model_") for i in ids) == 14
    assert sum(i.startswith("split_") for i in ids) == 2
    assert sum(i.startswith("baseline_") for i in ids) == 5
    assert sum(i.startswith("task_") for i in ids) == 18


def test_every_card_has_authored_content_that_parses():
    missing = [cid for cid in render.card_ids()
               if not (render.CONTENT_DIR / f"{cid}.json").exists()]
    assert missing == [], f"missing authored content: {missing}"
    for cid in render.card_ids():
        json.loads((render.CONTENT_DIR / f"{cid}.json").read_text())


def test_render_all_writes_39_named_cards(tmp_path):
    written = render.render_all(tmp_path)
    assert len(written) == 39
    for name in ("model_esm2_650m.md", "task_kcat.md",
                 "baseline_xgboost.md", "split_similarity.md"):
        assert (tmp_path / name).exists()


def test_model_card_merges_auto_and_authored():
    esm = _cards()["model_esm2_650m.md"]
    assert "embedding_dim: 1280" in esm                 # auto (manifest)
    assert "facebook/esm2_t33_650M_UR50D" in esm        # authored HF repo
    assert "Lin, Z. et al." in esm                      # authored citation
    assert "Benchmark standing" in esm                  # auto results block


def test_baseline_xgb_hyperparams_are_live_from_results_card():
    xgb = _cards()["baseline_xgboost.md"]
    assert '"n_estimators": 100' in xgb
    assert '"max_depth": 6' in xgb
    assert '"learning_rate": 0.1' in xgb
    assert "xgboost.readthedocs.io" in xgb              # authored doc ref


def test_task_kcat_definition_and_source():
    kcat = _cards()["task_kcat.md"]
    assert "task_type: regression" in kcat
    assert "`mae`" in kcat                              # primary metric (auto)
    assert "CatPred" in kcat                            # authored source


def test_real_case_task_has_no_stats_block_but_has_results():
    erickson = _cards()["task_erickson_petase.md"]
    assert "task_scope: real_case" in erickson
    assert "No `dataset_card.json` stats available" in erickson
    assert "Leaderboard" in erickson


def test_every_card_has_frontmatter_and_references():
    for name, text in _cards().items():
        assert text.startswith("---\n"), f"{name} missing frontmatter"
        assert "## References" in text, f"{name} missing References"


def test_render_is_deterministic():
    assert _cards() == _cards()


def test_scaffold_content_is_nondestructive():
    # Every card already has authored content, so scaffolding creates nothing
    # and never overwrites existing prose.
    assert render.scaffold_content() == []


def test_sources_are_torch_free_and_complete():
    assert len(sources.load_models()) == 14
    assert len(sources.load_task_registry()) == 18
    primary_metric_for, _ = sources._metric_conventions()
    assert primary_metric_for("classification") == "f1"
    assert primary_metric_for("regression") == "mae"
