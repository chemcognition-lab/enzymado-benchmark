"""Tests for the grounded chatbot layer (:mod:`llm.chat`).

Torch-free and network-free, mirroring ``tests/test_compare.py``: the tool wrappers run on
the live ``results_card.json`` / markdown cards, and the ``ChatSession`` wiring is exercised
with a fake Gemini client (no API key, no SDK call).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm import chat


# ---------------------------------------------------------------------------
# Pure tool wrappers (no key, no network)
# ---------------------------------------------------------------------------

def test_compare_two_models_returns_grounded_table():
    out = chat.compare_two_models("esm2_650m", "prost5")
    assert "Head-to-head" in out
    assert "Task wins" in out


def test_compare_model_sets_returns_combo_coverage():
    out = chat.compare_model_sets("ankh_large, prost5", "esm2_650m, esmc_600m")
    assert "Combination coverage" in out


def test_compare_model_sets_needs_both_sides():
    assert "at least one model" in chat.compare_model_sets("", "esm2_650m").lower()


def test_bad_model_name_recovers_instead_of_raising():
    out = chat.compare_two_models("not_a_model", "prost5")
    assert "esm2_650M" in out  # the friendly recovery lists the known keys


def test_model_info_reads_card_and_handles_unknown():
    card = chat.model_info("esm2_650M")
    assert "foundation model card" in card.lower()
    assert "Known models" in chat.model_info("not_a_model")  # does not raise


def test_task_info_resolves_results_and_registry_names():
    # `phopt` (results name) and `ph_opt` (registry name) both resolve to task_ph_opt.md.
    assert chat.task_info("phopt").strip()
    assert chat.task_info("ph_opt").strip()
    assert "Unknown task" in chat.task_info("does_not_exist")


def test_task_leaderboard_maps_phopt():
    out = chat.task_leaderboard("ph_opt")
    assert "Leaderboard" in out
    assert chat.task_leaderboard("phopt").strip()  # results-name spelling also works


def test_list_and_split_tools_are_nonempty():
    assert "esm2_650M" in chat.list_models()
    assert "kcat" in chat.list_tasks()
    assert "Harder by" in chat.split_difficulty()


def test_split_csv_trims_and_drops_empties():
    assert chat._split_csv("a, b ,, c ") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# ChatSession wiring (fake Gemini client — no network, no key)
# ---------------------------------------------------------------------------

def _fake_afc_history(*, tool_name: str, args: dict, result: str):
    """Build a minimal `automatic_function_calling_history` (a list of Content)."""
    call = SimpleNamespace(
        parts=[SimpleNamespace(
            function_call=SimpleNamespace(name=tool_name, args=args),
            function_response=None)])
    response = SimpleNamespace(
        parts=[SimpleNamespace(
            function_call=None,
            function_response=SimpleNamespace(name=tool_name, response={"result": result}))])
    return [call, response]


class _FakeChat:
    def __init__(self, resp):
        self._resp = resp
        self.sent: list[str] = []

    def send_message(self, message, **kwargs):
        self.sent.append(message)
        return self._resp


def _install_fake_client(monkeypatch, resp):
    """Patch ``chat.gemini._client`` with a recording fake; return (record, counter)."""
    record: dict = {}
    counter = {"n": 0}
    fake_chat = _FakeChat(resp)

    class _FakeChats:
        def create(self, *, model, config):
            record["model"] = model
            record["config"] = config
            return fake_chat

    def fake_client():
        counter["n"] += 1
        return SimpleNamespace(chats=_FakeChats())

    monkeypatch.setattr(chat.gemini, "_client", fake_client)
    return record, counter


def test_session_construction_is_lazy(monkeypatch):
    _, counter = _install_fake_client(monkeypatch, SimpleNamespace(text="hi"))
    chat.ChatSession()
    assert counter["n"] == 0  # no client created until the first send()


def test_send_registers_tools_and_parses_reply(monkeypatch):
    resp = SimpleNamespace(
        text="esm2_650M leads kcat on the similarity split (0.676).",
        automatic_function_calling_history=_fake_afc_history(
            tool_name="compare_two_models",
            args={"model_x": "esm2_650M", "model_y": "prost5"},
            result="kcat similarity: X 0.676 vs Y 0.5927"),
        prompt_feedback=None,
    )
    record, _ = _install_fake_client(monkeypatch, resp)

    session = chat.ChatSession()
    reply = session.send("Why does esm2_650M beat prost5?")

    # The tools + system prompt are registered on the chat config.
    cfg = record["config"]
    assert cfg.system_instruction.startswith(chat.CHAT_SYSTEM)
    assert len(cfg.tools) == len(chat.CORPUS_TOOLS)

    # The reply text is passed through and the tool call is parsed for display.
    assert reply.text == resp.text
    assert [c.name for c in reply.tool_calls] == ["compare_two_models"]
    assert reply.tool_calls[0].args == {"model_x": "esm2_650M", "model_y": "prost5"}
    assert "0.676" in reply.tool_calls[0].result_preview


def test_send_stuffs_the_whole_corpus_and_retires_card_tools(monkeypatch):
    resp = SimpleNamespace(text="hi", automatic_function_calling_history=[],
                           prompt_feedback=None)
    record, _ = _install_fake_client(monkeypatch, resp)

    session = chat.ChatSession()
    session.send("hello")
    cfg = record["config"]

    # Every card is inlined in the system instruction ...
    assert session.cards_in_context == chat.corpus.card_count()
    for card_id in ("model_esm2_650m", "task_kcat", "baseline_mlp", "split_random"):
        assert f"{chat.corpus.CARD_MARKER} {card_id} =====" in cfg.system_instruction

    # ... so the per-card readers are not registered, but the compute tools are.
    names = {t.__name__ for t in cfg.tools}
    assert names.isdisjoint({"model_info", "task_info"})
    assert {"compare_two_models", "task_leaderboard", "split_difficulty"} <= names


def test_missing_corpus_falls_back_to_card_tools(monkeypatch, tmp_path):
    """An empty markdown dir degrades to the tool-based knowledge base instead of failing."""
    system, tools, n_cards = chat.build_system_instruction(tmp_path)
    assert n_cards == 0
    assert "KNOWLEDGE BASE UNAVAILABLE" in system
    assert {"model_info", "task_info"} <= {t.__name__ for t in tools}


def test_send_raises_on_empty_response(monkeypatch):
    resp = SimpleNamespace(
        text=None,
        automatic_function_calling_history=[],
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    _install_fake_client(monkeypatch, resp)
    with pytest.raises(chat.gemini.GeminiError):
        chat.ChatSession().send("hello")
