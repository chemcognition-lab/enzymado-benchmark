"""Tests for the assistant's chatbot layer (``enzymado.llm.chat``).

The tools run on the published results and on cards built into a temporary
directory; ``ChatSession`` is exercised with a fake Gemini client and a fake SDK
module, so no network, API key or ``google-genai`` install is needed.
"""

from __future__ import annotations

import sys
import types

import pytest

from enzymado.llm import chat, corpus, render


@pytest.fixture(scope="module", autouse=True)
def cards_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("cards")
    render.render_all(out)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(corpus, "DEFAULT_MARKDOWN_DIR", out)
        corpus.load_corpus.cache_clear()
        yield out
    corpus.load_corpus.cache_clear()


def test_compare_two_models_returns_the_exact_table():
    out = chat.compare_two_models("esm2_650m", "prost5")
    assert "Head-to-head" in out
    assert "Task wins" in out


def test_compare_model_sets_needs_both_sides():
    assert "Combination coverage" in chat.compare_model_sets("ankh_large", "esm2_650m")
    assert "at least one model" in chat.compare_model_sets("", "esm2_650m")


def test_unknown_names_recover_instead_of_raising():
    assert "esm2_650M" in chat.compare_two_models("not_a_model", "prost5")
    assert "Known models" in chat.model_info("not_a_model")
    assert "Unknown task" in chat.task_info("does_not_exist")


def test_card_readers_return_cards():
    assert "model card" in chat.model_info("esm2_650M")
    assert chat.task_info("task_ph_opt").startswith("---")


def test_task_leaderboard_marks_ties_with_the_leader():
    out = chat.task_leaderboard("km")
    assert "random split" in out and "similarity split" in out
    assert "(within noise)" in out


def test_leaderboard_tool_shows_the_published_ranking():
    out = chat.leaderboard("properties", top=3)
    assert "| 1 | `prost5` | mlp |" in out
    assert "Unknown leaderboard" in chat.leaderboard("nope")


def test_list_and_split_tools():
    assert "noplm" in chat.list_models()
    assert "erickson_petase" in chat.list_tasks()
    assert "Drop" in chat.split_difficulty()


def test_split_csv_trims_and_drops_empties():
    assert chat._split_csv("a, b ,, c ") == ["a", "b", "c"]


def _history(tool: str, args: dict, result: str) -> list:
    call = types.SimpleNamespace(name=tool, args=args)
    reply = types.SimpleNamespace(name=tool, response={"result": result})
    return [
        types.SimpleNamespace(
            parts=[types.SimpleNamespace(function_call=call, function_response=None)]
        ),
        types.SimpleNamespace(
            parts=[types.SimpleNamespace(function_call=None, function_response=reply)]
        ),
    ]


@pytest.fixture
def fake_gemini(monkeypatch):
    """A fake SDK module and client; returns (record, set_response)."""
    sdk_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kw: types.SimpleNamespace(**kw),
        AutomaticFunctionCallingConfig=lambda **kw: types.SimpleNamespace(**kw),
    )
    genai = types.ModuleType("google.genai")
    genai.types = sdk_types
    if "google" not in sys.modules:
        monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    record: dict = {"clients": 0}

    class _Chat:
        def send_message(self, message, **kwargs):
            record.setdefault("sent", []).append(message)
            return record["response"]

    class _Chats:
        def create(self, *, model, config):
            record["config"] = config
            return _Chat()

    def fake_client():
        record["clients"] += 1
        return types.SimpleNamespace(chats=_Chats())

    monkeypatch.setattr(chat.gemini, "_client", fake_client)
    return record


def test_session_is_lazy(fake_gemini):
    chat.ChatSession()
    assert fake_gemini["clients"] == 0


def test_send_registers_tools_stuffs_corpus_and_parses_calls(fake_gemini):
    fake_gemini["response"] = types.SimpleNamespace(
        text="prost5 leads.",
        automatic_function_calling_history=_history(
            "compare_two_models", {"model_x": "prost5", "model_y": "esm2_650M"}, "0.42"
        ),
        prompt_feedback=None,
    )
    session = chat.ChatSession()
    reply = session.send("Why does prost5 beat esm2_650M?")
    config = fake_gemini["config"]
    assert config.system_instruction.startswith(chat.CHAT_SYSTEM)
    assert session.cards_in_context == corpus.card_count()
    for card_id in ("model_esm2_650m", "task_kcat", "leaderboard_properties"):
        assert f"{corpus.CARD_MARKER} {card_id} =====" in config.system_instruction
    names = {t.__name__ for t in config.tools}
    assert names.isdisjoint({"model_info", "task_info"})
    assert {"compare_two_models", "leaderboard", "split_difficulty"} <= names
    assert reply.text == "prost5 leads."
    assert [c.name for c in reply.tool_calls] == ["compare_two_models"]
    assert reply.tool_calls[0].result_preview == "0.42"


def test_missing_corpus_falls_back_to_card_tools(tmp_path):
    system, tools, n_cards = chat.build_system_instruction(tmp_path)
    assert n_cards == 0
    assert "KNOWLEDGE BASE UNAVAILABLE" in system
    assert {"model_info", "task_info"} <= {t.__name__ for t in tools}


def test_send_raises_on_a_blocked_response(fake_gemini):
    fake_gemini["response"] = types.SimpleNamespace(
        text=None,
        automatic_function_calling_history=[],
        prompt_feedback=types.SimpleNamespace(block_reason="SAFETY"),
    )
    with pytest.raises(chat.gemini.GeminiError):
        chat.ChatSession().send("hello")
