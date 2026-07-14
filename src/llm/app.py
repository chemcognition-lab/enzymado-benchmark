"""Streamlit chatbot for the Enzymado benchmark.

Run with:  ``streamlit run src/llm/app.py``

A pure-text assistant: you type questions and it answers, grounding every number in the
deterministic tools from :mod:`llm.chat` (which wrap :mod:`llm.compare` / :mod:`llm.sources`)
so figures are *computed*, never invented. It can explain how a model works, hypothesize why
one model or set of models beats another, and advise on experimental-design choices.

Streamlit executes this file as a script (not a package submodule), so we put ``src`` on the
path and import the package absolutely.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `llm` importable whether run from the repo root or elsewhere.
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from llm import chat, gemini

st.set_page_config(page_title="Enzymado — benchmark chat", page_icon="🧬", layout="centered")

st.title("🧬 Enzymado benchmark — chat")
st.caption(
    "Ask about the benchmark in plain language. Every number is computed on demand from "
    "`results_card.json` via grounded tools — not made up. Try: *“How does prost5 work?”*, "
    "*“Why does esm2_650M beat prost5?”*, *“Which model and split should I use for kcat?”* "
    "Type **/reset** to start over."
)

# The chatbot cannot do anything without a key; fail early and clearly.
if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "`GEMINI_API_KEY` is not set in the environment. Set it and reload — e.g. "
        "`export GEMINI_API_KEY=…` before `streamlit run src/llm/app.py`."
    )
    st.stop()

# `messages` is only for re-rendering; the real conversational memory lives inside the
# ChatSession's Gemini chat (its curated history, including tool round-trips).
st.session_state.setdefault("messages", [])
if "chat" not in st.session_state:
    st.session_state.chat = chat.ChatSession()


def _render_tools(tool_calls: list[chat.ToolCall]) -> None:
    if not tool_calls:
        return
    names = ", ".join(f"`{c.name}`" for c in tool_calls)
    with st.expander(f"Evidence · tools used: {names}"):
        for c in tool_calls:
            arg_str = ", ".join(f"{k}={v!r}" for k, v in c.args.items())
            st.markdown(f"**`{c.name}`**({arg_str})")
            if c.result_preview:
                st.code(c.result_preview, language="markdown")


# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_tools(msg.get("tool_calls", []))
            if msg.get("ungrounded"):
                nums = ", ".join(str(n) for n in msg["ungrounded"])
                st.warning(f"⚠ Reply mentions numbers not found in any tool output: {nums}")

if prompt := st.chat_input("Ask about the benchmark…"):
    # A typed command keeps the interface pure-text (no reset button).
    if prompt.strip().lower() in ("/reset", "/clear"):
        st.session_state.messages = []
        st.session_state.chat = chat.ChatSession()
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = st.session_state.chat.send(prompt)
            except gemini.GeminiError as e:
                st.error(str(e))
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠ {e}"})
            else:
                st.markdown(reply.text)
                _render_tools(reply.tool_calls)
                if reply.ungrounded_numbers:
                    nums = ", ".join(str(n) for n in reply.ungrounded_numbers)
                    st.warning(f"⚠ Reply mentions numbers not found in any tool output: {nums}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": reply.tool_calls,
                    "ungrounded": reply.ungrounded_numbers,
                })
