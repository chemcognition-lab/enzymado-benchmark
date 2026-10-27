"""Streamlit chatbot for the Enzymado benchmark.

A text assistant: the whole card corpus sits in each session's system instruction,
so descriptive answers come from the cards, while cross-model numbers (head-to-
heads, leaderboards, split drops) are computed on demand by the tools in
:mod:`chat`. Needs ``GEMINI_API_KEY`` and the ``llm`` extra, and the cards built
with ``python -m enzymado.llm``.

Example:
    streamlit run scripts/llm/chat_app.py
"""

from __future__ import annotations

import os

import streamlit as st

from enzymado.llm import chat, corpus, gemini

st.set_page_config(page_title="Enzymado chat", page_icon="🧬", layout="centered")
st.title("Enzymado benchmark chat")
st.caption(
    "Ask about the benchmark in plain language. The assistant reads every benchmark "
    "card and computes comparisons from the published results. Try: *How does "
    "prost5 work?*, *Why does prost5 beat esm2_650M?*, *Which model and split should "
    "I use for kcat?* Type **/reset** to start over."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "`GEMINI_API_KEY` is not set. Export it and reload, e.g. "
        "`export GEMINI_API_KEY=...` before `streamlit run scripts/llm/chat_app.py`."
    )
    st.stop()

try:
    st.caption(f"{corpus.card_count()} benchmark cards loaded into context.")
except (FileNotFoundError, OSError) as exc:
    st.warning(
        "No benchmark cards found, so the assistant runs without its knowledge base "
        "and fetches cards by tool. Run `python -m enzymado.llm` to build them. "
        f"({exc})"
    )

# `messages` only re-renders the page; the conversation's memory is the chat itself.
st.session_state.setdefault("messages", [])
if "chat" not in st.session_state:
    st.session_state.chat = chat.ChatSession()


def _render_tools(tool_calls: list[chat.ToolCall]) -> None:
    if not tool_calls:
        return
    names = ", ".join(f"`{c.name}`" for c in tool_calls)
    with st.expander(f"Evidence: tools used {names}"):
        for call in tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
            st.markdown(f"**`{call.name}`**({args})")
            if call.result_preview:
                st.code(call.result_preview, language="markdown")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_tools(message.get("tool_calls", []))

if prompt := st.chat_input("Ask about the benchmark..."):
    if prompt.strip().lower() in ("/reset", "/clear"):
        st.session_state.messages = []
        st.session_state.chat = chat.ChatSession()
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = st.session_state.chat.send(prompt)
            except gemini.GeminiError as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {exc}"}
                )
            else:
                st.markdown(reply.text)
                _render_tools(reply.tool_calls)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply.text,
                        "tool_calls": reply.tool_calls,
                    }
                )
