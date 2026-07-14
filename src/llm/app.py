"""Streamlit UI for the benchmark hypothesis engine.

Run with:  ``streamlit run src/llm/app.py``

Streamlit executes this file as a script (not as a package submodule), so we put
``src`` on the path and import the package absolutely. The deterministic evidence table
renders immediately; the Gemini hypotheses stream in behind a spinner.
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

from llm import compare, gemini, hypothesize
from llm import sources as S

st.set_page_config(page_title="Enzymado — benchmark hypotheses", page_icon="🧬", layout="wide")

MODELS = sorted(S.load_models())


def _default_index(preferred: str) -> int:
    """Index of a preferred default in MODELS, or 0 if the manifest lacks it."""
    return MODELS.index(preferred) if preferred in MODELS else 0


_VERDICT_COLOR = {
    "supported": "green", "partly-supported": "blue", "uncertain": "gray",
    "cherry-picked": "orange", "contradicted": "red",
}


def _render_hypotheses(result: hypothesize.Result) -> None:
    if not result.hypotheses:
        st.warning("No hypotheses returned.")
        return
    for i, h in enumerate(result.hypotheses, 1):
        header = f"**{i}. {h.claim}**"
        if h.verdict:
            color = _VERDICT_COLOR.get(h.verdict, "gray")
            header += f" &nbsp;:{color}[[{h.verdict}]]"
        header += f" &nbsp;·&nbsp; confidence {h.confidence:.2f}"
        st.markdown(header)
        st.markdown(f"*Mechanism.* {h.mechanism}")
        st.markdown(f"*Differentiating attribute.* `{h.differentiating_attribute}`")
        if h.evidence:
            st.markdown("*Evidence.*")
            for e in h.evidence:
                st.markdown(f"- {e}")
        if h.caveat:
            st.markdown(f"*Caveat.* {h.caveat}")
        if h.falsification_test:
            st.markdown(f"*Falsification test.* {h.falsification_test}")
        if h.unverified_numbers:
            nums = ", ".join(str(n) for n in h.unverified_numbers)
            st.warning(f"⚠ Cites numbers not found in the evidence table: {nums}")
        if h.verdict_reason:
            st.info(f"**Critic:** {h.verdict_reason}")
        st.divider()


st.title("🧬 Enzymado benchmark — hypothesis engine")
st.caption(
    "Deterministic head-to-head numbers + a Gemini generate-then-critique loop. "
    "The table below is computed exactly from `results_card.json`; the hypotheses are grounded in it."
)

# Loaded once (cached in sources), reused for the task list and the evidence table.
records = S.load_records()

with st.sidebar:
    st.header("Comparison")
    mode = st.radio("Mode", ["Pairwise (X vs Y)", "Combination (A vs B)"])
    if mode.startswith("Pairwise"):
        x = st.selectbox("Model X", MODELS, index=_default_index("esm2_650M"))
        y = st.selectbox("Model Y", MODELS, index=_default_index("prost5"))
        task = st.selectbox("Focus task (optional)", ["(all tasks)"] + sorted(
            {r["task"] for r in records}))
    else:
        set_a = st.multiselect("Set A", MODELS,
                               default=[m for m in ("ankh_large", "prost5") if m in MODELS])
        set_b = st.multiselect("Set B", MODELS,
                               default=[m for m in ("esm2_650M", "esmc_600m") if m in MODELS])
    st.header("Generation")
    n = st.slider("Max hypotheses", 1, 8, 5)
    critique = st.checkbox("Run self-critique pass", value=True)
    model_id = st.text_input("Gemini model", value=gemini.model_name())
    run = st.button("Generate hypotheses", type="primary")

# Always show the deterministic evidence, even before generating.
try:
    if mode.startswith("Pairwise"):
        h2h = compare.head_to_head(records, x, y)
        st.subheader("Precomputed evidence (authoritative)")
        st.markdown(compare.render_head_to_head_md(h2h))
        with st.expander("Differentiating attributes"):
            st.markdown(compare.render_attribute_diff_md(compare.attribute_diff(x, y)))
    else:
        if set_a and set_b:
            combo = compare.combo_coverage(records, set_a, set_b)
            st.subheader("Precomputed evidence (authoritative)")
            st.markdown(compare.render_combo_md(combo))
        else:
            st.info("Pick at least one model in each set.")
except KeyError as e:
    st.error(str(e))

if run:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set in the environment.")
    else:
        with st.spinner("Generating and critiquing hypotheses…"):
            try:
                if mode.startswith("Pairwise"):
                    result = hypothesize.explain_pairwise(
                        x, y, task_focus=None if task == "(all tasks)" else task,
                        n=n, model=model_id, critique=critique)
                else:
                    result = hypothesize.explain_combo(
                        set_a, set_b, n=n, model=model_id, critique=critique)
            except gemini.GeminiError as e:
                st.error(str(e))
            else:
                st.subheader("Hypotheses (best-supported first)")
                _render_hypotheses(result)