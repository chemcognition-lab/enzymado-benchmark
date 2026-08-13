"""Streamlit leaderboard — same data, same story as leaderboard.py's HTML/JS
report, built entirely in Python instead. Reuses leaderboard.py's data
extraction (extract_records/build_payload) so there is one source of truth
for "what counts as a usable result row"; only the rendering layer differs.

Usage:
    streamlit run src/visualizations/streamlit_leaderboard/app.py

Requires the "viz" extra: pip install -e ".[viz]"
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualizations.leaderboard import MODEL_FAMILIES, extract_records, load_results_card  # noqa: E402

FAMILY_LABEL = {"linear": "Linear", "mlp": "MLP", "xgboost": "XGBoost"}
# Same validated categorical palette as the HTML/JS leaderboard (light-mode steps) —
# see the dataviz skill's palette.md: slots 1/2/3, worst-pair CVD ΔE 9.2, all-pairs safe.
FAMILY_COLOR = {"linear": "#2a78d6", "mlp": "#eb6834", "xgboost": "#1baf7a"}
SPLIT_LABEL = {"random": "Random split", "loco": "Similarity-clustered (LOCO)"}
MODE_LABEL = {"no_attention": "Mean-pooled", "attention": "Attention-pooled"}
METRIC_LABEL = {
    "mae": "MAE", "rmse": "RMSE", "r2": "R²", "kendall_tau": "Kendall's τ",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
}


@st.cache_data
def load_data(results_card_path: str) -> tuple[pd.DataFrame, dict]:
    card = load_results_card(Path(results_card_path))
    records, task_meta = extract_records(card)
    df = pd.DataFrame([asdict(r) for r in records])
    meta = {task: asdict(m) for task, m in task_meta.items()}
    return df, meta


def ranked_embeddings(rows: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """One row per embedding: its per-family values plus a rank_value (mean
    across whichever families completed), sorted best-first. Mirrors
    rankedEmbeddings() in leaderboard_app.js exactly, just in pandas."""
    pivot = rows.pivot_table(index="embedding", columns="family", values="value", aggfunc="first")
    pivot["rank_value"] = pivot[[c for c in MODEL_FAMILIES if c in pivot.columns]].mean(axis=1)
    return pivot.sort_values("rank_value", ascending=not higher_is_better)


def main() -> None:
    st.set_page_config(page_title="Enzymado leaderboard", layout="wide")
    st.title("Enzymado leaderboard")
    st.caption(
        "Test-set primary metric per foundation-model embedding, from `results_card.json`. "
        "Bars are grouped by model family; ranking uses the mean of whichever families ran "
        "for that embedding."
    )

    default_card = str(REPO_ROOT / "results_card.json")
    results_card_path = st.sidebar.text_input("results_card.json path", value=default_card)
    try:
        df, task_meta = load_data(results_card_path)
    except FileNotFoundError:
        st.error(f"Couldn't find {results_card_path}")
        return
    if df.empty:
        st.warning("No usable records in this results_card.json.")
        return

    task_order = sorted(task_meta, key=lambda t: (task_meta[t]["scope"] != "properties", t))
    task = st.sidebar.selectbox(
        "Task", task_order,
        format_func=lambda t: f"{t}  ·  {task_meta[t]['scope']}",
    )
    meta = task_meta[task]

    task_rows = df[df["task"] == task]
    combos = task_rows[["split", "mode"]].drop_duplicates()
    split = st.sidebar.selectbox("Split", sorted(combos["split"].unique()), format_func=lambda s: SPLIT_LABEL.get(s, s))
    modes_for_split = sorted(combos.loc[combos["split"] == split, "mode"].unique())
    mode = st.sidebar.selectbox("Pooling", modes_for_split, format_func=lambda m: MODE_LABEL.get(m, m))

    rows = task_rows[(task_rows["split"] == split) & (task_rows["mode"] == mode)]
    ranked = ranked_embeddings(rows, meta["higher_is_better"])

    metric_label = METRIC_LABEL.get(meta["metric"], meta["metric"])
    st.subheader(task)
    direction = "higher is better" if meta["higher_is_better"] else "lower is better"
    st.caption(f"{metric_label} on the test split · {direction} · {meta['task_type']}")

    if ranked.empty:
        st.info("No completed runs for this task / split / pooling combination.")
        return

    view = st.radio("View", ["Chart", "Table"], horizontal=True, label_visibility="collapsed")

    if view == "Chart":
        fig = go.Figure()
        for family in MODEL_FAMILIES:
            if family not in ranked.columns:
                continue
            fig.add_trace(
                go.Bar(
                    name=FAMILY_LABEL[family],
                    x=ranked.index,
                    y=ranked[family],
                    marker_color=FAMILY_COLOR[family],
                )
            )
        fig.update_layout(
            barmode="group",
            xaxis_title=None,
            yaxis_title=metric_label,
            legend_title=None,
            height=460,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        # Native Plotly toolbar already includes a PNG export button (camera icon) —
        # no extra export code needed, unlike the hand-rolled SVG/PNG export in the
        # HTML/JS version.
        st.plotly_chart(fig, use_container_width=True)
    else:
        table = ranked[[c for c in MODEL_FAMILIES if c in ranked.columns]].reset_index()
        table.columns = ["Embedding model"] + [FAMILY_LABEL[c] for c in table.columns[1:]]
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        f"Ranked by mean {metric_label} across the model families that completed for each embedding."
    )


if __name__ == "__main__":
    main()
