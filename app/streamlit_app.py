import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from agents.graph import run as run_graph

st.set_page_config(page_title="Tringa's Multi-Agent RAG", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #545d38; }

    /* Titles */
    h1, h2, h3 { color: #2f5d50; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 10px;
    }

    /* Buttons */
    button {
        background-color: #6b8f71 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Multi-Agent RAG Demo")
st.caption(
    "Ask questions about your documents. The system runs Guardrails → Planner → Retriever → Writer → Verifier "
    "(with optional retry) and logs each run."
)

with st.sidebar:
    st.header("Index")
    st.write("Put PDFs in `data/raw/` and run ingestion:")
    st.code("python -m agents.rag_ingest", language="bash")

    st.divider()
    top_k = st.slider("Top-K sources", 1, 10, 5)
    show_sources = st.toggle("Show sources", value=True)
    show_trace = st.toggle("Show trace", value=True)

tab_chat, tab_dashboard = st.tabs(["Chat", "Dashboard"])

# --- Chat tab ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your documents…")

    if question:
        # User bubble
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Running agents..."):
                state = run_graph(task=question, top_k=top_k)

            final = (state.get("final") or state.get("draft") or "").strip()
            trace = state.get("trace", []) or []
            sources = state.get("source_refs", []) or []

            blocked = any(
                e.get("agent") == "guardrails" and e.get("action") == "blocked"
                for e in trace
            )

            if blocked:
                st.error("Blocked by safety guardrails.")
                st.markdown(final)
            else:
                st.markdown(final)

                if show_sources and sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                if show_trace and trace:
                    with st.expander("Trace"):
                        for e in trace:
                            st.markdown(
                                f"- **{e.get('agent')}** :: {e.get('action')} :: {e.get('detail')}"
                            )
                            if e.get("meta"):
                                st.json(e["meta"])

        st.session_state.messages.append({"role": "assistant", "content": final})

# --- Dashboard tab ---
with tab_dashboard:
    st.subheader("Observability")
    log_file = "logs/runs.jsonl"

    if not os.path.exists(log_file):
        st.info("No runs logged yet. Ask a few questions first.")
    else:
        rows = [json.loads(l) for l in open(log_file, "r", encoding="utf-8") if l.strip()]
        df = pd.DataFrame(rows)

        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

        st.dataframe(df, use_container_width=True)

        st.metric("Total runs", len(df))
        if "retried" in df.columns:
            st.metric("Retry rate", f"{df['retried'].mean() * 100:.1f}%")

        # Show blocked rate if present
        if "trace" in df.columns:
            def _is_blocked(t):
                try:
                    return any(e.get("agent") == "guardrails" and e.get("action") == "blocked" for e in t)
                except Exception:
                    return False
            df["blocked"] = df["trace"].apply(_is_blocked)
            st.metric("Blocked rate", f"{df['blocked'].mean() * 100:.1f}%")
            st.bar_chart(df["blocked"])
