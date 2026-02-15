import os
import sys
import json
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from dashboard import render_dashboard
from agents.graph import run as run_graph

st.set_page_config(page_title="Tringa's Multi-Agent Chatbot", page_icon="🛒", layout="wide")

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

st.title("Multi-Agent System Demo")
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

    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your documents…")

    # Simple rate limiting per session
    if "request_times" not in st.session_state:
        st.session_state.request_times = []

    window_seconds = 60
    max_requests = 10

    if question:
        now = time.time()

        # Keep only timestamps within the rolling window
        st.session_state.request_times = [
            t for t in st.session_state.request_times
            if now - t < window_seconds
        ]

        if len(st.session_state.request_times) >= max_requests:
            st.error(f"Rate limit exceeded. Try again in a moment (max {max_requests}/min).")
            st.stop()

        st.session_state.request_times.append(now)

        # Clarification
        if st.session_state.pending_intent:
            question = f"{st.session_state.pending_intent} in {question}"
            st.session_state.pending_intent = None

        # User bubble
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Running agents..."):
                state = run_graph(task=question, top_k=top_k)

            final = (state.get("final") or state.get("draft") or "").strip()
            trace = state.get("trace", []) or []
            notes = state.get("notes", []) or []

            if state.get("needs_clarification"):
                st.session_state.pending_intent = state.get("task")

            sources = []
            for i, n in enumerate(notes, start=1):
                c = (n or {}).get("citation", {}) if isinstance(n, dict) else {}
                sources.append({
                    "n": i,
                    "file": c.get("source_file", ""),
                    "page": c.get("page", ""),
                    "chunk": c.get("chunk_in_page", ""),
                    "score": n.get("score") if isinstance(n, dict) else None,
                })

            blocked = any(
                e.get("agent") == "guardrails" and e.get("action") == "blocked"
                for e in trace
            )

            if blocked:
                st.error("Blocked by safety guardrails.")
                st.markdown(final)
            else:
                st.markdown(final)

                # ---- Sources (single, neat) ----
                if show_sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        if sources and isinstance(sources, list) and isinstance(sources[0], dict):
                            cols = [c for c in ["n", "file", "page", "chunk", "score"] if c in sources[0]]
                            st.dataframe(
                                pd.DataFrame(sources)[cols],
                                use_container_width=True,
                                hide_index=True,
                            )
                        elif sources:
                            for s in sources:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("No sources for this run.")

                if show_trace:
                    with st.expander(f"Trace ({len(trace)})"):
                        st.json(trace)

        st.session_state.messages.append({"role": "assistant", "content": final})

# --- Dashboard tab ---
with tab_dashboard:
    render_dashboard(ROOT_DIR)
