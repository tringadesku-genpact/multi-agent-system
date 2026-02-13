import os
import json
import pandas as pd
import streamlit as st


def _is_blocked(trace_list) -> bool:
    try:
        return any(
            e.get("agent") == "guardrails" and e.get("action") == "blocked"
            for e in (trace_list or [])
        )
    except Exception:
        return False


def render_dashboard(ROOT_DIR: str):
    st.subheader("Observability")

    log_file = os.path.join(ROOT_DIR, "logs", "runs.jsonl")
    if not os.path.exists(log_file):
        st.info("No runs logged yet. Ask a few questions first.")
        return

    # Read JSONL
    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_row_id"] = i 
            rows.append(rec)

    df = pd.DataFrame(rows)


    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    if "trace" in df.columns:
        df["blocked"] = df["trace"].apply(_is_blocked)
    else:
        df["blocked"] = False

    if "retried" not in df.columns:
        df["retried"] = False

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df))
    c2.metric("Blocked rate", f"{df['blocked'].mean() * 100:.1f}%")
    c3.metric("Retry rate", f"{df['retried'].mean() * 100:.1f}%")

    lat = pd.to_numeric(df.get("latency_ms", pd.Series(dtype="float64")), errors="coerce")
    if lat.notna().any():
        c4.metric("Average Latency", f"{lat.mean():.0f} ms")
    else:
        c4.metric("Average Latency", "—")


    st.divider()

    # Runs over time
    st.markdown("### Runs over time")
    if "timestamp_utc" in df.columns and df["timestamp_utc"].notna().any():
        tmp = df.dropna(subset=["timestamp_utc"]).copy()
        tmp["day"] = tmp["timestamp_utc"].dt.date
        st.line_chart(tmp.groupby("day").size())
    else:
        st.info("No timestamps found in logs yet.")

    st.divider()

    # Summary table (latest first)
    st.markdown("### Runs (summary)")
    table = df.copy()

    if "task" in table.columns:
        table["task_preview"] = table["task"].astype(str).str.replace("\n", " ").str.slice(0, 80)
    else:
        table["task_preview"] = ""

    if "final" in table.columns:
        table["final_preview"] = table["final"].astype(str).str.replace("\n", " ").str.slice(0, 80)
    else:
        table["final_preview"] = ""

    if "timestamp_utc" in table.columns:
        table = table.sort_values("timestamp_utc", ascending=False)
    else:
        table = table.sort_values("_row_id", ascending=False)

    if "latency_ms" in table.columns:
        table["latency_ms"] = pd.to_numeric(table["latency_ms"], errors="coerce").round(0)

    cols = [c for c in ["timestamp_utc", "latency_ms", "task_preview", "blocked", "retried", "final_preview"] if c in table.columns]
    st.dataframe(table[cols], use_container_width=True, hide_index=True)

    st.divider()

    # Inspect a run (trace + sources)
    st.markdown("### Inspect a run")

    table_reset = table.reset_index(drop=True)
    labels = []
    for i, r in table_reset.iterrows():
        ts = r.get("timestamp_utc", "")
        taskp = str(r.get("task_preview", ""))[:40]
        labels.append(f"{i} | {ts} | {taskp}")

    if not labels:
        st.info("No runs to inspect yet.")
        return

    pick = st.selectbox("Select a run", labels, index=0)
    idx = int(pick.split(" | ")[0])
    selected_row_id = int(table_reset.iloc[idx]["_row_id"])

    full_rec = df[df["_row_id"] == selected_row_id].iloc[0]

    cA, cB = st.columns(2)

    with cA:
        st.markdown("**Trace**")
        st.json(full_rec.get("trace", []))

    with cB:
        st.markdown("**Sources**")

        notes = full_rec.get("notes", []) or []

        sources = []
        if isinstance(notes, list):
            for i, n in enumerate(notes, start=1):
                if not isinstance(n, dict):
                    continue
                c = n.get("citation") or {}
                sources.append({
                    "n": i,
                    "file": c.get("source_file", ""),
                    "page": c.get("page", ""),
                    "chunk": c.get("chunk_in_page", ""),
                    "score": n.get("score"),
                })

        if sources:
            src_df = pd.DataFrame(sources)
            if "file" in src_df.columns:
                src_df["file"] = (
                    src_df["file"].astype(str)
                    .str.split("/").str[-1]
                    .str.split("\\").str[-1]
                )
            st.dataframe(
                src_df[[c for c in ["n", "file", "page", "chunk", "score"] if c in src_df.columns]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No sources saved for this run.")

