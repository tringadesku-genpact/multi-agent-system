from agents.state import AgentState, add_trace
from agents.rag_retrieve import retrieve_notes


def run(state: AgentState) -> AgentState:
    query = (state.get("retrieval_query") or "").strip()
    top_k = int(state.get("top_k", 5))

    if not query:
        state["notes"] = []
        # state["sources"] = []
        add_trace(
            state,
            agent="retriever",
            action="retrieve",
            detail="Empty retrieval query; returned 0 notes",
            meta={"query": query, "top_k": top_k, "notes": 0},
        )
        return state

    MIN_SCORE = 0.60  
    notes = retrieve_notes(query=query, top_k=top_k)
    # keep only relevant notes
    notes = [n for n in notes if float(n.get("score", 0) or 0) >= MIN_SCORE]
    state["notes"] = notes

    if not notes:
        # state["final"] = "Not found in the sources."
        state["stop"] = True
        add_trace(
            state,
            agent="retriever",
            action="no_evidence",
            detail="No relevant sources after score threshold; stopping",
            meta={"query": query, "top_k": top_k, "min_score": MIN_SCORE},
        )
        return state


    # structured sources
    source_refs = []
    for i, n in enumerate(notes, start=1):
        c = n.get("citation", {})
        source_refs.append(
            {
                "n": i,
                "file": c.get("source_file", ""),
                "page": c.get("page", ""),
                "chunk": c.get("chunk_in_page", ""),
                "score": round(float(n.get("score", 0.0)), 3) if n.get("score") is not None else None,
            }
        )

    # state["sources"] = source_refs

    add_trace(
        state,
        agent="retriever",
        action="retrieve",
        detail="Retrieved notes from FAISS",
        meta={"query": query, "top_k": top_k, "notes": len(notes)},
    )
    return state
