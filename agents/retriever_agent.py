from agents.state import AgentState, add_trace
from agents.rag_retrieve import retrieve_notes


def run(state: AgentState) -> AgentState:
    query = (state.get("retrieval_query") or "").strip()
    top_k = int(state.get("top_k", 5))

    if not query:
        state["notes"] = []
        state["sources"] = []
        add_trace(
            state,
            agent="retriever",
            action="retrieve",
            detail="Empty retrieval query; returned 0 notes",
            meta={"query": query, "top_k": top_k, "notes": 0},
        )
        return state

    notes = retrieve_notes(query=query, top_k=top_k)
    state["notes"] = notes

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

    state["sources"] = source_refs

    add_trace(
        state,
        agent="retriever",
        action="retrieve",
        detail="Retrieved notes from FAISS",
        meta={"query": query, "top_k": top_k, "notes": len(notes)},
    )
    return state
