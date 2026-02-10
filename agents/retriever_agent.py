from agents.state import AgentState, add_trace
from agents.rag_retrieve import retrieve_notes


def run(state: AgentState) -> AgentState:
    query = state["retrieval_query"]
    top_k = state.get("top_k", 5)

    notes = retrieve_notes(query=query, top_k=top_k)
    state["notes"] = notes

    add_trace(
        state,
        agent="retriever",
        action="retrieve",
        detail="Retrieved notes from FAISS",
        meta={"query": query, "top_k": top_k, "notes": len(notes)},
    )
    return state
