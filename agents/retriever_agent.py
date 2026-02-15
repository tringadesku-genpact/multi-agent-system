from agents.state import AgentState, add_trace
from agents.rag_retrieve import retrieve_notes
import re

# Detect vague / underspecified prompts
_GENERIC = re.compile(r"\b(compare|two|risk|risks|list|top|tell|explain|what|why|how)\b", re.I)

def _is_too_vague(q: str) -> bool:
    q = (q or "").strip()
    if len(q.split()) <= 4:
        return True
    generic_hits = len(_GENERIC.findall(q))
    return generic_hits >= max(2, len(q.split()) // 2)


def run(state: AgentState) -> AgentState:
    query = (state.get("retrieval_query") or "").strip()
    top_k = int(state.get("top_k", 5))

    if not query:
        state["notes"] = []
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
    notes = [n for n in notes if float(n.get("score", 0) or 0) >= MIN_SCORE]
    state["notes"] = notes

    # No evidence
    if not notes:
        add_trace(
            state,
            agent="retriever",
            action="no_evidence",
            detail="No relevant sources after score threshold; stopping",
            meta={"query": query, "top_k": top_k, "min_score": MIN_SCORE},
        )

        state["notes"] = []
        state["needs_retry"] = False
        state["stop"] = True

        if _is_too_vague(query):
            state["needs_clarification"] = True
            state["clarification_question"] = (
                "I can answer that question if you provide more context, rephrase to something more domain specific!"
            )
            state["final"] = state["clarification_question"]
        else:
            state["final"] = (
                "The retrieved documents do not provide sufficient evidence "
                "to answer this question."
            )

        return state

    add_trace(
        state,
        agent="retriever",
        action="retrieve",
        detail="Retrieved notes from FAISS",
        meta={"query": query, "top_k": top_k, "notes": len(notes)},
    )

    return state
