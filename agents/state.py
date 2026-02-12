from typing import TypedDict, List, Dict, Any


class TraceEvent(TypedDict, total=False):
    agent: str
    action: str
    detail: str
    meta: Dict[str, Any]


class RAGNote(TypedDict):
    text: str
    citation: Dict[str, Any]
    score: float


class AgentState(TypedDict, total=False):
    task: str
    top_k: int

    # planner outputs
    retrieval_query: str
    deliverable_sections: List[str]

    # retriever outputs
    notes: List[RAGNote]

    # writer outputs
    draft: str

    # verifier outputs later
    final: str
    needs_retry: bool
    retried: bool

    # logs
    trace: List[TraceEvent]

    # --- ADDED ---
    stop: bool


def add_trace(state: AgentState, agent: str, action: str, detail: str = "", meta=None) -> None:
    if meta is None:
        meta = {}
    state.setdefault("trace", []).append(
        {"agent": agent, "action": action, "detail": detail, "meta": meta}
    )
