import re
from agents.state import AgentState, add_trace


# --- Intent detection helpers ---

def _contains(task: str, *words) -> bool:
    return any(re.search(rf"\b{w}\b", task, re.I) for w in words)


def _task_length(task: str) -> int:
    return len(task.split())


# --- Planner implementation ---

def run(state: AgentState) -> AgentState:
    original_task = state["task"].strip()
    task = original_task.lower()

    # ------------------------------------------------------------
    # 1) Decide which tools/agents are needed
    # ------------------------------------------------------------

    tools = ["retriever", "writer", "verifier"]

    # (later you may add "web" conditionally)
    state["tools_to_use"] = tools

    # ------------------------------------------------------------
    # 2) Break request into steps
    # ------------------------------------------------------------

    plan_steps = [
        "Analyze user intent and formulate retrieval query",
        "Use Retriever to gather evidence from documents",
        "Use Writer to synthesize answer from evidence",
        "Use Verifier to check grounding and citations",
    ]

    state["plan_steps"] = plan_steps

    # ------------------------------------------------------------
    # 3) Create retrieval strategy
    # ------------------------------------------------------------

    # Base query is always the user intent
    retrieval_query = original_task

    # Light enrichment based on intent (not domain hardcoded)
    intent_tags = []

    if _contains(task, "risk", "mitigat", "threat", "challenge"):
        intent_tags += ["risks", "mitigation", "limitations"]

    if _contains(task, "compare", "difference", "vs"):
        intent_tags += ["comparison", "differences"]

    if _contains(task, "define", "what is", "explain"):
        intent_tags += ["definition", "examples"]

    if _contains(task, "email", "draft", "message"):
        intent_tags += ["summary", "recommendations"]

    # Only enrich if task is very short/vague
    if _task_length(task) <= 6 and intent_tags:
        retrieval_query = f"{original_task} {' '.join(intent_tags)} evidence"

    state["retrieval_query"] = retrieval_query

    # ------------------------------------------------------------
    # 4) Decide output format dynamically
    # ------------------------------------------------------------

    sections = []

    if _task_length(task) <= 6:
        sections = ["Answer (with citations)"]

    elif _contains(task, "compare", "difference", "vs"):
        sections = [
            "Comparison summary (with citations)",
            "Key differences",
            "Implications",
        ]

    elif _contains(task, "risk", "mitigat", "threat", "challenge"):
        sections = [
            "Main risks (with citations)",
            "Mitigations",
            "Practical advice",
        ]

    elif _contains(task, "define", "what is", "explain"):
        sections = [
            "Explanation (with citations)",
            "Examples",
            "Limitations",
        ]

    else:
        sections = [
            "Executive brief",
            "Key insights (with citations)",
            "Recommendations",
        ]

    if _contains(task, "email", "draft", "message"):
        sections.append("Email draft")

    state["deliverable_sections"] = sections

    # ------------------------------------------------------------
    # 5) Trace logging
    # ------------------------------------------------------------

    add_trace(
        state,
        agent="planner",
        action="plan",
        detail="Interpreted intent and planned workflow",
        meta={
            "tools": tools,
            "steps": plan_steps,
            "retrieval_query": retrieval_query,
            "sections": sections,
        },
    )

    return state
