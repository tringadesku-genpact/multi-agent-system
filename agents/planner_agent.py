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

    # Only enrich if task is very short/vague AND not obviously a direct request
    is_direct_request = _contains(
        task,
        "list", "show", "give", "tell", "top", "rank",
        "why", "how", "compare", "difference", "vs",
        "define", "what is", "explain"
    )

    if _task_length(task) <= 6 and intent_tags and not is_direct_request:
        retrieval_query = f"{original_task} {' '.join(intent_tags)} evidence"


    state["retrieval_query"] = retrieval_query

    # ------------------------------------------------------------
    # 4) Decide output format dynamically (generalized intent routing)
    # ------------------------------------------------------------

    sections = []

    # Business-only: greetings / small talk → don't answer
    if _contains(task, "hi", "hello", "hey", "yo", "sup", "thanks", "thank you"):
        sections = ["Not found in the sources."]

    # Ranking / "top N" / "best"
    elif _contains(task, "top", "best", "rank", "ranking", "highest", "lowest") or re.search(r"\btop\s*\d+\b", task):
        sections = [
            "Top results (ranked, with citations)",
            "Evidence per item (1–2 lines)",
        ]

    # List / enumerate / "tell me" / "show me"
    elif _contains(task, "list", "show", "give", "provide", "tell me", "tell us") or task.strip().endswith(":"):
        sections = [
            "List of items (with citations)",
        ]

    # "Why" questions → rationale + evidence + implications
    elif _contains(task, "why", "reason", "rationale"):
        sections = [
            "Direct answer (with citations)",
            "Key reasons (with citations)",
            "Implications / what it means",
        ]

    # "How" questions → steps/process + caveats
    elif _contains(task, "how", "steps", "process", "approach", "implement"):
        sections = [
            "Direct answer (with citations)",
            "Steps / approach (with citations)",
            "Pitfalls / caveats (with citations)",
        ]

    # "Where" / "used in" / "applied in"
    elif _contains(task, "where", "used", "applied", "application", "use case", "use-case"):
        sections = [
            "Direct answer (with citations)",
            "Where it applies (grouped by category/stage)",
            "Examples (with citations)",
        ]

    # Comparison
    elif _contains(task, "compare", "difference", "vs"):
        sections = [
            "Comparison summary (with citations)",
            "Key differences (with citations)",
            "Implications",
        ]

    # Risks
    elif _contains(task, "risk", "mitigat", "threat", "challenge", "issue", "problem"):
        sections = [
            "Main risks / issues (with citations)",
            "Mitigations (with citations)",
            "Practical advice",
        ]

    # Definition / explanation
    elif _contains(task, "define", "what is", "explain", "meaning"):
        sections = [
            "Explanation (with citations)",
            "Examples (with citations)",
            "Limitations",
        ]

    # Default business answer (structured, not too generic)
    else:
        sections = [
            "Direct answer (with citations)",
            "Key points (with citations)",
            "Recommendations (if supported by sources)",
        ]

    # Email override (keep business-style)
    if _contains(task, "email", "draft", "message"):
        sections = ["Email draft (grounded in sources)"]

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
