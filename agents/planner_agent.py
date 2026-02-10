from agents.state import AgentState, add_trace


def run(state: AgentState) -> AgentState:
    task = state["task"].strip()

    # Minimal planner: use task as the initial retrieval query
    state["retrieval_query"] = task

    # Minimal deliverable template (we’ll expand later if needed)
    state["deliverable_sections"] = [
        "Executive brief",
        "Key insights (with citations)",
        "Risks & mitigations",
        "Recommendations",
        "Action items",
    ]

    add_trace(
        state,
        agent="planner",
        action="plan",
        detail="Created retrieval query and deliverable sections",
        meta={"sections": state["deliverable_sections"]},
    )
    return state
