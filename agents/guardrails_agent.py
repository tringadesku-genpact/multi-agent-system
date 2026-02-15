import re
from agents.state import AgentState, add_trace

INJECTION_PATTERNS = [
    r"ignore( all| previous)? instructions",
    r"disregard( all| previous)? instructions",
    r"system prompt",
    r"developer message",
    r"reveal .*prompt",
    r"show .*instructions",
    r"you are now",
    r"do anything now",
    r"jailbreak",
]

HARMFUL_PATTERNS = [
    r"\bcrack\b",
    r"\bhack\b",
    r"\bbreak into\b",
    r"\bexploit\b",
    r"\bbackdoor\b",
    r"\bbypass\b",
    r"\bprivilege escalation\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bkeylogger\b",
]

OVERRIDE_PATTERNS = [
    r"ignore (the )?sources",
    r"answer from (general|your) knowledge",
    r"without sources",
    r"without citations",
    r"no citations",
    r"don't cite",
    r"don't use (the )?retriever",
]

BLOCK_PATTERNS = HARMFUL_PATTERNS + OVERRIDE_PATTERNS + INJECTION_PATTERNS


def run(state: AgentState) -> AgentState:
    task = (state.get("task") or "").strip()
    lower = task.lower()

    add_trace(state, "guardrails", "check", "Checked input for safety")

    if len(task) > 4000:
        task = task[:4000]
        state["task"] = task
        add_trace(state, "guardrails", "truncate_input", "Truncated task to 4000 chars")

    if any(re.search(p, lower) for p in BLOCK_PATTERNS):
        state["final"] = "Not found in the sources."
        state["stop"] = True
        add_trace(state, "guardrails", "blocked", "Blocked unsafe/override request")
        return state

    state.pop("stop", None)
    return state
