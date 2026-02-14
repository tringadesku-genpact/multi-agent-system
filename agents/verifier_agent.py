import re
from agents.state import AgentState, add_trace
from agents.query_rewriter_agent import run as rewrite_query

SECRET_PATTERNS = [
    r"OPENAI_API_KEY\s*=\s*\S+",
    r"sk-[A-Za-z0-9]{20,}",
]

def _redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = text
    for p in SECRET_PATTERNS:
        redacted = re.sub(p, "[REDACTED]", redacted)
    return redacted


CITATION_RE = re.compile(r"\[(\d+)\]")


def _split_body_and_sources(text: str):
    marker_patterns = ["\n### Sources", "\n## Sources", "\n# Sources"]
    for m in marker_patterns:
        idx = text.find(m)
        if idx != -1:
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""


def _paragraphs(text: str):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _has_citation(paragraph: str) -> bool:
    return bool(CITATION_RE.search(paragraph))


def _citations_in_range(text: str, max_n: int) -> bool:
    nums = [int(m.group(1)) for m in CITATION_RE.finditer(text)]
    return all(1 <= n <= max_n for n in nums) if nums else False


def _is_heading(p: str) -> bool:
    return p.lstrip().startswith("#")


def _is_generic_conclusion(p: str) -> bool:
    # Generic wrap-up sentences that are not new factual claims.
    low = p.lower()
    starters = (
        "in conclusion",
        "overall",
        "in summary",
        "therefore",
        "by focusing",
        "by addressing",
        "by improving",
    )
    return any(low.startswith(s) for s in starters)


def _needs_citation(p: str) -> bool:
    if _is_heading(p):
        return False
    if _is_generic_conclusion(p) and len(p.split()) < 30:
        return False
    if len(p.split()) < 8:
        return False
    return True


def run(state: AgentState) -> AgentState:
    draft = state.get("draft", "")

    # Output guardrail: redact secrets before anything else 
    redacted = _redact_secrets(draft)
    if redacted != draft:
        add_trace(
            state,
            "verifier",
            "redacted_secrets",
            "Redacted secret-like patterns from draft",
        )
    draft = redacted
    state["draft"] = draft  # uses redacted version

    notes = state.get("notes", [])
    max_n = len(notes)

    if not draft.strip():
        state["final"] = "Not found in the sources."
        state["needs_retry"] = False
        add_trace(state, "verifier", "verify", "Empty draft; set final to not found")
        return state

    body, sources_appendix = _split_body_and_sources(draft)
    paras = _paragraphs(body)

    missing_citation = [
        p for p in paras
        if _needs_citation(p)
        and ("Not found in the sources." not in p)
        and (not _has_citation(p))
    ]

    citations_ok = _citations_in_range(draft, max_n) if max_n > 0 else False

    add_trace(
        state,
        "verifier",
        "verify",
        "Checked citations and grounding (headings/conclusions ignored; Sources appendix excluded from paragraph checks)",
        meta={
            "paragraphs_checked": len(paras),
            "missing_citation_paragraphs": len(missing_citation),
            "citations_in_range": citations_ok,
            "notes_available": max_n,
            "has_sources_appendix": bool(sources_appendix),
        },
    )

    # Retry once if problems
    if (missing_citation or not citations_ok) and not state.get("retried", False):
        state["retried"] = True
        state["needs_retry"] = True

        rewrite_query(state)

        add_trace(
            state,
            "verifier",
            "retry_requested",
            "Grounding failed; requesting one retry",
            meta={"new_query": state.get("retrieval_query", "")},
        )
        return state

    # If still failing after retry - stop
    if missing_citation or not citations_ok:
        state["final"] = "Not found in the sources."
        state["needs_retry"] = False
        state["stop"] = True

        add_trace(
            state,
            "verifier",
            "blocked_unverified",
            "Failed grounding after retry; stopping",
            meta={
                "missing_citation_paragraphs": len(missing_citation),
                "citations_in_range": citations_ok,
            },
        )
        return state

    state["final"] = draft
    state["needs_retry"] = False
    add_trace(state, "verifier", "finalized", "Answer finalized after strict grounding")
    return state


