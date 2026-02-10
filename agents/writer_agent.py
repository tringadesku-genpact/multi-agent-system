import os
from dotenv import load_dotenv
from openai import OpenAI

from agents.state import AgentState, add_trace

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_sources(notes):
    parts = []
    for i, n in enumerate(notes, start=1):
        c = n["citation"]
        citation = f'{c["source_file"]} | page {c["page"]} | chunk {c["chunk_in_page"]}'
        parts.append(f"[{i}] {citation}\n{n['text']}")
    return "\n\n".join(parts)


def run(state: AgentState) -> AgentState:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env (never commit it).")

    task = state["task"]
    sections = state.get("deliverable_sections", [])
    notes = state.get("notes", [])

    if not notes:
        state["draft"] = "Not found in the sources."
        add_trace(state, "writer", "draft", "No notes returned; wrote not-found response")
        return state

    sources_block = _format_sources(notes)

    system = (
        "You are a business analyst for food supply chain technology.\n"
        "Use ONLY the provided sources.\n"
        "Cite sources using [1], [2], etc.\n"
        "Every paragraph must include at least one citation.\n"
        "If something is not supported, write: 'Not found in the sources.'\n"
        "Do not invent facts."
    )

    user = (
        f"Task:\n{task}\n\n"
        f"Deliverable sections:\n- " + "\n- ".join(sections) + "\n\n"
        f"Sources:\n{sources_block}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    state["draft"] = resp.choices[0].message.content

    add_trace(
        state,
        agent="writer",
        action="draft",
        detail="Generated deliverable draft from notes",
        meta={"notes_used": len(notes), "sections": sections},
    )
    return state
