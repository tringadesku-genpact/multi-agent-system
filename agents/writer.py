import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.rag_retrieve import retrieve_notes

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def format_sources(notes):
    parts = []
    for i, n in enumerate(notes, start=1):
        c = n["citation"]
        citation = f'{c["source_file"]} | page {c["page"]} | chunk {c["chunk_in_page"]}'
        parts.append(f"[{i}] {citation}\n{n['text']}")
    return "\n\n".join(parts)


def answer_with_sources(task: str, top_k: int = 5) -> str:
    notes = retrieve_notes(task, top_k=top_k)

    if not notes:
        return "Not found in the sources."

    context = format_sources(notes)

    system_prompt = """
You are an analyst for food supply chain technology.

Rules:
- Use ONLY the provided sources.
- Cite using [1], [2] style.
- Every paragraph must contain at least one citation.
- If something is not supported, write: "Not found in the sources."
- Be concise and factual.
"""

    user_prompt = f"""
Task: {task}

Sources:
{context}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY in .env first")
        return

    task = input("Enter your question/task: ").strip()
    if not task:
        return

    answer = answer_with_sources(task)
    print("\n=== Grounded Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
