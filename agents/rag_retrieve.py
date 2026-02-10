import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.jsonl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_metadata():
    rows = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieve_notes(query: str, top_k: int = 5):
    """Return list of notes with text + citation."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Run: python agents/rag_ingest.py first")

    index = faiss.read_index(str(INDEX_PATH))
    metadata = _load_metadata()

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, top_k)

    notes = []
    for doc_id, score in zip(ids[0], scores[0]):
        if doc_id < 0:
            continue
        row = metadata[int(doc_id)]

        notes.append({
            "text": row["text"],
            "citation": {
                "source_file": row["source_file"],
                "page": row["page"],
                "chunk_in_page": row["chunk_in_page"],
            },
            "score": float(score),
        })

    return notes


def main():
    query = input("Enter your question: ").strip()
    if not query:
        return

    notes = retrieve_notes(query)

    print("\n=== Top results ===")
    for i, n in enumerate(notes, start=1):
        c = n["citation"]
        print(f"\n#{i} score={n['score']:.3f}")
        print(f"Citation: {c['source_file']} | page {c['page']} | chunk {c['chunk_in_page']}")
        print(n["text"][:400])


if __name__ == "__main__":
    main()
