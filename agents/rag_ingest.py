import os
import json
from pathlib import Path

import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer


RAW_PDFS_DIR = Path("data/raw_pdfs")
INDEX_DIR = Path("data/index")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.jsonl"

# Chunking settings (good defaults)
CHUNK_SIZE = 900      # characters
CHUNK_OVERLAP = 150   # characters
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def extract_pdf_pages(pdf_path: Path):
    """Return list of (page_number, text) from a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text") or ""
        text = " ".join(text.split())  # normalize whitespace
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Simple character chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def main():
    if not RAW_PDFS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_PDFS_DIR}")

    pdf_files = sorted([p for p in RAW_PDFS_DIR.glob("*.pdf")])
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {RAW_PDFS_DIR}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_texts = []
    metadata_rows = []

    print(f"Found {len(pdf_files)} PDFs. Extracting and chunking...")

    chunk_global_id = 0
    for pdf in pdf_files:
        pages = extract_pdf_pages(pdf)
        for page_num, page_text in pages:
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for local_chunk_id, chunk in enumerate(chunks):
                all_texts.append(chunk)
                metadata_rows.append({
                    "id": chunk_global_id,
                    "source_file": pdf.name,
                    "page": page_num,
                    "chunk_in_page": local_chunk_id,
                    "text": chunk
                })
                chunk_global_id += 1

    if not all_texts:
        raise RuntimeError("No text extracted from PDFs. Are they scanned images?")

    print(f"Total chunks: {len(all_texts)}. Embedding...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine-like similarity if embeddings are normalized
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ Done.")
    print(f"Saved FAISS index: {INDEX_PATH}")
    print(f"Saved metadata:  {META_PATH}")
    print(f"Chunks indexed:  {index.ntotal}")


if __name__ == "__main__":
    main()
