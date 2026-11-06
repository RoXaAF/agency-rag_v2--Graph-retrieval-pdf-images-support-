"""Build the vector index from docs/*.txt  ->  db/  (run: python ingest.py)"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
CHUNK_SIZE = 500
OVERLAP = 100


def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(ROOT / "db"))
    # Rebuild from scratch so deleted/edited docs never leave stale chunks behind.
    try:
        client.delete_collection("agency")
    except Exception:
        pass
    col = client.create_collection("agency", metadata={"hnsw:space": "cosine"})

    files = sorted((ROOT / "docs").glob("*.txt"))
    if not files:
        raise SystemExit("No .txt files found in docs/")
    for path in files:
        pieces = chunk(path.read_text(encoding="utf-8"))
        col.add(
            ids=[f"{path.name}-{i}" for i in range(len(pieces))],
            documents=pieces,
            embeddings=model.encode(pieces).tolist(),
            metadatas=[{"source": path.name}] * len(pieces),
        )
        print(f"Indexed {path.name}: {len(pieces)} chunks")


if __name__ == "__main__":
    main()
