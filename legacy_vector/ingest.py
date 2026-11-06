"""Build/refresh the vector index from docs/*.txt and docs/*.pdf  ->  db/  (run: python ingest.py [--full])

Chunking: split on blank lines (paragraphs/sections), pack paragraphs into chunks of up to
CHUNK_SIZE characters, carry the last paragraph over as overlap, and prefix every chunk with the
document title. Fewer, self-contained chunks = faster embedding, retrieval and shorter LLM prompts.
Unchanged files (same content hash) are skipped, and all new chunks are embedded in one batch.
"""
import hashlib
import json
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
CHUNK_SIZE = 900
OVERLAP_PARAS = 1  # paragraphs repeated at the start of the next chunk


def read_doc(path: Path) -> str:
    if path.suffix == ".pdf":
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def chunk(text, size=CHUNK_SIZE):
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if len(paras) == 1 and len(paras[0]) > size:  # e.g. PDF text with no blank lines
        paras = [p.strip() for p in paras[0].split("\n") if p.strip()]
    if not paras:
        return []
    title = paras[0].split("\n")[0][:80]
    chunks, cur = [], []
    for p in paras:
        if cur and sum(len(x) for x in cur) + len(p) > size:
            chunks.append("\n\n".join(cur))
            cur = cur[-OVERLAP_PARAS:] if OVERLAP_PARAS else []
        cur.append(p)
    chunks.append("\n\n".join(cur))
    return [c if c.startswith(title) else f"{title}\n{c}" for c in chunks]


def main(full=False):
    client = chromadb.PersistentClient(path=str(ROOT / "db"))
    manifest_path = ROOT / "db" / "manifest.json"
    manifest = {} if full or not manifest_path.exists() else json.loads(manifest_path.read_text())
    if full:
        try:
            client.delete_collection("agency")
        except Exception:
            pass
    col = client.get_or_create_collection("agency", metadata={"hnsw:space": "cosine"})

    files = sorted(p for p in (ROOT / "docs").iterdir() if p.suffix in (".txt", ".pdf"))
    if not files:
        raise SystemExit("No .txt or .pdf files found in docs/")

    # Remove chunks of deleted files, and of changed files (re-added below).
    names = {p.name for p in files}
    for gone in set(manifest) - names:
        col.delete(where={"source": gone})
        del manifest[gone]

    ids, docs, metas = [], [], []
    for path in files:
        text = read_doc(path)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if manifest.get(path.name) == digest:
            print(f"Unchanged  {path.name}")
            continue
        col.delete(where={"source": path.name})
        pieces = chunk(text)
        ids += [f"{path.name}-{i}" for i in range(len(pieces))]
        docs += pieces
        metas += [{"source": path.name}] * len(pieces)
        manifest[path.name] = digest
        print(f"Indexed    {path.name}: {len(pieces)} chunks")

    if docs:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        col.add(ids=ids, documents=docs, embeddings=model.encode(docs, batch_size=64).tolist(), metadatas=metas)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Done. {col.count()} chunks in index.")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
