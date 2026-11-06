"""RAG core: retrieve -> prompt -> Gemma via Ollama.  CLI chat: python ask.py"""
import time
from pathlib import Path

import httpx

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
LLM_MODEL = "gemma4:12b"  # exact name from `ollama list`
TOP_K = 7
MAX_DISTANCE = 0.9  # cosine distance; chunks farther than this are treated as irrelevant
HISTORY_TURNS = 3   # previous user/assistant exchanges sent to the model

SYSTEM_PROMPT = (
    "You are the friendly assistant for Bright Studio, a small marketing agency. "
    "Answer using ONLY the context provided. If the answer is not in the context, "
    "say you don't know and suggest contacting hello@brightstudio.example. "
    "Be concise. Mention which source file(s) you used."
)

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_col = chromadb.PersistentClient(path=str(ROOT / "db")).get_collection("agency")


def retrieve(question, k=TOP_K):
    res = _col.query(query_embeddings=_embedder.encode([question]).tolist(), n_results=k)
    hits = zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    return [(d, m, dist) for d, m, dist in hits if dist <= MAX_DISTANCE]


def _search_query(question, history):
    """Short follow-ups ("and how long is the contract?") need the previous question for retrieval."""
    prev = [h["content"] for h in history if h["role"] == "user"]
    return f"{prev[-1]} {question}" if prev and len(question.split()) < 8 else question


def stream_answer(question, history=None):
    """Yield (partial_answer, hits). `history` is a list of {"role","content"} dicts."""
    history = history or []
    hits = retrieve(_search_query(question, history))
    context = "\n\n".join(f"[{m['source']}]\n{d}" for d, m, _ in hits) or "(no relevant context found)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": h["role"], "content": h["content"]} for h in history[-2 * HISTORY_TURNS:]]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    for attempt in range(2):  # Ollama can fail transiently on a cold load / low GPU memory
        text = ""
        try:
            for part in ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                think=False,
                keep_alive="30m",
                options={"temperature": 0, "num_ctx": 4096},
            ):
                text += part["message"]["content"] or ""
                yield text, hits
            return
        except (ollama.ResponseError, httpx.TransportError):
            if attempt == 1:
                raise
            time.sleep(3)


def ask(question, history=None):
    text, hits = "", []
    for text, hits in stream_answer(question, history):
        pass
    return text, hits


if __name__ == "__main__":
    while True:
        q = input("\nAsk about the agency (or 'quit'): ")
        if q.lower() == "quit":
            break
        answer, hits = ask(q)
        print("\n--- Retrieved chunks ---")
        for doc, m, dist in hits:
            print(f"[{m['source']}] ({dist:.2f}) {doc[:80]}...")
        print("\n--- Answer ---")
        print(answer)
