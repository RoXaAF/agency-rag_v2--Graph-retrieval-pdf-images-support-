"""Commands: ingest, build-graph, ask, chat, inspect.   python cli.py <command> [--limit N]"""
import argparse
import random
import time

import config
from rag import answer, graph_build, llm, loaders, store


def cmd_ingest(a):
    llm.check_ollama(need_vision=True)
    files = sorted(p for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in loaders.LOADERS)
    files = files[:a.limit] if a.limit else files
    db = store.connect()
    store.reset(db)
    total = 0
    for i, p in enumerate(files, 1):
        print(f"file {i}/{len(files)}: {p.name}")
        chunks = loaders.LOADERS[p.suffix.lower()](p)
        store.add(db, chunks)
        kinds = {}
        for c in chunks:
            kinds[c["type"]] = kinds.get(c["type"], 0) + 1
        pages = f", pages {sorted({c['page'] for c in chunks})}" if p.suffix == ".pdf" else ""
        print(f"  -> {len(chunks)} chunks {kinds}{pages}")
        total += len(chunks)
    print(f"Done: {total} chunks in {config.DB_PATH}")


def cmd_build(a):
    llm.check_ollama()
    chunks = store.all_chunks(store.connect())
    if not chunks:
        raise SystemExit("No chunks. Run `python cli.py ingest` first.")
    t = time.time()
    g = graph_build.build(chunks, a.limit)
    graph_build.save(g)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {config.GRAPH_PATH} ({time.time() - t:.1f}s)")


def cmd_ask(a):
    llm.check_ollama()
    g, db = graph_build.load(), store.connect()
    print(f"Q: {a.question}")
    text, _ = answer.ask(g, db, a.question)
    print(f"\nA: {text}")


def cmd_chat(a):
    llm.check_ollama()
    g, db = graph_build.load(), store.connect()
    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            text, _ = answer.ask(g, db, q)
            print(f"\nA: {text}")


def cmd_inspect(a):
    g = graph_build.load()
    print(f"Nodes: {g.number_of_nodes()}  Edges: {g.number_of_edges()}")
    print("\nTop entities (by degree):")
    for n, d in sorted(g.degree, key=lambda x: -x[1])[:15]:
        print(f"  {d:3d}  {n}")
    print("\nEntity merges:")
    merged = [(n, d["aliases"]) for n, d in g.nodes(data=True) if d.get("aliases")]
    for n, al in merged:
        print(f"  {n}  <-  {al}")
    if not merged:
        print("  (none)")
    edges = list(g.edges(data=True))
    print("\nSample edges:")
    for u, v, d in random.sample(edges, min(20, len(edges))):
        print(f"  {u} --{d['relation']}--> {v}   ({d['chunk_id']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ingest", cmd_ingest), ("build-graph", cmd_build)):
        s = sub.add_parser(name)
        s.add_argument("--limit", type=int, help="process only the first N items")
        s.set_defaults(fn=fn)
    s = sub.add_parser("ask")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)
    sub.add_parser("chat").set_defaults(fn=cmd_chat)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
