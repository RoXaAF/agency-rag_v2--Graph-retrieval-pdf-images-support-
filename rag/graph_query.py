"""Entity linking, graph expansion, edge ranking."""
import re
from collections import deque

from rapidfuzz import fuzz

import config
from rag import graph_build, llm, store

ENTITY_PROMPT = """List the named entities (people, companies, services, packages, products, places) mentioned in the question.
Return JSON only: {"entities": ["...", "..."]}
Question: """


def extract_entities(question):
    data = llm.chat_json([{"role": "user", "content": ENTITY_PROMPT + question}])
    ents = data.get("entities", []) if isinstance(data, dict) else []
    return [e for e in ents if isinstance(e, str) and e.strip()]


def link(g, entities):
    """Exact name -> alias -> fuzzy. Returns {entity: node}."""
    aliases = {a: n for n, d in g.nodes(data=True) for a in d.get("aliases", [])}
    linked = {}
    for e in entities:
        name = graph_build.normalize(e)
        if name in g:
            linked[e] = name
        elif name in aliases:
            linked[e] = aliases[name]
        else:
            best = max(g.nodes, key=lambda n: fuzz.token_set_ratio(name, n), default=None)
            if best and fuzz.token_set_ratio(name, best) >= config.FUZZY_LINK:
                linked[e] = best
    return linked


def expand(g, seeds):
    """BFS up to HOPS, edges treated as undirected. Returns {(u, v, key): distance}.
    A hub node contributes at most MAX_EDGES_PER_NODE edges (lowest-degree neighbours first)."""
    found, seen = {}, {s: 0 for s in seeds}
    queue = deque(seeds)
    while queue:
        n = queue.popleft()
        d = seen[n]
        if d >= config.HOPS:
            continue
        edges = list(g.out_edges(n, keys=True)) + list(g.in_edges(n, keys=True))
        edges.sort(key=lambda e: g.degree(e[1] if e[0] == n else e[0]))
        for u, v, k in edges[:config.MAX_EDGES_PER_NODE]:
            found.setdefault((u, v, k), d + 1)
            other = v if u == n else u
            if other not in seen:
                seen[other] = d + 1
                queue.append(other)
    return found


def rank(g, found, seeds, question):
    """Prefer close edges, edges touching several seeds, edges whose relation/entities appear in the question."""
    qwords = set(re.findall(r"\w+", question.lower()))
    scored = []
    for (u, v, k), dist in found.items():
        rel = g[u][v][k]["relation"]
        n_seeds = len({u, v} & set(seeds))
        score = 3.0 / dist
        score += 2.0 if n_seeds > 1 else 0
        score += 1.5 * bool(set(rel.split("_")) & qwords)
        score += 0.5 * bool(set(re.findall(r"\w+", u + " " + v)) & qwords)
        score -= 0.1 * min(g.degree(u), g.degree(v)) ** 0.5
        scored.append((score, u, v, k))
    scored.sort(key=lambda x: -x[0])
    facts = []
    for score, u, v, k in scored[:config.MAX_FACTS]:
        d = g[u][v][k]
        facts.append({"s": u, "p": d["relation"], "o": v, "chunk_id": d["chunk_id"], "source": d["source"], "score": score})
    return facts


def retrieve(g, db, question):
    """Returns trace: entities, linked, facts, chunk_ids, fallback."""
    entities = extract_entities(question)
    linked = link(g, entities)
    facts, chunk_ids, fallback = [], [], False
    if linked:
        seeds = list(dict.fromkeys(linked.values()))
        facts = rank(g, expand(g, seeds), seeds, question)
        for f in facts:  # chunks in order of best fact
            if f["chunk_id"] not in chunk_ids:
                chunk_ids.append(f["chunk_id"])
        chunk_ids = chunk_ids[:config.MAX_CHUNKS]
    if not chunk_ids:
        fallback = True
        chunk_ids = [c["id"] for c in store.search(db, question, config.MAX_CHUNKS)]
    return {"entities": entities, "linked": linked, "facts": facts, "chunk_ids": chunk_ids, "fallback": fallback}
