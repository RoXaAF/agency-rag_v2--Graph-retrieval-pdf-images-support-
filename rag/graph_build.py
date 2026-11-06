"""Triple extraction, entity normalization, graph creation."""
import json
import re
import time

import networkx as nx
from rapidfuzz import fuzz

import config
from rag import llm

# One place for the extraction prompt. Bump config.PROMPT_VERSION when editing it.
EXTRACT_PROMPT = """Extract facts from the text as triples (subject, relation, object).
Rules:
- Use short, consistent entity names (e.g. "Marcus Webb", "Growth package", "Bright Studio").
- Replace pronouns and phrases like "the agency", "we", "our" with the real name (the agency is "Bright Studio").
- Include only facts stated in the text. Do not guess.
- Prefer these relation verbs: leads, works_on, offers, costs, served, achieved, includes, has_role, requires, located_in. Use another short snake_case verb only if none fits.
- Put numbers and prices in the object (e.g. "$1,800 per month").
Return JSON only: {"triples": [{"s": "...", "p": "...", "o": "..."}]}

Text:
"""

ARTICLES = re.compile(r"^(the|a|an)\s+")


def normalize(name):
    name = " ".join(str(name).lower().split())
    return ARTICLES.sub("", name).strip(" .,;:\"'")


def extract_triples(chunk):
    k = llm.key(chunk["text"], config.PROMPT_VERSION, config.LLM_MODEL)

    def run():
        data = llm.chat_json([{"role": "user", "content": EXTRACT_PROMPT + chunk["text"]}])
        if not isinstance(data, dict) or not isinstance(data.get("triples"), list):
            return None
        return [t for t in data["triples"] if isinstance(t, dict) and all(isinstance(t.get(x), str) and t[x].strip() for x in "spo")]

    try:
        triples, hit = llm.cached("triples", k, run)
    except Exception as e:  # never crash the whole run
        print(f"    ! {chunk['id']}: {e}")
        return [], False
    if triples is None:
        print(f"    ! invalid JSON for {chunk['id']}, skipped")
        (config.CACHE_DIR / "triples" / f"{k}.json").unlink(missing_ok=True)
        return [], False
    return triples, hit


def build(chunks, limit=None):
    chunks = chunks[:limit] if limit else chunks
    raw = []  # (s, p, o, chunk)
    for i, c in enumerate(chunks, 1):
        t0 = time.time()
        triples, hit = extract_triples(c)
        print(f"  chunk {i}/{len(chunks)} {c['id']}: {len(triples)} triples {'(cached)' if hit else f'{time.time() - t0:.1f}s'}")
        raw += [(t["s"], t["p"], t["o"], c) for t in triples]

    # Count surface forms, then merge near-duplicates (most frequent form wins).
    counts = {}
    for s, _, o, _ in raw:
        for n in (normalize(s), normalize(o)):
            if n:
                counts[n] = counts.get(n, 0) + 1
    canon, merges = {}, []
    for name in sorted(counts, key=lambda n: (-counts[n], n)):
        # Never fuzzy-merge names containing digits: "$1,200" vs "$1,800" are different facts.
        has_digit = any(ch.isdigit() for ch in name)
        match = None if has_digit else next(
            (c for c in set(canon.values()) if not any(ch.isdigit() for ch in c) and fuzz.ratio(name, c) >= config.FUZZY_MERGE), None)
        canon[name] = match or name
        if match:
            merges.append((name, match))

    g = nx.MultiDiGraph()
    for s, p, o, c in raw:
        s, o = canon.get(normalize(s)), canon.get(normalize(o))
        if not s or not o or s == o:
            continue
        for n in (s, o):
            if n not in g:
                g.add_node(n, aliases=[], mentions=0)
            g.nodes[n]["mentions"] += 1
        g.add_edge(s, o, relation=" ".join(p.lower().split()).replace(" ", "_"), chunk_id=c["id"], source=c["source"])
    for alias, c in merges:
        if c in g and alias not in g.nodes[c]["aliases"]:
            g.nodes[c]["aliases"].append(alias)
    return g


def save(g, path=None):
    path = path or config.GRAPH_PATH
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nx.node_link_data(g, edges="edges", source="from", target="to"), indent=1), encoding="utf-8")


def load(path=None):
    path = path or config.GRAPH_PATH
    if not path.exists():
        raise SystemExit("No graph found. Run `python cli.py ingest` then `python cli.py build-graph`.")
    return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), edges="edges", source="from", target="to", multigraph=True, directed=True)
