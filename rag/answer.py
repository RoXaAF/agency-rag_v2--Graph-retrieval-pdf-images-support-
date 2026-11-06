"""Prompt assembly and generation."""
import config
from rag import graph_query, llm, store

SYSTEM = (
    "You answer questions about Bright Studio, a marketing agency. Use ONLY the facts and source chunks provided. "
    "If the answer is not in them, say you do not know. Cite sources as [file, page], "
    "e.g. [pricing.txt] or [annual_report_2025.pdf, p2]. Be concise."
)


def _cite(source, page=0):
    return f"[{source}, p{page}]" if page else f"[{source}]"


def build_prompt(question, trace, chunks):
    """Trim to fit num_ctx (~4 chars/token): drop facts first, then chunks."""
    budget = config.NUM_CTX * 4 - len(SYSTEM) - len(question) - 1500  # leave room for the answer
    facts = [f"- {f['s']} --{f['p']}--> {f['o']}  {_cite(f['source'])}" for f in trace["facts"]]
    parts = [f"{_cite(c['source'], c['page'])} ({c['id']})\n{c['text']}" for c in chunks]
    while facts and sum(map(len, facts)) + sum(map(len, parts)) > budget:
        facts.pop()
    while len(parts) > 1 and sum(map(len, facts)) + sum(map(len, parts)) > budget:
        parts.pop()
    trace["facts_used"], trace["chunks_used"] = len(facts), len(parts)
    return ("Graph facts:\n" + ("\n".join(facts) or "(none)") +
            "\n\nSource chunks:\n" + ("\n\n".join(parts) or "(none)") + f"\n\nQuestion: {question}")


def ask(g, db, question, verbose=True):
    trace = graph_query.retrieve(g, db, question)
    chunks = store.get(db, trace["chunk_ids"])
    prompt = build_prompt(question, trace, chunks)
    trace["chunk_ids"] = trace["chunk_ids"][:trace["chunks_used"]]
    if verbose:
        print(f"  entities in question: {trace['entities']}")
        print(f"  linked to nodes:      {trace['linked'] or 'NONE'}")
        print(f"  graph facts:          {trace['facts_used']} used")
        for f in trace["facts"][:trace["facts_used"]]:
            print(f"    {f['s']} --{f['p']}--> {f['o']}   ({f['chunk_id']})")
        print(f"  chunks used:          {trace['chunk_ids']}")
        print(f"  fallback FTS5 search: {'YES (no entity linked)' if trace['fallback'] else 'no'}")
    text = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return text.strip(), trace
