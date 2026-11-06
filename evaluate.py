"""Run test_questions.json and print a report.   python evaluate.py [--limit N]"""
import argparse
import json
import time

import config
from rag import answer, graph_build, llm, store

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
limit = ap.parse_args().limit

llm.check_ollama()
tests = json.loads((config.ROOT / "test_questions.json").read_text(encoding="utf-8"))[:limit]
g, db = graph_build.load(), store.connect()
norm = lambda s: s.lower().replace("’", "'")

passed = cited = 0
for i, t in enumerate(tests, 1):
    start = time.time()
    text, trace = answer.ask(g, db, t["q"], verbose=False)
    kws, low = [norm(k) for k in t["keywords"]], norm(text)
    ok = all(k in low for k in kws) if t.get("all_keywords") else any(k in low for k in kws)
    has_cite = "[" in text and "]" in text
    passed += ok
    cited += has_cite or t["type"] == "trap"
    print(f"\n#{i}/{len(tests)} [{t['type']}] {t['q']}")
    print(f"  Expected : {t['expect']}")
    print(f"  Linked   : {list(trace['linked'].values())}  fallback={trace['fallback']}")
    print(f"  Chunks   : {trace['chunk_ids']}")
    print(f"  Answer   : {text[:400]}")
    print(f"  Result   : {'PASS' if ok else 'FAIL'}  cited={has_cite}  ({time.time() - start:.1f}s)")
print(f"\nScore: {passed}/{len(tests)} correct | cited {cited}/{len(tests)}")
