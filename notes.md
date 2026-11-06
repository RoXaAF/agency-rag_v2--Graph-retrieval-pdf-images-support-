# Experiment log

Scoring: `python evaluate.py` runs the 11 questions in `test_questions.json` (5 direct, 2 fuzzy, 2 multi-file, 2 traps).
An answer passes if it contains the expected keyword(s). "Retrieval" means the expected source file was among the retrieved chunks.

## Baseline: paragraph chunking (900 chars), k = 5, num_ctx 4096
Score: 9/11, retrieval 8/9

- #6 (fuzzy, "one-off website project" price): FAIL. **Retrieval problem.** Only `services.txt` and `faq.txt` came back; the standalone web design price lives in `pricing.txt`. The question never says "price" or "package", so services chunks outranked it.
- #8 (multi-file, "cheapest package and who handles that work"): FAIL. **Retrieval problem.** `pricing.txt` was found, but the question never mentions social media, so no `team.txt` chunk was retrieved. The model correctly said the context did not specify who handles it.

## Experiment 1: k = 5 -> 7
Score: 10/11, retrieval 9/9

- What changed: `TOP_K` in `ask.py`. A retrieval-only check showed `pricing.txt` appears at rank 6 for #6.
- Result: #6 now passes. #8 still fails: `team.txt` is still not in the top 7.
- Why I think so: more chunks improved recall. Prompts are larger (about 7 x 900 chars), but still fit in 4096 tokens.
- Remaining miss (#8) needs a two-step lookup (find the cheapest package, then find who handles it). Fixes to try next: query decomposition by the LLM, or hybrid keyword search.

## Experiment 2: chunking, fixed 500 chars -> paragraph-based 900 chars (done before the baseline)
- 29 chunks became 23 across 6 files (the PDF added 2).
- Chunks keep whole sections plus the document title, so one retrieved chunk usually holds a full fact.
- Ingestion skips unchanged files (content hash) and embeds all new chunks in one batch: a re-run with no changes takes about 8 s, mostly model loading, and embeds nothing.
- Per-answer time is about 7 to 16 s on this machine (4 GB GPU, 12B model split across GPU and CPU).

## Other observations

- `num_ctx` 8192 caused a `CUDA error: out of memory` on the 4 GB GPU. I lowered it to 4096, added one automatic retry in `ask.py`, and have not seen the error since.
- Trap questions pass. For "legal services" the docs explicitly say we do not offer them, so the bot answers "No" rather than "I don't know". That is correct. The CEO's home address gets "I don't know".
- The PDF (`annual_report_2025.pdf`) is indexed by text. Its chart (`results_chart.png`) is not read by the pipeline, so the PDF also contains the chart numbers as plain text. Question #9 is answered from that PDF text.

## Ideas not yet tried
- Hybrid search (keyword + vector) or a reranker
- LLM query decomposition for multi-file questions like #8
- Switching the embedding model to `nomic-embed-text` via Ollama
- Reading text out of images (OCR) so chart PNGs can be indexed directly

---
## GraphRAG baseline (HOPS=2, chunk 500/100, FUZZY_MERGE=90, prompt v1)
Score: 10/13, cited 12/13 (the uncited one is a trap answered "I do not know").
- Fails #5, #6, #7 (all multi-hop). Cause is retrieval: #5 linked only "branding" (not the branding lead / PeakFit); #6/#7 had the right seed but the expansion ranked case-study/PDF chunks above team.txt / pricing.txt.
- Traps and paraphrases passed, mostly via the FTS5 fallback (entity linking missed 5 of 13 questions).
- Bug found and fixed: fuzzy merge joined "$1,200"/"$1,800", "$750"/"$950"; names with digits are no longer merged.
- Next experiments: HOPS=3, raise MAX_CHUNKS, rank chunks by number of facts touched, add FTS5 results alongside graph chunks.
