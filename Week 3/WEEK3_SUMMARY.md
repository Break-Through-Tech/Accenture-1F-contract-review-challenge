# CUAD Week 3 — Chunking Strategy

## Goal

Find a chunking strategy for the CUAD `context` text that:
- keeps train and test separate,
- creates reasonably sized chunks,
- tries to respect logical paragraph/sentence boundaries,
- avoids cutting annotated legal clauses across chunks,
- does not create excessive duplicated text from overlap.

I compared several fixed-size and paragraph-aware strategies on the **official 408-contract train set and 102-contract test set**.

---

## What I tested

| Strategy | Idea |
|---|---|
| Fixed 300, no overlap | Every 300 whitespace-delimited words |
| Fixed 500, no overlap | Every 500 words |
| Fixed 350 + 50 overlap | Sliding 350-word windows |
| Fixed 500 + 100 overlap | Sliding 500-word windows |
| Paragraph 300 + 50 overlap | Respect paragraph/sentence boundaries where possible |
| Paragraph 325 + 50 overlap | Same, slightly larger |
| **Paragraph 350 + 50 overlap** | **Recommended** |
| Paragraph 400 + 75 overlap | Larger logical chunks |
| Paragraph 500, no overlap | Logical chunks without duplicated context |
| Paragraph 500 + 75 overlap | Large logical chunks |
| Paragraph 600 + 100 overlap | Largest tested chunks |

### Main metric: annotated span coverage

For every CUAD answer span, I checked whether the **entire labeled span** fits inside at least one generated chunk.

This is useful because a chunking strategy that cuts a labeled clause in half can make later clause classification harder.

---

## Recommended strategy

### **Paragraph-aware, ~350 words, 50-word overlap**

Process:
1. Start from each contract's raw `context`.
2. Prefer blank-line/paragraph boundaries.
3. If a paragraph is too large, split at sentence-like punctuation.
4. If a sentence/block is still too large, fall back to a word boundary.
5. Build chunks up to roughly **350 words**.
6. Give each later chunk about **50 words of backward overlap**.

### Results

| Metric | Train | Test |
|---|---:|---:|
| Annotated spans fully contained | **99.18%** | **99.55%** |
| Spans split across boundaries | 92 / 11,180 | 12 / 2,643 |
| Avg. chunks per contract | 28.7 | 25.7 |
| Avg. chunk size | 329 words | 325 words |
| 95th percentile chunk size | 398 words | 398 words |
| Max chunk size | 400 words | 400 words |
| Text processed because of overlap | 1.17x | 1.17x |

This gives **very high clause coverage (~99%+)** without the much larger chunks or heavier overlap of the 400–600 word alternatives.

---

## Why not just use fixed-size chunks?

Fixed chunks are simple, but they often cut directly through clauses.

Examples from the experiment:

- Fixed 300 with no overlap covered only **~87%** of annotated spans.
- Fixed 500 with no overlap covered only **~92%**.
- Adding overlap improves fixed chunks substantially, but paragraph-aware chunks preserve logical boundaries better.

Even **paragraph-aware 500-word chunks with no overlap** achieved about **99% coverage**, showing that logical boundaries themselves help a lot.

---

## Why not choose the highest-coverage strategy?

`paragraph_600_overlap_100` reached about **99.8% coverage**, slightly higher than the recommendation.

I did not choose it because:
- its chunks are much larger,
- larger chunks create more input for the model,
- a transformer such as BERT/RoBERTa commonly has a limited token window,
- "600 words" can easily become more than 600 tokenizer tokens.

So the 350 + 50 strategy is a better current balance between:
- logical coherence,
- annotation coverage,
- chunk size,
- duplicated context.

**Important:** this experiment measures size in whitespace-delimited words because the final transformer/tokenizer has not been fixed yet. Once the team chooses the actual model, the same algorithm should be converted to use that model's tokenizer and enforce a true token cap (for example, leaving room under a 512-token model limit).

---

## Output datasets

The `data/` folder contains the recommended chunking applied separately to:

- `train_chunks_paragraph350_overlap50.csv`
- `test_chunks_paragraph350_overlap50.csv`

Each row contains:
- contract title,
- chunk ID,
- raw character start/end,
- word count,
- chunk text,
- CUAD categories whose annotated spans are fully contained,
- CUAD categories that overlap the chunk.

The raw character offsets make it possible to map chunks back to the original contract text.

---

## Result files

The `results/` folder contains:

- `chunking_strategy_comparison.csv` — metrics for every tested strategy
- `chunking_contract_detail.csv` — per-contract results
- `strategy_span_coverage.png` — train/test coverage comparison
- `coverage_vs_redundancy.png` — coverage vs duplicated-text tradeoff
- `sample_recommended_chunks.csv` — example chunks from short/medium/long contracts

---

## Recommendation to the team

Use **paragraph-aware chunking with approximately 350 words and 50 words of overlap** as the current baseline.

Then, once the transformer is chosen, replace the word-count limit with the **actual tokenizer's token count** and retest nearby settings (for example ~300–400 tokens with ~40–60 tokens of overlap).

The main success metric should remain:

> **What percentage of annotated CUAD spans are fully contained in at least one chunk?**

This makes the chunking choice measurable rather than purely subjective.
