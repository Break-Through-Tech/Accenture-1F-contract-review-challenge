# CUAD Week 3 — Chunking Strategy

## Goal

The Week 3 goal was to find a chunking strategy for the CUAD contract `context` text that:

- keeps the official train and test sets separate,
- produces chunks small enough for a transformer later,
- preserves legal clauses as much as possible,
- respects natural paragraph/sentence boundaries,
- avoids unnecessary duplicate text.

The main question was:

> **How can we split long contracts into smaller pieces without cutting important labeled clauses in half?**

---

## Input Data

The experiment uses the official CUAD JSON files separately:

- `train_separate_questions.json` — 408 contracts
- `test.json` — 102 contracts

The contract text comes from each contract's `paragraphs[].context`.

The CUAD annotated answer spans are used only to evaluate whether a chunking strategy preserves the full labeled text. The official train/test split is never changed.

---

# How the chunking works

## 1. Fixed-size chunking

The simplest baseline is to cut every N words.

Example:

```text
Chunk 1 = words 1–300
Chunk 2 = words 301–600
Chunk 3 = words 601–900
```

This is easy, but it can split a legal clause in the middle:

```text
Chunk 1:
"...Distributor may not assign this Agreement without prior written"

Chunk 2:
"consent of the Company..."
```

Neither chunk contains the full Anti-Assignment clause.

---

## 2. Overlap

Overlap repeats some text between neighboring chunks.

With 350-word chunks and 50-word overlap:

```text
Chunk 1 = words 1–350
Chunk 2 = words 301–650
```

Words 301–350 appear in both chunks.

### Why overlap helps

If an important clause starts near a chunk boundary, repeating nearby text gives the next chunk another chance to contain the whole clause.

### Cost of overlap

Overlap makes the model process some text more than once. So more overlap improves coverage, but increases computation and duplication.

---

## 3. Paragraph-aware chunking

The recommended strategy does not blindly cut every 350 words.

The algorithm:

1. prefers blank-line/paragraph boundaries,
2. if a paragraph is too large, tries sentence-like punctuation boundaries,
3. if a sentence/block is still too large, falls back to a word boundary,
4. packs neighboring logical units together until the target size is reached,
5. adds a small backward overlap to later chunks.

So the target is roughly 350 words, but the algorithm prefers natural legal-text boundaries.

---

# Strategies Tested

| Strategy | Description |
|---|---|
| Fixed 300, no overlap | 300-word chunks |
| Fixed 500, no overlap | 500-word chunks |
| Fixed 350 + 50 overlap | Sliding 350-word chunks |
| Fixed 500 + 100 overlap | Larger overlapping fixed chunks |
| Paragraph 300 + 50 overlap | Smaller paragraph-aware chunks |
| Paragraph 325 + 50 overlap | Slightly larger paragraph-aware chunks |
| **Paragraph 350 + 50 overlap** | **Recommended baseline** |
| Paragraph 400 + 75 overlap | Larger paragraph-aware chunks |
| Paragraph 500, no overlap | Large logical chunks without overlap |
| Paragraph 500 + 75 overlap | Large logical chunks with overlap |
| Paragraph 600 + 100 overlap | Largest tested option |

---

# Evaluation Criteria

## 1. Annotated span coverage

This was the main metric.

CUAD already tells us the exact character span of each labeled clause. For every annotated span, I checked whether the **entire span** fits inside at least one generated chunk.

A span counts as covered when:

```text
chunk_start <= answer_start
and
chunk_end >= answer_end
```

Higher coverage is better because it means fewer labeled clauses are cut apart.

---

## 2. Split spans

A split span is a CUAD annotation that is **not fully contained in any chunk**.

This is effectively the failure count for the main coverage metric.

Lower is better.

---

## 3. Average chunk size

This is the average number of words in a chunk.

Why it matters:

- chunks that are too small can lose useful context,
- chunks that are too large can contain unrelated material,
- large chunks may exceed the model's token limit.

---

## 4. 95th-percentile chunk size

This tells us how large the bigger chunks tend to be without being dominated by one extreme outlier.

For example:

```text
95th percentile = 398 words
```

means 95% of chunks are 398 words or shorter.

---

## 5. Maximum chunk size

The largest chunk produced.

This matters because transformer models have a maximum context window.

---

## 6. Average chunks per contract

This shows how many model inputs one contract would generate.

More chunks mean more computation and more predictions to combine later.

---

## 7. Word redundancy ratio

This measures the cost of overlap:

```text
total words across all chunks
--------------------------------
words in the original contracts
```

Interpretation:

```text
1.00x = no repeated text
1.17x = about 17% extra text
1.24x = about 24% extra text
```

Lower is more efficient, but some overlap is useful for preserving clauses near boundaries.

---

# Main Result

The best balance I found was:

> **Paragraph-aware chunking with ~350 words and 50 words of overlap**

Results:

| Metric | Train | Test |
|---|---:|---:|
| Annotated span coverage | **99.18%** | **99.55%** |
| Average chunk size | ~329 words | ~325 words |
| 95th percentile chunk size | ~398 words | ~398 words |
| Maximum chunk size | 400 words | 400 words |
| Average chunks per contract | ~28.7 | ~25.7 |
| Word redundancy ratio | ~1.17x | ~1.17x |

So the strategy preserved more than 99% of CUAD's annotated spans while adding only about 17% duplicated text.

---

# What the comparison showed

## Fixed 300, no overlap

Coverage was only about:

- Train: **86.7%**
- Test: **87.2%**

This cut too many clauses across boundaries.

## Fixed 500, no overlap

Coverage improved to roughly:

- Train: **92.2%**
- Test: **91.8%**

Larger chunks help, but arbitrary fixed boundaries still split clauses.

## Fixed 500 + 100 overlap

Coverage rose to about:

- Train: **99.0%**
- Test: **99.1%**

So overlap clearly helps, but this option processes about 24% extra text.

## Paragraph 500, no overlap

Coverage was about:

- Train: **99.1%**
- Test: **99.3%**

This is useful because it shows that logical paragraph/sentence boundaries themselves help a lot, even without overlap.

## Paragraph 600 + 100 overlap

This achieved the highest coverage, around **99.8%**, but I did not recommend it because the chunks are much larger and are more likely to be problematic once a real transformer tokenizer is used.

---

# Why 350 + 50 Was Selected

The goal was not to maximize one metric at any cost.

I wanted a balance between:

- clause preservation,
- logical coherence,
- chunk size,
- overlap cost,
- future model compatibility.

The 350 + 50 strategy gets well above 99% coverage while keeping chunks relatively compact.

In simple terms:

```text
small fixed chunks
    -> too many clauses split

larger fixed chunks
    -> better, but boundaries are arbitrary

paragraph-aware boundaries
    -> much better

paragraph-aware + modest overlap
    -> very high coverage without too much duplication
```

---

# Important Limitation: Words vs. Tokens

The current experiment measures size using whitespace-delimited words.

Transformers actually use **tokens**, not words.

For example, one complicated legal/company name may become several tokens.

So:

> **350 words does not guarantee fewer than 512 transformer tokens.**

This is why the current result should be treated as a strong logical chunking baseline.

Once the final model is chosen, the same experiment should be repeated with that model's tokenizer, likely testing something around:

```text
300–400 tokens
40–60 token overlap
paragraph-aware boundaries
```

---

# Output Files

## `data/train_chunks_paragraph350_overlap50.csv`

Recommended chunking applied to the official training set.

## `data/test_chunks_paragraph350_overlap50.csv`

Recommended chunking applied to the official test set.

Train and test stay separate.

Each row includes:

- contract title,
- chunk ID,
- raw character start/end,
- word count,
- chunk text,
- CUAD categories fully contained in the chunk,
- number of fully contained spans,
- categories overlapping the chunk.

The character offsets let us map any chunk back to the original contract text.

---

# Result Files

## `results/chunking_strategy_comparison.csv`

The main summary table.

It contains, for each strategy:

- average chunks per contract,
- average chunk size,
- 95th-percentile chunk size,
- maximum chunk size,
- annotation coverage,
- number of split spans,
- redundancy ratio.

This is the main quantitative evidence behind the recommendation.

## `results/chunking_contract_detail.csv`

Per-contract results.

Useful for finding:

- unusually long contracts,
- contracts that create many chunks,
- contracts where spans are still split.

## `results/strategy_span_coverage.png`

Visual comparison of annotation coverage across strategies.

Higher is better.

## `results/coverage_vs_redundancy.png`

Shows the tradeoff between:

- higher annotation coverage,
- extra duplicated text from overlap.

Ideally a strategy has high coverage without a large redundancy ratio.

## `results/sample_recommended_chunks.csv`

A few generated chunks from contracts of different lengths.

This is useful for manually checking that the chunk boundaries actually look sensible, rather than relying only on metrics.

---

# Reproducing the Experiment

The code is in:

```text
week3_chunking_experiment.py
```

Place the script where it can access:

```text
train_separate_questions.json
test.json
```

Then run:

```bash
python week3_chunking_experiment.py
```

The script:

1. loads train and test separately,
2. extracts each contract's `context`,
3. applies every chunking strategy,
4. checks each chunk against CUAD's annotated spans,
5. computes coverage and efficiency metrics,
6. generates comparison CSVs and graphs,
7. produces the recommended chunked train/test datasets.

---

# Current Recommendation

Use:

> **Paragraph-aware chunking with ~350 words and ~50 words of overlap**

as the current baseline.

The main justification is:

> It preserves over 99% of CUAD's annotated clause spans while keeping chunks reasonably small and adding only about 17% duplicated text.

Once the final transformer is chosen, repeat the same experiment using the model's actual tokenizer rather than word counts.
