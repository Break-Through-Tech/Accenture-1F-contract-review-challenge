# Tokenizer-Based CUAD Chunking Extension

## Goal

The original Week 3 chunking experiment used whitespace-delimited words. Since transformer models operate on tokens, I repeated the chunking analysis using the `bert-base-uncased` tokenizer.

The goal was to find a tokenizer-based chunking strategy that:

- preserves annotated CUAD clause spans,
- respects paragraph/sentence boundaries,
- stays safely below a 512-token transformer limit,
- avoids excessive repeated text from overlap.

## Main Finding

Paragraph-aware token chunking consistently outperformed fixed token windows.

The strongest practical baseline was:

> **Paragraph-aware chunking with a maximum of 480 raw tokens and 64 tokens of backward overlap.**

Results:

| Metric | Train | Test |
|---|---:|---:|
| Annotated span coverage | 98.92% | 99.36% |
| Fully preserved spans | 11,059 / 11,180 | 2,626 / 2,643 |
| Split spans | 121 | 17 |
| Avg. chunks per contract | 28.83 | 25.12 |
| Avg. chunk size | 427.5 tokens | 425.0 tokens |
| Max chunk size | 480 tokens | 480 tokens |
| Token redundancy ratio | 1.106x | 1.109x |

## Why 480 + 64?

A 512-token paragraph-aware strategy achieved slightly higher training coverage, but chunks of 512 raw tokens leave no room for model-specific special tokens such as `[CLS]` and `[SEP]`.

Using 480 raw tokens:

- leaves room below a typical 512-token model limit,
- preserves approximately 99% of CUAD annotations,
- requires only about 10–11% duplicated token processing,
- maintains natural paragraph/sentence boundaries.

The 448-token alternative provided slightly lower annotation coverage, so 480 + 64 was selected as the better tradeoff.

## Chunk Labeling Rule

For downstream clause classification:

1. A category is positive for a chunk only when at least one complete annotated span for that category is fully contained in the chunk.
2. If an annotation only partially overlaps the chunk, that category should be flagged or masked as ambiguous rather than treated as a clean negative.
3. Any-overlap labels are retained only for analysis/debugging.

This avoids teaching the model that a partial fragment of a true clause is a negative example.

## Current Recommendation

Use:

> **paragraph_tok_480_overlap_64**

as the current tokenizer-based chunking baseline.

This result currently uses `bert-base-uncased` as a provisional tokenizer. Once the team selects the final transformer model, rerun the experiment using that model's actual tokenizer before final model training.