# Chunker Fix: Token Counting Under DistilRoBERTa

## Summary

When the Week 3 paragraph-aware chunker (`paragraph_tok_480_overlap_64`) was run with the `distilroberta-base` tokenizer, it silently **dropped about 1% of every split's contract text** and **lost about 4–5% of annotated CUAD clauses**: they were never placed in any chunk. Those clauses were missing from the training data, so the model never saw them, and the test text they sat in was never evaluated.

The cause was a token-counting error that does not show up with `bert-base-uncased`, the tokenizer used for the original Week 3 experiment. It appeared only after `training/prepare_dataset.py` switched to the DistilRoBERTa tokenizer.

The fix is in `Week 3/tokenizer_chunking_experiment.py`. `training/prepare_dataset.py` now also checks that no text is dropped and reports clause coverage, so this cannot happen silently again.

## Impact

Measured on the output of `training/prepare_dataset.py` (DistilRoBERTa, 480 tokens, 64-token overlap):

| Split | Version | Chunks | Annotated spans fully in a chunk | Contract text never chunked | Contracts affected |
|---|---|---:|---:|---:|---:|
| train | before | 13,288 | 95.21% (10,644 / 11,180) | 1.25% (222,977 chars) | 296 / 408 |
| train | **after** | **14,015** | **98.62% (11,026 / 11,180)** | **0** | **0** |
| test | before | 2,782 | 96.33% (2,546 / 2,643) | 0.95% (37,445 chars) | 69 / 102 |
| test | **after** | **2,906** | **99.21% (2,622 / 2,643)** | **0** | **0** |

After the fix, DistilRoBERTa coverage matches the ~99% reported for BERT in `TOKENIZER_CHUNKING_NEXT_STEP.md`. The remaining ~1% are clauses longer than one chunk, or ones that straddle a boundary even with the overlap. These are expected and are masked rather than mislabeled.

**`bert-base-uncased` results are unchanged** (see [Week 3 results](#week-3-results-are-unaffected)).

## Root Cause

`paragraph_token_chunks` works in three steps:

1. Split the contract into paragraphs (and sentences, for very long paragraphs).
2. Pack consecutive paragraphs into a chunk until it reaches `max_tokens` (480).
3. Map each packed chunk onto the full contract's token sequence, add backward overlap, and clamp the chunk to `max_tokens`.

Step 2 measured each paragraph by **tokenizing it on its own** and adding up the counts. Step 3 then measured the same span **inside the full contract's token stream**. With BERT the two counts agree. With DistilRoBERTa they don't:

- **Newlines are tokens.** RoBERTa's byte-level BPE turns the `\n\n` between paragraphs into real tokens (`Ċ`, `ĊĊ`). BERT discards whitespace. Tokenizing a paragraph by itself never counts the newline tokens around it.
- **Context-dependent merges.** A paragraph tokenized on its own starts without a leading-space marker (`Ġ`), so its count can differ slightly from the same text inside the contract.

Contracts with many short paragraphs (cover pages, signature blocks, numbered lists) undercount the most. A chunk believed to hold 480 tokens can really hold well over 480.

Step 3 then contained this line:

```python
if te - ts > max_tokens:
    ts = te - max_tokens
```

It was meant to keep the *overlap* from pushing a chunk past the limit. But when the chunk was already too big, it moved the chunk's start forward and **cut off the beginning of new, non-overlapping text**. That text belonged to no other chunk, so it disappeared.

### Example

`DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement` (test set): the cover page is about 20 short paragraphs separated by blank lines. The first chunk was clipped to start at character 356, so the agreement date at characters 210–228 ("September 26, 2018") was in no chunk. Both its **Agreement Date** and **Effective Date** labels were lost.

## The Fix

`paragraph_token_chunks` now measures everything in **one coordinate system: token positions in the full contract**.

1. Each paragraph or sentence unit is mapped to its token range `[a, b)` in the full-document tokenization. Each range extends to where the next unit begins, so the newline tokens between units are counted.
2. A unit that is still longer than `max_tokens` is split into `max_tokens` windows.
3. Units are packed while `end - start <= max_tokens`, using these exact positions.
4. The backward overlap is added, and the clamp now **only trims overlap**. Packed spans never exceed `max_tokens`, so the clamp can never remove new text.

Paragraph and sentence boundaries, the 480/64 settings, the labeling rule and the output columns are all unchanged.

## New Safeguards in `training/prepare_dataset.py`

- **No-gap check.** For every contract, the first chunk must start at token 0, and each chunk must start at or before the previous chunk's end. Otherwise the script stops with `AssertionError: chunks leave a gap in contract: <title>`. Running the new script against the old chunker fails immediately, as intended.
- **Coverage report.** The script prints annotated-span coverage per split and writes `training/data/chunking_summary.csv`:

  ```text
  split,contracts,chunks,covered_spans,total_spans,span_coverage
  train,408,14015,11026,11180,0.9862
  test,102,2906,2622,2643,0.9921
  ```

  If coverage drops well below ~98% after a tokenizer or settings change, look at the chunker before training.

## Week 3 Results Are Unaffected

The full `tokenizer_chunking_experiment.py` strategy comparison (10 strategies × train/test, `bert-base-uncased`) was re-run with the fix. Results are **identical**:

- every row of `token_chunking_strategy_comparison.csv`: chunk counts, covered spans, coverage, chunk sizes, redundancy;
- `train_/test_chunks_paragraph_tok_480_overlap_64.csv`: byte-for-byte identical;
- all 1,020 rows of `token_chunking_contract_detail.csv`.

BERT's WordPiece tokenizer drops whitespace, and its tokens never span a whitespace boundary. So counting a paragraph on its own and counting it inside the full contract always gave the same number, and the bug never triggered. The committed Week 3 results stay valid.

## Action for the Team

If you generated `training/data/` or trained a model before this fix, **regenerate the dataset and retrain**:

```bash
python training/prepare_dataset.py
python training/train_distilroberta.py --epochs 3
```

Models trained on the old data never saw about 4.8% of the training clauses, and their test metrics were computed with about 3.7% of test clauses missing.

## How to Reproduce the Measurement

1. Run `training/prepare_dataset.py` once with the old chunker and once with the fixed one.
2. For each contract, mark the characters covered by any chunk (`start_char`/`end_char` in `*_chunks.csv`).
3. Count non-whitespace characters that no chunk covers.
4. Count CUAD answer spans that no single chunk fully contains.
