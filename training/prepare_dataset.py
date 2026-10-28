#!/usr/bin/env python3
"""Task #2: finalize chunking and build the tokenized multi-label dataset.

Reuses the Week 3 paragraph-aware chunker (Week 3/tokenizer_chunking_experiment.py)
with the recommended paragraph_tok_480_overlap_64 strategy, but re-runs it with the
model's own tokenizer (distilroberta-base) instead of the provisional bert-base-uncased,
as TOKENIZER_CHUNKING_NEXT_STEP.md recommends.

Labeling rule (same as Week 3):
  - category fully contained in a chunk   -> label 1
  - category only partially overlaps      -> masked (ignored in the loss / metrics)
  - otherwise                             -> label 0

Outputs in --out (default training/data, gitignored):
  train_chunks.csv / test_chunks.csv    chunk text + label strings
  train_tokenized.pt / test_tokenized.pt  input_ids, labels, label_mask, contract titles
  labels.json                           ordered category list = model output heads
  dataset_summary.csv                   chunk counts and positives per category
  chunking_summary.csv                  per split: chunks, annotated-span coverage

Sanity checks (the script fails if violated): every contract is fully covered
by its chunks with no gaps, so no contract text is silently dropped.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKER_PATH = REPO_ROOT / "Week 3" / "tokenizer_chunking_experiment.py"

# Week 3 recommended strategy: paragraph-aware, 480 raw tokens, 64-token overlap.
CHUNK_KIND, CHUNK_MAX_TOKENS, CHUNK_OVERLAP = "paragraph", 480, 64


def load_chunker():
    spec = importlib.util.spec_from_file_location("tokenizer_chunking_experiment", CHUNKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_labels(value):
    return [x for x in str(value).split(" | ") if x and x != "nan"]


def check_chunks(chunks_df, dataset, chunker):
    """Assert chunks tile every contract and return annotated-span coverage."""
    for title, g in chunks_df.groupby("contract_title", sort=False):
        g = g.sort_values("chunk_id")
        starts, ends = g["token_start"].to_numpy(), g["token_end"].to_numpy()
        if starts[0] != 0 or (starts[1:] > ends[:-1]).any():
            raise AssertionError(f"chunks leave a gap in contract: {title}")

    covered = total = 0
    by_title = {t: g for t, g in chunks_df.groupby("contract_title")}
    for contract in chunker.load_contracts(dataset, ""):
        g = by_title[contract["title"]]
        for a in contract["answers"]:
            total += 1
            covered += bool(((g["start_char"] <= a["start"]) & (g["end_char"] >= a["end"])).any())
    return covered, total


def encode_split(chunks_df, tokenizer, categories, max_length):
    index = {c: i for i, c in enumerate(categories)}
    enc = tokenizer(list(chunks_df["chunk_text"]), truncation=True, max_length=max_length)

    labels = torch.zeros((len(chunks_df), len(categories)), dtype=torch.float32)
    label_mask = torch.ones_like(labels)
    for row, (full, partial) in enumerate(
        zip(chunks_df["full_containment_labels"], chunks_df["partial_only_categories"])
    ):
        for cat in split_labels(full):
            if cat in index:
                labels[row, index[cat]] = 1.0
        for cat in split_labels(partial):
            if cat in index:
                label_mask[row, index[cat]] = 0.0

    return {
        "input_ids": enc["input_ids"],
        "labels": labels,
        "label_mask": label_mask,
        "contract_title": list(chunks_df["contract_title"]),
        "categories": categories,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(REPO_ROOT / "data/cuad/train_separate_questions.json"))
    ap.add_argument("--test", default=str(REPO_ROOT / "data/cuad/test.json"))
    ap.add_argument("--tokenizer", default="distilroberta-base")
    ap.add_argument("--out", default=str(REPO_ROOT / "training/data"))
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--min-train-positives",
        type=int,
        default=10,
        help="Drop categories with fewer positive train chunks than this (too rare to learn).",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chunker = load_chunker()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for exact offset mapping.")

    chunk_fn = chunker.make_fn(tokenizer, CHUNK_KIND, CHUNK_MAX_TOKENS, CHUNK_OVERLAP)

    splits = {}
    chunking_rows = []
    for split, path in [("train", args.train), ("test", args.test)]:
        print(f"Chunking {split} with {args.tokenizer} ({CHUNK_KIND} {CHUNK_MAX_TOKENS}/{CHUNK_OVERLAP})...")
        dataset = json.loads(Path(path).read_text(encoding="utf-8"))
        df = chunker.build_dataset(dataset, split, chunk_fn)
        covered, total = check_chunks(df, dataset, chunker)
        chunking_rows.append(
            {
                "split": split,
                "contracts": df["contract_title"].nunique(),
                "chunks": len(df),
                "covered_spans": covered,
                "total_spans": total,
                "span_coverage": round(covered / total, 4),
            }
        )
        df.to_csv(out / f"{split}_chunks.csv", index=False)
        splits[split] = df
        print(f"  {df['contract_title'].nunique()} contracts -> {len(df)} chunks, "
              f"annotated span coverage {covered / total:.2%} ({covered}/{total})")

    pd.DataFrame(chunking_rows).to_csv(out / "chunking_summary.csv", index=False)

    # Output heads: every CUAD category with enough fully-contained train positives.
    train_counts = pd.Series(
        [c for labels in splits["train"]["full_containment_labels"] for c in split_labels(labels)]
    ).value_counts()
    categories = sorted(train_counts[train_counts >= args.min_train_positives].index)
    dropped = sorted(set(train_counts.index) - set(categories))
    print(f"Using {len(categories)} categories; dropped (too rare): {dropped or 'none'}")
    (out / "labels.json").write_text(json.dumps(categories, indent=2), encoding="utf-8")

    summary_rows = []
    for split, df in splits.items():
        encoded = encode_split(df, tokenizer, categories, args.max_length)
        torch.save(encoded, out / f"{split}_tokenized.pt")
        positives = encoded["labels"].sum(0)
        masked = (1 - encoded["label_mask"]).sum(0)
        for i, cat in enumerate(categories):
            summary_rows.append(
                {
                    "split": split,
                    "category": cat,
                    "positive_chunks": int(positives[i]),
                    "masked_chunks": int(masked[i]),
                    "total_chunks": len(df),
                }
            )
        max_len = max(len(ids) for ids in encoded["input_ids"])
        print(f"  {split}: max sequence length incl. special tokens = {max_len}")

    pd.DataFrame(summary_rows).to_csv(out / "dataset_summary.csv", index=False)
    print(f"Saved dataset to {out}")


if __name__ == "__main__":
    main()
