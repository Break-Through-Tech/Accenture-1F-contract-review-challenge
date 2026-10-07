#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# Only test the two new candidate strategies.
STRATEGIES = [
    ("fixed_tok_256_overlap_32", "fixed", 256, 32),
    ("fixed_tok_384_overlap_48", "fixed", 384, 48),
    ("fixed_tok_384_overlap_64", "fixed", 384, 64),
    ("fixed_tok_512_overlap_64", "fixed", 512, 64),

    ("paragraph_tok_256_overlap_32", "paragraph", 256, 32),
    ("paragraph_tok_384_overlap_48", "paragraph", 384, 48),
    ("paragraph_tok_384_overlap_64", "paragraph", 384, 64),
    ("paragraph_tok_448_overlap_64", "paragraph", 448, 64),
    ("paragraph_tok_480_overlap_64", "paragraph", 480, 64),
    ("paragraph_tok_512_overlap_64", "paragraph", 512, 64),
]


def get_category(qa):
    m = re.search(r'related to "([^"]+)"', qa.get("question", ""))

    if m:
        return m.group(1)

    return re.sub(
        r"_\d+$",
        "",
        qa["id"].split("__", 1)[1],
    )


def load_contracts(dataset, split):
    out = []

    for item in dataset["data"]:
        for pidx, p in enumerate(item["paragraphs"]):

            anns = []

            for qa in p["qas"]:
                cat = get_category(qa)

                for a in qa.get("answers", []):
                    anns.append(
                        {
                            "category": cat,
                            "start": a["answer_start"],
                            "end": a["answer_start"] + len(a["text"]),
                            "text": a["text"],
                        }
                    )

            out.append(
                {
                    "split": split,
                    "title": item["title"],
                    "paragraph_index": pidx,
                    "context": p["context"],
                    "answers": anns,
                }
            )

    return out


def tok_offsets(tokenizer, text):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    return enc["input_ids"], enc["offset_mapping"]


def make_chunk(text, offsets, i, j, cid):
    s = offsets[i][0]
    e = offsets[j - 1][1]

    return {
        "chunk_id": cid,
        "start_char": int(s),
        "end_char": int(e),
        "token_start": i,
        "token_end": j,
        "token_count": j - i,
        "text": text[s:e],
    }


def fixed_token_chunks(text, tokenizer, max_tokens, overlap):
    ids, offsets = tok_offsets(tokenizer, text)

    if not ids:
        return []

    step = max_tokens - overlap

    if step <= 0:
        raise ValueError(
            "overlap must be smaller than max_tokens"
        )

    out = []
    i = 0
    cid = 0

    while i < len(ids):

        j = min(
            i + max_tokens,
            len(ids),
        )

        out.append(
            make_chunk(
                text,
                offsets,
                i,
                j,
                cid,
            )
        )

        cid += 1

        if j == len(ids):
            break

        i += step

    return out


def paragraph_units(text):
    units = []

    for m in re.finditer(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        text,
        flags=re.S,
    ):
        if m.group(0).strip():
            units.append(
                (
                    m.start(),
                    m.end(),
                )
            )

    if not units and text.strip():

        units = [
            (
                len(text) - len(text.lstrip()),
                len(text.rstrip()),
            )
        ]

    return units


def sentence_units(text, s, e):
    seg = text[s:e]

    cuts = [0]

    for m in re.finditer(
        r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])",
        seg,
    ):
        cuts.append(
            m.end()
        )

    cuts.append(
        len(seg)
    )

    out = []

    for a, b in zip(
        cuts[:-1],
        cuts[1:],
    ):
        x = s + a
        y = s + b

        if text[x:y].strip():
            out.append(
                (
                    x,
                    y,
                )
            )

    return out


def token_count(
    tokenizer,
    text,
    s,
    e,
):
    return len(
        tokenizer(
            text[s:e],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def split_large(
    text,
    tokenizer,
    s,
    e,
    max_tokens,
):

    pieces = []

    cur = []
    cur_n = 0

    for x, y in sentence_units(
        text,
        s,
        e,
    ):

        n = token_count(
            tokenizer,
            text,
            x,
            y,
        )

        if n > max_tokens:

            if cur:

                pieces.append(
                    (
                        cur[0][0],
                        cur[-1][1],
                    )
                )

                cur = []
                cur_n = 0

            local = text[x:y]

            _, offs = tok_offsets(
                tokenizer,
                local,
            )

            i = 0

            while i < len(offs):

                j = min(
                    i + max_tokens,
                    len(offs),
                )

                pieces.append(
                    (
                        x + offs[i][0],
                        x + offs[j - 1][1],
                    )
                )

                i = j

        elif (
            not cur
            or cur_n + n <= max_tokens
        ):

            cur.append(
                (
                    x,
                    y,
                )
            )

            cur_n += n

        else:

            pieces.append(
                (
                    cur[0][0],
                    cur[-1][1],
                )
            )

            cur = [
                (
                    x,
                    y,
                )
            ]

            cur_n = n

    if cur:

        pieces.append(
            (
                cur[0][0],
                cur[-1][1],
            )
        )

    return pieces


def paragraph_token_chunks(
    text,
    tokenizer,
    max_tokens,
    overlap,
):

    ids, offs = tok_offsets(
        tokenizer,
        text,
    )

    if not ids:
        return []

    units = []

    for s, e in paragraph_units(text):

        if (
            token_count(
                tokenizer,
                text,
                s,
                e,
            )
            <= max_tokens
        ):

            units.append(
                (
                    s,
                    e,
                )
            )

        else:

            units.extend(
                split_large(
                    text,
                    tokenizer,
                    s,
                    e,
                    max_tokens,
                )
            )

    packed = []

    cur = []
    cur_n = 0

    for s, e in units:

        n = token_count(
            tokenizer,
            text,
            s,
            e,
        )

        if (
            not cur
            or cur_n + n <= max_tokens
        ):

            cur.append(
                (
                    s,
                    e,
                )
            )

            cur_n += n

        else:

            packed.append(
                (
                    cur[0][0],
                    cur[-1][1],
                )
            )

            cur = [
                (
                    s,
                    e,
                )
            ]

            cur_n = n

    if cur:

        packed.append(
            (
                cur[0][0],
                cur[-1][1],
            )
        )

    starts = np.array(
        [
            s
            for s, _ in offs
        ]
    )

    ends = np.array(
        [
            e
            for _, e in offs
        ]
    )

    out = []

    for cid, (s, e) in enumerate(packed):

        ts = int(
            np.searchsorted(
                ends,
                s,
                side="right",
            )
        )

        te = int(
            np.searchsorted(
                starts,
                e,
                side="left",
            )
        )

        te = max(
            ts + 1,
            min(
                te,
                len(offs),
            ),
        )

        if cid > 0:
            ts = max(
                0,
                ts - overlap,
            )

        # Make sure overlap never causes the raw chunk
        # to exceed max_tokens.
        if te - ts > max_tokens:
            ts = te - max_tokens

        out.append(
            make_chunk(
                text,
                offs,
                ts,
                te,
                cid,
            )
        )

    return out


def classify_chunk(ch, anns):

    full = [
        a
        for a in anns
        if (
            ch["start_char"]
            <= a["start"]
            and ch["end_char"]
            >= a["end"]
        )
    ]

    overlap = [
        a
        for a in anns
        if (
            a["start"]
            < ch["end_char"]
            and a["end"]
            > ch["start_char"]
        )
    ]

    full_categories = sorted(
        {
            a["category"]
            for a in full
        }
    )

    overlap_categories = sorted(
        {
            a["category"]
            for a in overlap
        }
    )

    partial = sorted(
        set(overlap_categories)
        - set(full_categories)
    )

    return (
        full,
        overlap,
        full_categories,
        overlap_categories,
        partial,
    )


def make_fn(
    tokenizer,
    kind,
    max_tokens,
    overlap,
):

    if kind == "fixed":

        return lambda t: fixed_token_chunks(
            t,
            tokenizer,
            max_tokens,
            overlap,
        )

    return lambda t: paragraph_token_chunks(
        t,
        tokenizer,
        max_tokens,
        overlap,
    )


def evaluate(
    contracts,
    name,
    fn,
    tokenizer,
):

    details = []

    sizes = []

    orig = 0
    proc = 0
    total = 0
    covered = 0
    partial_events = 0

    for c in contracts:

        chunks = fn(
            c["context"]
        )

        ids, _ = tok_offsets(
            tokenizer,
            c["context"],
        )

        orig += len(ids)

        proc += sum(
            x["token_count"]
            for x in chunks
        )

        sizes += [
            x["token_count"]
            for x in chunks
        ]

        cc = 0

        for a in c["answers"]:

            total += 1

            ok = any(
                (
                    x["start_char"]
                    <= a["start"]
                    and x["end_char"]
                    >= a["end"]
                )
                for x in chunks
            )

            if ok:

                covered += 1
                cc += 1

        for x in chunks:

            partial_events += len(
                classify_chunk(
                    x,
                    c["answers"],
                )[4]
            )

        details.append(
            {
                "split": c["split"],
                "strategy": name,
                "contract_title": c["title"],
                "paragraph_index": c[
                    "paragraph_index"
                ],
                "contract_tokens": len(ids),
                "chunk_count": len(chunks),
                "span_count": len(
                    c["answers"]
                ),
                "covered_span_count": cc,
                "span_coverage": (
                    cc
                    / len(c["answers"])
                    if c["answers"]
                    else np.nan
                ),
            }
        )

    n = sum(
        d["chunk_count"]
        for d in details
    )

    return (
        {
            "strategy": name,
            "contracts": len(contracts),
            "total_chunks": n,
            "avg_chunks_per_contract": (
                n / len(contracts)
            ),
            "avg_chunk_tokens": float(
                np.mean(sizes)
            ),
            "median_chunk_tokens": float(
                np.median(sizes)
            ),
            "p95_chunk_tokens": float(
                np.percentile(
                    sizes,
                    95,
                )
            ),
            "max_chunk_tokens": int(
                max(sizes)
            ),
            "span_coverage": (
                covered / total
            ),
            "covered_spans": covered,
            "total_spans": total,
            "split_spans": (
                total - covered
            ),
            "token_redundancy_ratio": (
                proc / orig
            ),
            "partial_overlap_category_events": (
                partial_events
            ),
        },
        pd.DataFrame(details),
    )


def build_dataset(
    dataset,
    split,
    fn,
):

    rows = []

    for item in dataset["data"]:

        for pidx, p in enumerate(
            item["paragraphs"]
        ):

            anns = []

            for qa in p["qas"]:

                cat = get_category(
                    qa
                )

                for a in qa.get(
                    "answers",
                    [],
                ):

                    anns.append(
                        {
                            "category": cat,
                            "start": a[
                                "answer_start"
                            ],
                            "end": (
                                a["answer_start"]
                                + len(a["text"])
                            ),
                            "text": a[
                                "text"
                            ],
                        }
                    )

            for ch in fn(
                p["context"]
            ):

                (
                    full,
                    overlap,
                    full_categories,
                    overlap_categories,
                    partial,
                ) = classify_chunk(
                    ch,
                    anns,
                )

                rows.append(
                    {
                        "split": split,
                        "contract_title": item[
                            "title"
                        ],
                        "paragraph_index": pidx,
                        "chunk_id": ch[
                            "chunk_id"
                        ],
                        "start_char": ch[
                            "start_char"
                        ],
                        "end_char": ch[
                            "end_char"
                        ],
                        "token_start": ch[
                            "token_start"
                        ],
                        "token_end": ch[
                            "token_end"
                        ],
                        "token_count": ch[
                            "token_count"
                        ],
                        "chunk_text": ch[
                            "text"
                        ],
                        "full_containment_labels": (
                            " | ".join(
                                full_categories
                            )
                        ),
                        "fully_contained_span_count": (
                            len(full)
                        ),
                        "any_overlap_labels": (
                            " | ".join(
                                overlap_categories
                            )
                        ),
                        "overlapping_span_count": (
                            len(overlap)
                        ),
                        "partial_only_categories": (
                            " | ".join(
                                partial
                            )
                        ),
                        "has_partial_only_overlap": (
                            bool(partial)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        default=(
            "train_separate_questions.json"
        ),
    )

    ap.add_argument(
        "--test",
        default="test.json",
    )

    ap.add_argument(
        "--out",
        default=(
            "token_chunking_outputs"
        ),
    )

    ap.add_argument(
        "--tokenizer",
        default=(
            "bert-base-uncased"
        ),
    )

    ap.add_argument(
        "--recommended-strategy",
        default=(
            "paragraph_tok_480_overlap_64"
        ),
    )

    args = ap.parse_args()

    out = Path(
        args.out
    )

    (
        out
        / "results"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out
        / "data"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    train = json.loads(
        Path(
            args.train
        ).read_text(
            encoding="utf-8"
        )
    )

    test = json.loads(
        Path(
            args.test
        ).read_text(
            encoding="utf-8"
        )
    )

    tok = (
        AutoTokenizer
        .from_pretrained(
            args.tokenizer,
            use_fast=True,
        )
    )

    if not tok.is_fast:
        raise ValueError(
            "Fast tokenizer required "
            "for exact offset mapping."
        )

    train_contracts = load_contracts(
        train,
        "train",
    )

    test_contracts = load_contracts(
        test,
        "test",
    )

    fns = {
        name: make_fn(
            tok,
            kind,
            max_tokens,
            overlap,
        )
        for (
            name,
            kind,
            max_tokens,
            overlap,
        ) in STRATEGIES
    }

    if (
        args.recommended_strategy
        not in fns
    ):

        raise ValueError(
            "Unknown recommended "
            f"strategy: "
            f"{args.recommended_strategy}"
        )

    summaries = []
    details = []

    for split, contracts in [
        (
            "train",
            train_contracts,
        ),
        (
            "test",
            test_contracts,
        ),
    ]:

        for name, fn in fns.items():

            print(
                f"Evaluating "
                f"{split}: {name}"
            )

            summary, detail = (
                evaluate(
                    contracts,
                    name,
                    fn,
                    tok,
                )
            )

            summary[
                "split"
            ] = split

            summary[
                "tokenizer"
            ] = args.tokenizer

            summaries.append(
                summary
            )

            details.append(
                detail
            )

    summary_df = (
        pd.DataFrame(
            summaries
        )
    )

    detail_df = (
        pd.concat(
            details,
            ignore_index=True,
        )
    )

    summary_df.to_csv(
        out
        / "results"
        / (
            "token_chunking_"
            "strategy_comparison.csv"
        ),
        index=False,
    )

    detail_df.to_csv(
        out
        / "results"
        / (
            "token_chunking_"
            "contract_detail.csv"
        ),
        index=False,
    )

    recommended_fn = fns[
        args.recommended_strategy
    ]

    train_chunks = build_dataset(
        train,
        "train",
        recommended_fn,
    )

    test_chunks = build_dataset(
        test,
        "test",
        recommended_fn,
    )

    train_chunks.to_csv(
        out
        / "data"
        / (
            "train_chunks_"
            f"{args.recommended_strategy}"
            ".csv"
        ),
        index=False,
    )

    test_chunks.to_csv(
        out
        / "data"
        / (
            "test_chunks_"
            f"{args.recommended_strategy}"
            ".csv"
        ),
        index=False,
    )

    label_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "chunks": len(
                    train_chunks
                ),
                "chunks_with_full_labels": int(
                    (
                        train_chunks[
                            "full_containment_labels"
                        ]
                        != ""
                    ).sum()
                ),
                "chunks_with_partial_only_overlap": int(
                    train_chunks[
                        "has_partial_only_overlap"
                    ].sum()
                ),
            },
            {
                "split": "test",
                "chunks": len(
                    test_chunks
                ),
                "chunks_with_full_labels": int(
                    (
                        test_chunks[
                            "full_containment_labels"
                        ]
                        != ""
                    ).sum()
                ),
                "chunks_with_partial_only_overlap": int(
                    test_chunks[
                        "has_partial_only_overlap"
                    ].sum()
                ),
            },
        ]
    )

    label_summary.to_csv(
        out
        / "results"
        / "chunk_labeling_summary.csv",
        index=False,
    )

    cols = [
        "split",
        "strategy",
        "avg_chunks_per_contract",
        "avg_chunk_tokens",
        "p95_chunk_tokens",
        "max_chunk_tokens",
        "span_coverage",
        "split_spans",
        "token_redundancy_ratio",
        "partial_overlap_category_events",
    ]

    print()

    print(
        summary_df[
            cols
        ]
        .sort_values(
            [
                "split",
                "span_coverage",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nRecommended labeling rule:"
    )

    print(
        "full containment = positive; "
        "partial-only overlap = "
        "ambiguous/mask, not clean negative."
    )


if __name__ == "__main__":
    main()