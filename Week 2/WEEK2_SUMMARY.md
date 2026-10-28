# CUAD Week 2 — Cleaning + Class Imbalance

## What I did

I made an independent pass over the official CUAD train/test files to:
1. lightly clean contract formatting without changing legal meaning,
2. reorganize the annotations into an easier contract-by-category format,
3. measure class imbalance across the 41 CUAD categories,
4. inspect common words and category-specific terms,
5. note open questions for the Accenture advisor.

I kept the official **408-contract training split** and **102-contract test split** unchanged.

---

## 1. Cleaning approach

I used **conservative cleaning only**.

### Changed
- converted non-breaking spaces to normal spaces,
- normalized newline encodings,
- collapsed repeated spaces/tabs,
- reduced very large blank-line runs,
- trimmed leading/trailing whitespace.

### Intentionally kept
- punctuation,
- capitalization,
- numbers and dollar amounts,
- dates and durations,
- stopwords,
- legal terms and boilerplate.

This is important because phrases such as **"not"**, **"unless"**, **"30 days"**, **"$500,000"**, **"perpetual"**, and **"irrevocable"** can materially change contractual meaning.

The original JSON files were not modified. The cleaned text is stored separately.

---

## 2. Important dataset-structure finding

The train and test JSON files represent the same 41 CUAD categories, but their raw QA rows are structured differently:

- **Train:** 41–112 raw QA entries per contract (about 55 on average)
- **Test:** exactly 41 raw QA entries per contract
- Both still contain exactly **41 unique categories per contract**

The reason is that the training file can split multiple answer spans for the same category into separate QA entries.

Therefore, I **aggregated by contract + category** before calculating class imbalance.

Example:

| contract | category | present | answer_count |
|---|---|---:|---:|
| Contract A | Exclusivity | 1 | 3 |
| Contract A | Source Code Escrow | 0 | 0 |

This avoids accidentally counting one contract several times simply because it has several annotated spans for one category.

I also checked all **13,823 annotated answer spans**, and all of their original `answer_start` offsets matched the raw contract text exactly.

---

## 3. Cleaned result datasets

The `data/` folder contains:

- `train_contracts_cleaned.csv` — one row per training contract, including cleaned text
- `test_contracts_cleaned.csv` — one row per test contract
- `train_category_labels.csv` — one row per training contract × category
- `test_category_labels.csv` — one row per test contract × category

The category-label files are mainly for analysis and make class imbalance much easier to inspect.

---

## 4. Class imbalance

The categories are **very unevenly distributed**.

### Most common overall

| Category | Positive contracts | Rate |
|---|---:|---:|
| Document Name | 510/510 | 100.0% |
| Parties | 509/510 | 99.8% |
| Agreement Date | 470/510 | 92.2% |
| Governing Law | 437/510 | 85.7% |
| Expiration Date | 413/510 | 81.0% |
| Effective Date | 390/510 | 76.5% |
| Anti-Assignment | 374/510 | 73.3% |

### Rarest overall

| Category | Positive contracts | Rate |
|---|---:|---:|
| Source Code Escrow | 13/510 | 2.5% |
| Price Restrictions | 15/510 | 2.9% |
| Unlimited/All-You-Can-Eat-License | 17/510 | 3.3% |
| Affiliate License-Licensor | 23/510 | 4.5% |
| Most Favored Nation | 28/510 | 5.5% |
| Third Party Beneficiary | 32/510 | 6.3% |
| No-Solicit Of Customers | 34/510 | 6.7% |

A concrete example is **Source Code Escrow**, which appears in only **13 of 510 contracts**.

A classifier could therefore achieve very high raw accuracy by almost always predicting that rare categories are absent. This is why **precision, recall, and F1 are more useful than accuracy** for this project.

I did **not** oversample, undersample, or otherwise alter class balance. That should be handled later during model training, not during cleaning, and the official test distribution should remain untouched.

### Test-set issue

The following category has **zero positive examples in the official test set**:

- Price Restrictions

That means positive-class recall/F1 for that category cannot be meaningfully evaluated using this test set alone.

Full counts are in `results/class_imbalance.csv`.

---

## 5. Common-word analysis

I used **training data only** for this exploration.

I produced:
- `common_words_train_filtered.csv` — frequent nontrivial words across cleaned training contracts
- `category_top_terms_train.csv` — common terms within positive annotated spans for each category

Global contract words are dominated by generic legal language, so **category-specific terms are more useful** than raw global frequency.

Examples of the kinds of patterns that appear:

- Termination → termination, terminate, notice
- Assignment → assignment, assign, consent
- License → license, rights, use, sublicense
- Renewal → renewal, term, notice
- Source Code Escrow → source, code, escrow

These are descriptive findings only; I did not use them to modify the dataset.

---

## 6. Risk-labeling takeaway

CUAD supplies **clause-category labels**, not Low/Medium/High risk labels.

So I would keep the tasks separate:

**contract text → CUAD clause category → risk assessment using wording/context**

Some categories such as `Document Name`, `Parties`, or `Agreement Date` may make more sense as **N/A/context-only** rather than forcing them into Low/Medium/High.

---

## 7. Questions for the Accenture advisor

1. Whose perspective should Low/Medium/High risk represent?
2. Should every CUAD category receive a risk label, or should metadata categories be N/A?
3. Should risk be based on category, actual clause wording, or both?
4. What factors does Accenture normally use to distinguish Low, Medium, and High?
5. How should ambiguous clauses be handled when commercial context is missing?
6. When one category has multiple annotated spans, should risk be scored per span or once for the combined context?
7. How should categories with very few or zero positive examples in the official test set be evaluated?

---

## Files in this folder

### `data/`
Cleaned/aggregated datasets.

### `results/`
- `class_imbalance.csv`
- `class_imbalance_train.png`
- `train_test_prevalence.png`
- `common_words_train_filtered.csv`
- `category_top_terms_train.csv`

### `WEEK2_SUMMARY.md`
This file — the complete explanation of what I did and what I found.
