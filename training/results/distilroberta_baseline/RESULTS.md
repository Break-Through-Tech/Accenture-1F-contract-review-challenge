# DistilRoBERTa Baseline: Results (October 28, 2026)

First full training of the DistilRoBERTa clause classifier on the corrected chunk dataset (after the chunker fix in #17), with class-imbalance handling, tuned thresholds, a TF-IDF comparison and error analysis. Covers issues #10, #8, #7 and #6.

## TL;DR

- **Recommended model: `sqrt_weighted` with per-category thresholds tuned on validation.** Contract-level macro F1 is **0.71 on the 7 TF-IDF categories** (TF-IDF: 0.59), **0.69 on the 19 risk-rubric categories** and 0.60 on all 40 evaluable categories.
- **Class weighting is essential.** Without it, the model predicts almost nothing for rare categories at 0.5 (contract macro F1 0.16). Capped square-root weighting lifts that to 0.59 before any threshold tuning.
- **It beats TF-IDF on 5 of 7 categories**, with big gains on Insurance (0.92 vs 0.62), Renewal Term (0.72 vs 0.41) and Notice Period to Terminate Renewal (0.63 vs 0.42). It trails on **Change Of Control** (0.37 vs 0.55) and is level on **Termination For Convenience** (0.55 vs 0.58).
- **Weak spots:** the rarest categories (3 or fewer test contracts) score 0, and Change Of Control and Covenant Not To Sue are often missed when worded unusually.

## Setup

| | |
|---|---|
| Data | `training/prepare_dataset.py` (DistilRoBERTa tokenizer, 480 tokens / 64 overlap, 41 categories) |
| Split | official 408 train / 102 test contracts; 40 train contracts held out for validation |
| Chunks | 12,674 train / 1,341 val / 2,906 test |
| Model | `distilroberta-base`, 41 sigmoid outputs, masked BCE |
| Training | 3 epochs, lr 3e-5, batch 16, warmup 10%, best epoch by validation macro F1 |
| Hardware | Apple M4 Pro (MPS), ~10 min/epoch |

Runs:

- **`baseline`:** no class weighting.
- **`sqrt_weighted`:** `--pos-weight-mode sqrt`. Each category's positives are weighted by √(neg/pos), capped at 30. Weights range from 4.9 to 28; raw neg/pos would reach ~600.

Reproduce:

```bash
python training/prepare_dataset.py
python training/train_distilroberta.py --epochs 3 --out training/outputs/baseline
python training/train_distilroberta.py --epochs 3 --pos-weight-mode sqrt --out training/outputs/sqrt_weighted
python training/evaluate.py --runs training/outputs/baseline training/outputs/sqrt_weighted
```

## Results

### Summary (test set)

Macro F1 excludes Price Restrictions, which has no positive test examples.

| Run | Thresholds | Level | Micro F1 | Macro F1 (40) | Macro F1 (19 risk) | Macro F1 (7 TF-IDF) |
|---|---|---|---:|---:|---:|---:|
| baseline | 0.5 | contract | 0.529 | 0.164 | 0.111 | 0.084 |
| baseline | tuned | contract | 0.669 | 0.461 | 0.550 | 0.618 |
| sqrt_weighted | 0.5 | contract | 0.755 | 0.589 | 0.685 | 0.669 |
| **sqrt_weighted** | **tuned** | **contract** | **0.752** | **0.596** | **0.694** | **0.706** |
| sqrt_weighted | 0.5 | chunk | 0.527 | 0.417 | 0.478 | 0.507 |
| sqrt_weighted | tuned | chunk | 0.458 | 0.445 | 0.530 | 0.572 |
| TF-IDF + LR | — | contract | — | — | — | 0.592 |

Contract level answers "does this contract contain the clause?", which is what triage needs. Chunk level answers "is the clause in this chunk?" and is harder: one contract has about 28 chunks, and nearby chunks often discuss related topics.

### Versus TF-IDF (contract-level F1)

| Category | TF-IDF | sqrt_weighted (0.5) | sqrt_weighted (tuned) |
|---|---:|---:|---:|
| Anti-Assignment | 0.865 | 0.939 | **0.939** |
| Audit Rights | 0.706 | 0.790 | **0.821** |
| Insurance | 0.618 | 0.870 | **0.915** |
| Renewal Term | 0.407 | 0.591 | **0.719** |
| Notice Period To Terminate Renewal | 0.424 | 0.429 | **0.634** |
| Termination For Convenience | **0.576** | 0.557 | 0.545 |
| Change Of Control | **0.545** | 0.508 | 0.368 |

### By category (sqrt_weighted, tuned, contract level)

- **Strong (F1 ≥ 0.9), 7 categories:** Parties, Document Name, Governing Law, Anti-Assignment, Agreement Date, License Grant, Insurance.
- **Good (0.7–0.9), 7 categories:** Expiration Date, Cap On Liability, Audit Rights, Effective Date, Revenue/Profit Sharing, Minimum Commitment, Renewal Term.
- **Moderate (0.4–0.7), 16 categories:** including Liquidated Damages, Exclusivity, IP Ownership Assignment, Uncapped Liability, Non-Compete, Covenant Not To Sue (0.61), Termination For Convenience (0.55).
- **Weak (< 0.4), 10 categories:** Change Of Control (0.37), No-Solicit Of Customers, Joint IP Ownership, Affiliate License-Licensee/Licensor, Non-Disparagement, Third Party Beneficiary, plus Source Code Escrow, Most Favored Nation and Unlimited/All-You-Can-Eat License, which score 0 with 3 or fewer test contracts each.

Of the 19 risk-rubric categories, Change Of Control is the only one in the weak tier, and it's rated **High** risk in the rubric. It's the top priority for improvement.

The full table is in `per_category_sqrt_weighted.csv`.

## Error Analysis

From `error_analysis_sqrt_weighted.csv` (top 5 false positives and false negatives per category, with chunk text):

1. **Confusion from shared wording.** Covenant Not To Sue false positives are mostly *"covenant not to compete"* / non-compete sections. The model keys on "covenant … not to" rather than "sue". Change Of Control false positives are anti-assignment clauses that mention a party *"acquiring all or substantially all of such party's assets"*.
2. **Unusual wording is missed.** Change Of Control false negatives include a "Future Acquisitions" pricing clause and acquisition-triggered milestone payments, where the term "change of control" never appears. Termination For Convenience misses short, informal clauses ("can be terminated at any time at the discretion of either party").
3. **Termination over-triggering.** Termination For Convenience false positives are other termination sections (for death or disability, breach, expiration). The model learned "termination section" rather than "termination *without cause*".
4. **Possible annotation gaps.** One Change Of Control "false positive" is a heading, *"11.3 Termination Upon Certain Changes of Control. [\*\*\*]"*, whose body is redacted, so CUAD has nothing to label. Some false positives may be correct and are worth a manual check before over-tuning.
5. **Threshold instability on rare categories.** Third Party Beneficiary has exactly 5 validation positives (the minimum for tuning), and its tuned chunk threshold fell to 0.05, producing 835 false positive chunks. Treat its tuned chunk threshold as unreliable.

## Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| Imbalance handling | √(neg/pos) positive weights, capped at 30 | Raw weights (~600) destabilize training; no weighting collapses rare categories |
| Checkpoint selection | best validation **macro** F1 | Micro F1 is dominated by easy, frequent categories |
| Thresholds | tuned per category **and per level** on validation; fall back to 0.5 below 5 validation positives | A flat 0.5 is badly miscalibrated under weighting/imbalance; contract scores (max over chunks) need their own thresholds |
| Macro averages | exclude categories with no test positives | Price Restrictions has none in the official test split |
| Category scope | model detects all 41; the **risk score uses the 19 rubric categories**; the identification categories (Document Name, Parties, the dates, Governing Law) feed the clause register as metadata; the rest are reported without risk weight until the advisor assigns levels | Keeps triage tied to the reviewed rubric without discarding detections |

## Limitations

- **Small validation set (40 contracts).** Thresholds for categories with 5–15 validation positives are noisy; Change Of Control is the clearest case (0.5 beats tuned on test).
- **One seed per configuration.** Differences of a few F1 points between runs are within noise.
- **The rarest categories can't be learned or judged reliably:** 3 or fewer test contracts, 20–30 training chunks.
- **Not compared with the CUAD paper's span-level AUPR,** since the task here is chunk/contract classification.

## Recommended Next Steps

1. Adopt `sqrt_weighted` + tuned contract thresholds as the baseline for the risk layer.
2. Try more epochs: validation macro F1 was still rising at epoch 3 (0.26 → 0.40 → 0.41).
3. Use 5-fold contract-level cross-validation on train to choose thresholds more robustly than 40 validation contracts allow.
4. For Change Of Control, Covenant Not To Sue and Termination For Convenience, review the error rows with the advisor. Consider keyword features or a rule-based backstop for the risk layer.
5. Try a larger or legal-domain encoder (e.g. `roberta-base`, `nlpaueb/legal-bert-base-uncased`) on Colab now that the pipeline is validated.
