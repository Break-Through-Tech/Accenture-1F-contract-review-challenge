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

The exported CSV contains the **top five chunk-level false positives and false negatives per category**, selected using tuned chunk thresholds. Its `n_errors_in_category` is the count of chunk errors, not contract errors. Risk scoring instead uses contract-level thresholds and max probability over chunks, so sampled rows diagnose language and localization patterns; they do not establish contract-level scoring precision. A high-confidence “false positive” chunk can coexist with a correctly labeled contract, and an annotated contract can have the model's best chunk somewhere unrelated to the clause.

### Risk-Rubric Categories

Contract-level tuned F1 is reported below. Pattern descriptions refer to sampled chunk errors; where the sample is weak or ambiguous, the scoring rule is intentionally left alone. No category tier or level weight is changed by this analysis.

| Category (F1) | False-positive / false-negative pattern and likely cause | Risk-rule decision |
|---|---|---|
| Cap On Liability (0.875) | FP chunks include broad indemnity/liability context; FN chunks can separate the cap or exception from the related obligation. Likely section overlap and chunk boundaries, not evidence that the category definition is wrong. | Stay unchanged; a cap's presence alone does not establish its adequacy. |
| Uncapped Liability (0.600) | FP example overlaps a Cap On Liability clause; broad indemnity language can be mistaken for unlimited exposure. Few FNs (3 sampled error count) limit conclusions. | Refine model evidence qualification to require express uncapped/unlimited language tied to liability, damages, or indemnity. |
| Insurance (0.915) | FP sample is a generic section heading; an FN is a high-level insurance-contract passage where the best chunk is unrelated to coverage requirements. Mostly heading/context and evidence-localization noise. | Stay unchanged. |
| Liquidated Damages (0.667) | FP overlaps a liability-cap table; FNs include formula- or schedule-based consequences and post-termination amounts. Likely financial-remedy overlap and fragmented evidence. | Stay unchanged; amounts and triggers need clause-level review. |
| Warranty Duration (0.600) | FP mentions initial term/continuations rather than a warranty period; FNs include short duration wording separated from the warranty. Likely term/warranty vocabulary overlap and localization. | Stay unchanged. |
| Anti-Assignment (0.939) | Some FP rows expressly restrict assignment or transfers but allow affiliate/asset-sale exceptions; some may be annotation gaps. FN rows include genuine consent restrictions in contracts whose highest-probability chunk is unrelated. | Stay unchanged; strongest rubric-category result, and sampled “errors” do not justify an automatic assignment penalty adjustment. |
| Change Of Control (0.368) | FP rows often describe an asset-sale exception in an assignment clause, not a control trigger with consequences; a redacted “termination upon changes of control” heading is annotation-ambiguous. FNs include acquisition/control provisions involving repayment, service limits, or notice, sometimes with unusual phrasing. | Refine: require a control/ownership event plus a contractual consequence, search all chunks, and retain matched evidence; allow a conservative named backstop. |
| Renewal Term (0.719) | FP example is termination/expiration context; FNs include explicit renewal terms outside the selected chunk. Likely term/expiration overlap and chunk placement. | Stay unchanged; existing renewal-plus-notice combination already captures the most actionable interaction. |
| Notice Period To Terminate Renewal (0.634) | FP examples can include generic notice or term/renewal passages; FNs can be in a separate renewal-notice section. Likely “notice” and “renewal” co-occurrence without a linked opt-out deadline, plus localization. | Refine model evidence qualification to link notice/deadline language to ending or declining renewal. |
| Termination For Convenience (0.545) | FPs are breach, death/disability, expiration, or other termination grounds; FNs include short “at any time / discretion” language. The model overlearns termination-section vocabulary and misses paraphrases. | Refine model evidence qualification to require without-cause, convenience, any-reason, or discretionary termination language. |
| Post-Termination Services (0.596) | Many FP chunks mention termination near ordinary survival/liquidated-damages language; FNs include continued supply or transition support. Likely broad termination/survival overlap and the distinction between surviving duties and actual services. | Refine model evidence qualification to require a service/support/performance obligation continuing after termination or expiry. |
| License Grant (0.928) | Occasional FP is general IP/license context; FNs often occur in franchise or detailed grant text away from the top chunk. Strong contract-level precision/recall suggests chunk evidence is the main issue. | Stay unchanged. |
| IP Ownership Assignment (0.667) | FP chunks overlap Anti-Assignment (assignment of a contract versus ownership of IP); FNs include substantive “ownership of materials” provisions. Likely overloaded “assignment/ownership” wording. | Clarify the distinction in the rubric; no separate scoring change. |
| Non-Transferable License (0.588) | FPs share broad license/IP and exclusivity wording; FNs may describe limits on sublicensing or transfers without the category phrase. Likely overlap with License Grant and Anti-Assignment. | Clarify in the rubric; no scoring change without contract-level evidence. |
| Non-Compete (0.590) | FPs overlap no-solicit, confidentiality, or other restrictive-covenant text; FNs are relatively few but may use narrow or indirect competition restrictions. Likely related-covenant vocabulary and boundaries. | Stay unchanged; the non-compete/exclusivity combination remains a review signal, not a legal conclusion. |
| Exclusivity (0.667) | FP samples include license-grant and other commercial restrictions; FNs may express exclusivity through “sole” rights or territory/business terms. Likely semantic overlap and less canonical phrasing. | Stay unchanged; review context and party perspective before changing the High tier. |
| Minimum Commitment (0.730) | FP snippets can be generic performance/expense language; FNs include quota or numeric schedule terms. Likely obligation-versus-minimum distinction and tables/chunking. | Stay unchanged; the existing combination with Exclusivity is a useful context flag. |
| Revenue/Profit Sharing (0.737) | FPs overlap minimum commitments, royalties, or payment terms; FNs include percentage/royalty schedules and indirect revenue language. Likely commercial-payment overlap. | Stay unchanged. |
| Audit Rights (0.821) | Few FPs; sampled FNs include audit-fee language or clauses whose audit right is elsewhere in the contract. Likely evidence localization, with no strong false-positive theme. | Stay unchanged. |

### Broader Error Signals

- **Overlap beyond the rubric:** Covenant Not To Sue false positives often contain “covenant not to compete”; Joint IP Ownership confuses general license/IP text; Non-Compete, No-Solicit, and Non-Disparagement share restrictive-covenant language. These categories are not all risk-scored, so the classifier errors do not justify changing rubric weights.
- **Rare categories and thresholds:** Third Party Beneficiary's chunk threshold of 0.05 produces 835 chunk false positives; the category also has very poor contract-level precision. Most Favored Nation (3 test positives), Source Code Escrow (2), and Unlimited/All-You-Can-Eat License (3) have too little support for conclusions. Do not use these threshold/error samples to add risk rules.
- **Evidence placement matters:** Some annotated Anti-Assignment, Change Of Control, renewal, and IP clauses appear in contracts where the reported top-scoring chunk is unrelated. Search all contract chunks for qualifying evidence before using a clause detection to alter confidence or risk points.

### Decisions From This Review

- The model-only evidence rules now cover Change Of Control, Termination For Convenience, Notice Period To Terminate Renewal, Post-Termination Services, and Uncapped Liability. They qualify model detections only when clause-specific wording is found somewhere in the contract's chunks; otherwise they downweight confidence. Change Of Control also has a conservative backstop requiring both a control/acquisition signal and a contractual consequence.
- Every match, backstop, or downweight is named in the clause register, and the qualifying chunk is retained. Gold/annotation scoring, rubric category levels, weights, severity floor, bucket cutoffs, missing-protection rules, and combinations are unchanged.
- Anti-Assignment stays unchanged. Its strong contract-level F1 and sampled genuine restrictions/possible annotation gaps argue against lowering its score or adding a lexical gate.
- The rubric now distinguishes assignment of an agreement from ownership changes, generic termination from termination without cause, renewal notice from generic notice, and surviving duties from post-termination services.

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

1. When model prediction outputs and chunk data are available, rerun model-level risk scoring with the current evidence rules (for example, `python risk/score_contracts.py --source model --run training/outputs/sqrt_weighted`).
2. Compare the revised rankings and risk buckets with the previous baseline, including rank correlation, bucket agreement and migrations, and the Change Of Control evidence-backstop precision/recall.
3. As separate future model work, consider contract-level cross-validation for threshold selection and experiments with more epochs or a legal-domain encoder.
4. Advisor or domain-expert review of the revised examples and risk rankings is optional future validation; it is not a prerequisite for completing this error-analysis and rule-refinement task.
