# Contract Risk Scoring

Turns clause detections into a **clause register** and a **Low / Medium / High triage bucket** per contract. It's rule-based on purpose: every point of a contract's score comes from a named reason, so a reviewer can see why a contract ranks where it does.

## How a contract is scored

Category risk levels come from the rubric, [`evaluation/risk_categories.csv`](../evaluation/risk_categories.csv) (19 categories, High or Medium). All other weights live in [`risk_rules.json`](risk_rules.json), so they can be changed without touching code.

| # | Signal | Rule (defaults) |
|---|---|---|
| 1 | **Clauses present** | Each detected rubric clause adds *level weight × confidence*: High = 3, Medium = 1. Categories outside the rubric are listed in the register but add nothing. |
| 2 | **Most severe clause** | A High-risk clause detected with confidence ≥ 0.8 puts the contract in at least the Medium bucket. |
| 3 | **Missing protections** | No Cap On Liability: +2. Uncapped Liability with no Cap On Liability: another +2. |
| 4 | **Risky combinations** | +1 each, scaled by the lower confidence: Exclusivity + Minimum Commitment, Renewal Term + Notice Period To Terminate Renewal, Non-Compete + Exclusivity. |

**Buckets.** High starts at the 80th percentile and Medium at the 50th percentile of scores from the 408 training contracts, using CUAD annotations. These thresholds apply to new contracts and the severity floor can raise the assigned bucket.

**Evidence rules (model source only).** The clause error analysis in [`RESULTS.md`](../training/results/distilroberta_baseline/RESULTS.md) motivated category-specific evidence checks for Change Of Control, Termination For Convenience, Notice Period To Terminate Renewal, Post-Termination Services, and Uncapped Liability. A rule searches every chunk in the contract, not just the classifier's highest-scoring chunk. A match retains the model confidence and records the matching chunk; an unmatched model detection keeps 25% of its confidence for those rules and 10% for Change Of Control. Change Of Control additionally has a fixed backstop score contribution based on confidence 0.6, but only when the same chunk contains both a control/acquisition signal and a contractual consequence.

Each clause-register row records `source`, `rule_reason`, and `evidence`, so a reviewer can see whether the model was supported, downweighted, or backstopped. These are conservative model-side evidence checks, not legal determinations. They do not alter CUAD annotation/gold scoring, category tiers, weights, severity floor, bucket cutoffs, missing-protection rules, or combinations. Anti-Assignment remains unchanged; its contract-level F1 is 0.939 and sampled apparent errors include genuine restrictions and possible annotation gaps.

## Run

```bash
# CUAD annotations as input: a perfect-classifier ceiling, needs no trained model
python risk/score_contracts.py --source gold

# Model predictions (after train_distilroberta.py and evaluate.py)
python run_pipeline.py score --run-name sqrt_weighted_run1
```

Batch model scoring uses test predictions and contract thresholds from the
registered run, plus `training/data/test_chunks.csv` for evidence text.
Outputs go to `risk/outputs/sqrt_weighted_run1/model/`.

| File | Contents |
|---|---|
| `contract_scores.csv` | Rank, bucket, score, High/Medium clause counts and every reason with its points |
| `clause_register.csv` | One row per detected clause: category, risk level, confidence, points, source, named `rule_reason`, and evidence text |
| `sensitivity.csv` | How many contracts change bucket, and rank correlation with the default, when cutoffs move ±5 points or level weights change |
| `model_vs_gold.csv` | (model) Rank correlation and bucket agreement with the gold ranking under the same rules |
| `evidence_rules_check.csv` | (model) Contract-level precision/recall of the Change Of Control evidence backstop against annotations |

## Results and interpretation

The [calibration report](../evaluation/risk_calibration.md) contains the current
test results and committed baseline comparison. The CUAD-derived reference
applies the same scoring rules to annotated clause presence. Independent
advisor review is needed to assess practical triage quality.

- Confidence values are model outputs or rule-assigned scoring inputs. They
  should be read alongside the evidence and rule source.
- Missing-protection points reflect a missing detection. A missed liability
  cap can add points even when the contract contains one.
- The rubric scores clause presence and selected combinations. The affected
  party, dollar amounts, duration, and scope still require reviewer judgment.
- The severity floor raises the bucket when a High-risk detection reaches
  its confidence threshold. The contract reasons and floor flag explain how
  the resulting bucket was assigned.
