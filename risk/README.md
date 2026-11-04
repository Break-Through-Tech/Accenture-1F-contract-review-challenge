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

**Buckets.** High = top 20% of scores, Medium = next 30%, Low = the rest. The cutoffs are percentiles of the scores of the 408 **training** contracts, computed from CUAD annotations. So they're fixed by the rules, not by the model or the test set.

**Evidence rules (model source only).** The clause error analysis in [`RESULTS.md`](../training/results/distilroberta_baseline/RESULTS.md) motivated category-specific evidence checks for Change Of Control, Termination For Convenience, Notice Period To Terminate Renewal, Post-Termination Services, and Uncapped Liability. A rule searches every chunk in the contract, not just the classifier's highest-scoring chunk. A match retains the model confidence and records the matching chunk; an unmatched model detection is downweighted to 25% confidence. Change Of Control additionally has a 0.5-confidence backstop, but only when the same chunk contains both a control/acquisition signal and a contractual consequence.

Each clause-register row records `source`, `rule_reason`, and `evidence`, so a reviewer can see whether the model was supported, downweighted, or backstopped. These are conservative model-side evidence checks, not legal determinations. They do not alter CUAD annotation/gold scoring, category tiers, weights, severity floor, bucket cutoffs, missing-protection rules, or combinations. Anti-Assignment remains unchanged; its contract-level F1 is 0.939 and sampled apparent errors include genuine restrictions and possible annotation gaps.

## Run

```bash
# CUAD annotations as input: a perfect-classifier ceiling, needs no trained model
python risk/score_contracts.py --source gold

# Model predictions (after train_distilroberta.py and evaluate.py)
python risk/score_contracts.py --source model --run training/outputs/sqrt_weighted
```

Model mode reads `predictions_test.npz` from the run, the contract-level thresholds from `training/outputs/evaluation/thresholds_<run>.csv` (`--threshold-scheme 0.5` ignores them), and `training/data/test_chunks.csv` for evidence text.

Outputs go to `risk/outputs/<source>/` (gitignored):

| File | Contents |
|---|---|
| `contract_scores.csv` | Rank, bucket, score, High/Medium clause counts and every reason with its points |
| `clause_register.csv` | One row per detected clause: category, risk level, confidence, points, source, named `rule_reason`, and evidence text |
| `sensitivity.csv` | How many contracts change bucket, and rank correlation with the default, when cutoffs move ±5 points or level weights change |
| `model_vs_gold.csv` | (model) Rank correlation and bucket agreement with the gold ranking under the same rules |
| `evidence_rules_check.csv` | (model) Contract-level precision/recall of the Change Of Control evidence backstop against annotations |

## Previous Baseline Results

These are the initial test-set results before the evidence-rule refinements above. They are retained as a historical baseline; rerun model scoring to evaluate the revised rules.

| | Gold | Model |
|---|---:|---:|
| High / Medium / Low | 18 / 46 / 38 | 15 / 31 / 56 |
| Rank correlation with gold | — | **0.87** |
| Bucket agreement with gold | — | 73% |
| Gold-High contracts also High | — | 14 / 18 (1 flagged Low) |

- **Rankings are stable.** Moving a cutoff by 5 points or changing a level weight moves at most 10% of contracts between buckets, and rank correlation with the default stays ≥ 0.98.
- **The model under-ranks Medium contracts.** 20 of 46 gold-Medium contracts land in Low. Model confidences sit around 0.7–0.9 instead of 1, so scores shrink against cutoffs set on annotations, and the 0.8 severity floor rarely fires (26 contracts vs 63 in gold).

## Limitations and open questions

- **The "four signals" in the project brief aren't defined in the repo.** The four here are a reasonable reading. Confirm them with the advisor.
- **The weights are a first guess.** Calibrating them needs the advisor's hand-ranked contracts. Then compare with Spearman correlation and bucket agreement, as `model_vs_gold.csv` does against annotations.
- **Party perspective is ignored.** A Non-Compete is a risk for the restricted party and a benefit to the other. The score assumes you're reviewing for the party taking on the obligations.
- **The severity floor is broad.** 62% of test contracts contain at least one High-risk clause (Exclusivity alone: 33 of 102), so in gold mode the floor decides most Medium buckets.
- The notebook's initial category tiers are aligned to the current rubric; `evaluation/risk_categories.csv` remains the source of truth.
- **A clause's presence isn't its severity.** A cap of $10M and a cap of $100 both count as "Cap On Liability". Reading amounts, durations and scope is a possible next step.
