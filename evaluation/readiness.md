# November readiness and December handoff

## Advisor ranking validation ([#24](https://github.com/Break-Through-Tech/Accenture-1F-contract-review-challenge/issues/24))

1. Agree on the reviewing party and what action each priority bucket implies.
   Freeze the model run, rubric, and rules for the exercise.
2. Start with 12–20 reviewable contracts. Include clear cases across buckets,
   near-boundary cases, and examples where a consequential clause may have
   been missed. Record how cases were chosen and which excerpts were shown.
3. Before showing model outputs, capture the advisor's rank or tied priority,
   bucket, decisive evidence, and rationale. Afterward, review disagreements
   together using the clause register. Mark whether the disagreement came
   from detection, evidence, the rubric, or contract context.
4. Report Spearman correlation, bucket agreement, and the confusion matrix.
   Read every advisor-High case scored Low and record the consequence of
   missing it. Treat the deliberately sampled set as a review exercise.
   Use fresh examples to confirm changes prompted by it.

For the coach walkthrough, show a clear High, a boundary case, and a
disagreement. Trace a detection from chunk evidence and threshold through
points, any missing-protection or combination points, and the severity floor.

## Change Of Control ([#25](https://github.com/Break-Through-Tech/Accenture-1F-contract-review-challenge/issues/25))

A focused check on the saved 40-contract validation split held the existing
evidence pattern fixed. For each contract, take the maximum Change Of Control
chunk probability and flag the category if it reaches the candidate threshold
or any chunk matches the rule. Compare to CUAD presence on the 10 positives.

| Detection setting | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current 0.65 plus evidence | 9 | 6 | 1 | 0.600 | 0.900 |
| 0.50 plus evidence | 9 | 14 | 1 | 0.391 | 0.900 |
| 0.75 plus evidence | 8 | 6 | 2 | 0.571 | 0.800 |
| Evidence alone | 8 | 6 | 2 | 0.571 | 0.800 |

Keep 0.65 and the [calibrated scoring settings](risk_calibration.md). Each
alternative worsened precision or recall. The next useful audit is to review
the current false positives and missed positive with the advisor and assess
whether one evidence-pattern change addresses a recurring failure. A future
candidate should preserve recall and avoid additional High-to-Low changes
before adoption. This check used validation only and evaluated detection
membership. A change would also need a contract-bucket check.

## December readiness ([#26](https://github.com/Break-Through-Tech/Accenture-1F-contract-review-challenge/issues/26))

- [ ] Have a teammate follow the registered-run instructions from a clean checkout.
- [ ] Include a ranked contract queue and an evidence-backed clause register.
- [ ] Report per-category precision, recall and F1 alongside baseline results,
  with model-run and split provenance.
- [ ] Report advisor rankings, serious disagreements, and boundary sensitivity.
- [ ] Time a small paired review exercise with and without the register. Use
  comparable cases and alternate which workflow goes first. Report the sample
  size, time difference, and whether either workflow missed a high-priority issue.
- [ ] Finalize the report with limitations and a worked example from clause
  evidence through risk points and review priority.

Defer new encoders and automated financial scoring unless the advisor review
identifies a specific failure that either would address.
