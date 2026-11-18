# Change Of Control risk calibration

## Experiment setup

Used the existing sqrt-weighted DistilRoBERTa bundle registered as
`sqrt_weighted_run1`. The [run manifest](../runs/sqrt_weighted_run1/manifest.json)
points to `training/outputs/sqrt_weighted/` and its tuned thresholds CSV.
The bundle's `RUN_INFO.md` records training at commit `47083b5`. The saved
validation predictions aligned by chunk index and title to 1,341 rows in
`training/data/train_chunks.csv` across 40 contracts. The test predictions
aligned to all 2,906 rows in the regenerated `training/data/test_chunks.csv`
across 102 contracts. The experiment reused the saved predictions and
contract-level thresholds.

The grid crossed five backstop confidences (0.3, 0.4, 0.5, 0.6, 0.7) with
three missing-evidence multipliers (0.1, 0.25, 0.5). Scoring used the existing
CUAD annotations as gold and the pipeline's fixed percentile procedure on the
408 labeled contracts in the original training pool. Rank correlation compares
model and gold score ranks. Bucket agreement compares the assigned buckets
contract by contract. Change Of Control precision and recall count detections
from the model and keyword backstop.

## Validation results and selection

All 15 rows are in the [calibration grid](risk_calibration_grid.csv).
Candidates had to preserve baseline rank correlation and avoid additional
Gold-High contracts being scored Low. Among eligible candidates, selection
maximized the mean of rank correlation and bucket agreement at the precision
reported in the grid. Ties favored fewer Gold-High-to-Low errors, more Gold-High
contracts scored High, higher Change Of Control precision and recall, fewer
bucket changes, then lower backstop confidence and lower missing-evidence
multiplier, in that order.

| Validation metric | Current 0.5 / 0.25 | Selected 0.6 / 0.1 |
| --- | ---: | ---: |
| Rank correlation | 0.917 | 0.920 |
| Bucket agreement | 62.5% | 65.0% |
| High / Medium / Low | 7 / 13 / 20 | 7 / 14 / 19 |
| Gold-High predicted High | 7/11 | 7/11 |
| Gold-High predicted Low | 0 | 0 |
| Change Of Control precision | 0.600 | 0.600 |
| Change Of Control recall | 0.900 | 0.900 |
| Contracts changing bucket | 0 | 1 |

Across the grid, rank correlation ranged from 0.915 to 0.920. Every
configuration with confidence 0.6 or 0.7 moved one contract from Low to Medium
and achieved 65.0% agreement. All other configurations kept the baseline
buckets. Settings 0.6 / 0.1, 0.7 / 0.1, and 0.7 / 0.25 tied at the highest
selection score of 0.785. The lower backstop confidence broke the tie in favor
of **0.6 / 0.1**.

All 15 configurations had the same Change Of Control precision and recall.
The grid changes score contributions for clauses whose detection status is
already determined by the tuned threshold and evidence pattern. The selection
therefore improved validation ranking and bucket agreement with one bucket
change and zero Gold-High-to-Low errors.

To reproduce the validation selection separately:

```sh
python -m risk.calibrate_risk --validation-only --out /tmp/risk-calibration-validation.csv
```

## Held-out test

Settings **0.6 / 0.1** were selected on validation before test scoring. The
table uses the single selected test evaluation already recorded in the grid
and the [committed run 1 baseline](../results/end_to_end_run1/RESULTS.md).
The independent validation rerun preserved the selection, so the recorded
test result was retained.

| Test metric | Committed baseline | Selected | Difference |
| --- | ---: | ---: | ---: |
| Rank correlation | 0.875 | 0.881 | +0.006 |
| Bucket agreement | 72.5% | 74.5% | +2.0 percentage points |
| High / Medium / Low | 16 / 31 / 55 | 17 / 34 / 51 | +1 / +3 / −4 |
| Gold-High predicted High | 12/18 | 13/18 | +1 contract |
| Gold-High predicted Low | 1 | 0 | −1 contract |
| Change Of Control precision | 0.479 | 0.489 | +0.010 |
| Change Of Control recall | 0.885 | 0.885 | 0.000 |

Nine test contracts changed bucket versus the committed baseline.

## Final settings

[Risk rules](../risk/risk_rules.json) use `backstop_confidence = 0.6` and
`multiplier_if_missing = 0.1` for Change Of Control. The High, Medium, and
Low weights, bucket percentile cutoffs, severity floor, category tiers,
evidence patterns, and unrelated evidence rules retain their committed values.
Missing-protection and combination scoring also retain their committed values.

## Limitations

- Validation had 40 contracts and 11 Gold-High cases. One contract accounts
  for the full validation bucket-agreement gain. Rounded rank correlations
  leave three candidates tied at the reported precision.
- The fixed percentile reference pool includes the validation contracts' gold
  scores. Validation agreement is conditional on that pipeline convention.
- The committed baseline comes from a separate training run. The test
  differences reflect both run and scoring differences. In particular, the
  observed precision increase cannot be attributed to these two settings,
  which leave detection membership fixed on a given set of predictions.
- This is one CUAD split. Performance on other contract populations remains
  unmeasured.
