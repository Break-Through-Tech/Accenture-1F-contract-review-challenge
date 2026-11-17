# End-to-End Pipeline Run 1: Results

## Command

```bash
python run_pipeline.py all
```

## Environment

- Python 3.11, torch 2.15.0, transformers 5.22.2
- Device: mps (Apple GPU)
- Seed: 42
- Model: distilroberta-base, 3 epochs

## Model results (test set, contract level)

- Macro F1 on the 19 risk categories: **0.702** with tuned thresholds (0.683 at the default 0.5 threshold).
- On the 7 categories the TF-IDF baseline also covers: **0.713** for this model vs **0.592** for TF-IDF.
- Macro F1 over all categories: **0.601**.
- Strongest categories (of the 19 risk categories): Insurance 0.94, Anti-Assignment 0.94, License Grant 0.94, Cap On Liability 0.88.
- Weakest categories: Change Of Control 0.36, Uncapped Liability 0.51, Termination For Convenience 0.53. Change Of Control and Termination For Convenience are also below the TF-IDF baseline on those same categories.

Source: `per_category_sqrt_weighted_run1.csv` and `run_summary.csv` in this folder.

## Comparison to the team's earlier baseline

The team's `training/results/distilroberta_baseline/RESULTS.md` reports 0.694 (19 risk categories) and 0.596 (all categories) for the same model/threshold configuration. This run reproduces those numbers within about 0.01.

## Risk scoring (102 test contracts)

Bucket counts: 16 High, 31 Medium, 55 Low.

Versus the gold risk ranking:
- Rank correlation: 0.875
- Bucket agreement: 72.5%
- 12 of 18 gold-High contracts are also scored High by the model

Source: `contract_scores.csv` and `model_vs_gold.csv` in this folder.

### Sensitivity

Changing the High or Medium risk weights moves at most 6 contracts between buckets, and rank correlation versus the default weighting stays at or above 0.98. (`sensitivity.csv`)

### Change Of Control evidence backstop

The keyword backstop for Change Of Control raises recall from 0.27 (model only) to 0.89, with precision 0.48. (`evidence_rules_check.csv`)

## Notes

- This is one seed, one run. Differences of a few F1 points between runs are expected noise, not a real change.
- The comparison to the earlier baseline above is rough, since that was a different training run (not the same seed/run as this one), even though it used the same configuration.
