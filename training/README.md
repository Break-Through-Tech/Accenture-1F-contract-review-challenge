# Task #2: DistilRoBERTa Multi-Label Clause Classification

Milestone #2 (October), Task #2: finalize chunking, prepare the tokenized dataset, and set up a fine-tuning baseline.

## Pipeline

1. **`prepare_dataset.py`**: finalizes chunking. It reruns the Week 3 paragraph-aware chunker (`paragraph_tok_480_overlap_64`) with the **`distilroberta-base` tokenizer**, replacing the provisional `bert-base-uncased` one. It then labels each chunk using the Week 3 rule:
   - category fully contained in the chunk → positive
   - partial-only overlap → masked (ignored, not treated as a negative)
   - otherwise → negative

   It saves `train/test_chunks.csv`, `train/test_tokenized.pt`, `labels.json`, `dataset_summary.csv` and `chunking_summary.csv` to `training/data/`.

   Safeguards: the script stops if any contract's chunks leave a gap (contract text dropped), and it reports annotated-span coverage per split. Expected output:

   | Split | Contracts | Chunks | Span coverage |
   |---|---:|---:|---:|
   | train | 408 | 14,015 | 98.62% |
   | test | 102 | 2,906 | 99.21% |

   If you generated `training/data/` before the chunker token-count fix (October 27), regenerate it and retrain. The old data dropped about 1% of contract text and about 4–5% of annotated clauses. See [`Week 3/CHUNKER_TOKEN_COUNT_FIX.md`](../Week%203/CHUNKER_TOKEN_COUNT_FIX.md).
2. **`train_distilroberta.py`**: fine-tunes `distilroberta-base` with one sigmoid head per category and a masked BCE loss.
   - 10% of the official train contracts are held out (split by contract) for model selection.
   - The model is evaluated on the official CUAD test contracts.
   - Outputs go to `training/outputs/`: `best_model/`, `history.csv`, `chunk_level_metrics.csv`, `contract_level_metrics.csv`.
   - Contract-level metrics use the max probability over each contract's chunks, so they are comparable with the TF-IDF baseline in `notebooks/naeem_tfidf_baseline_risk_rules.ipynb`.

Generated data and model outputs are gitignored. Regenerate them with the commands below.

## Run in Google Colab

Open `notebooks/naeem_distilroberta_finetune_colab.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**, and run all cells.

## Run locally

```bash
pip install -r training/requirements.txt
python training/prepare_dataset.py
python training/train_distilroberta.py --epochs 3
# quick CPU smoke test:
python training/train_distilroberta.py --epochs 1 --limit 64
```

Useful flags: `--lr`, `--batch-size`, `--threshold`, `--use-pos-weight` (per-category positive weighting, a starting point for Task #3 imbalance handling).
