#!/usr/bin/env python3
"""Tasks #7, #8 and #6: evaluate trained runs, tune thresholds, export errors.

Reads predictions_val.npz / predictions_test.npz written by train_distilroberta.py
for one or more runs and produces, in --out (default training/outputs/evaluation):

  thresholds_<run>.csv          per-category thresholds tuned on validation, one set
                                for chunk level and one for contract level
  per_category_<run>.csv        test P/R/F1 per category, chunk + contract level,
                                at 0.5 and at the tuned thresholds
  run_summary.csv               micro/macro F1 per run x threshold scheme, overall,
                                on the 19 risk-rubric categories, and on the 7 TF-IDF
                                baseline categories
  tfidf_comparison.csv          contract-level P/R/F1 vs the TF-IDF baseline
  error_analysis_<run>.csv      top false positives / false negatives per category
                                on test chunks (tuned thresholds), with chunk text

Decisions:
  - Thresholds are tuned on validation (F1-maximizing over 0.05..0.95) separately
    per level: chunk thresholds on validation chunks, contract thresholds on
    validation contracts (max over chunks). A contract's max-over-chunks score
    sits higher than a single chunk's, so chunk thresholds don't transfer.
    Categories with fewer than --min-val-positives validation positives keep 0.5,
    since a threshold fit on a handful of examples is noise.
  - Macro averages skip categories with no test positives (Price Restrictions has
    none in the official test split), so they are not counted as F1 = 0.
  - Contract-level score = max chunk probability, label = any chunk fully or
    partially contains the clause (same as train_distilroberta.py).

Usage:
    python training/evaluate.py --runs training/outputs/baseline training/outputs/sqrt_weighted
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from train_distilroberta import TFIDF_BASELINE_CATEGORIES, contract_level

REPO_ROOT = Path(__file__).resolve().parents[1]

# Contract-level test results of the TF-IDF + logistic regression baseline
# (notebooks/naeem_tfidf_baseline_risk_rules.ipynb).
TFIDF_RESULTS = {
    "Anti-Assignment": (0.807, 0.931, 0.865),
    "Change Of Control": (0.517, 0.577, 0.545),
    "Insurance": (0.583, 0.656, 0.618),
    "Renewal Term": (0.393, 0.423, 0.407),
    "Notice Period To Terminate Renewal": (0.412, 0.438, 0.424),
    "Audit Rights": (0.638, 0.789, 0.706),
    "Termination For Convenience": (0.514, 0.655, 0.576),
}

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def norm(name):
    return name.lower().replace(" ", "")


def load(run, split):
    d = np.load(Path(run) / f"predictions_{split}.npz")
    return {k: d[k] for k in d.files}


def f1_at(y_true, scores, t):
    return precision_recall_fscore_support(
        y_true, (scores >= t).astype(int), average="binary", zero_division=0
    )[2]


def tune_thresholds(val, categories, min_pos):
    rows = []
    for i, cat in enumerate(categories):
        keep = val["masks"][:, i] > 0
        y, p = val["labels"][keep, i].astype(int), val["probs"][keep, i]
        n_pos = int(y.sum())
        if n_pos < min_pos:
            rows.append({"category": cat, "val_positives": n_pos, "threshold": 0.5,
                         "val_f1_at_0.5": round(f1_at(y, p, 0.5), 3), "val_f1_tuned": None,
                         "note": "too few val positives, kept 0.5"})
            continue
        f1s = np.array([f1_at(y, p, t) for t in GRID])
        # Best F1; ties broken toward 0.5.
        best = max(range(len(GRID)), key=lambda k: (round(f1s[k], 6), -abs(GRID[k] - 0.5)))
        rows.append({"category": cat, "val_positives": n_pos, "threshold": float(GRID[best]),
                     "val_f1_at_0.5": round(f1_at(y, p, 0.5), 3),
                     "val_f1_tuned": round(float(f1s[best]), 3), "note": ""})
    return pd.DataFrame(rows)


def metrics(probs, labels, masks, categories, thresholds):
    rows = []
    for i, cat in enumerate(categories):
        keep = masks[:, i] > 0
        y = labels[keep, i].astype(int)
        pred = (probs[keep, i] >= thresholds[i]).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        rows.append({"category": cat, "support": int(y.sum()), "predicted": int(pred.sum()),
                     "precision": round(float(p), 3), "recall": round(float(r), 3),
                     "f1": round(float(f), 3)})
    return pd.DataFrame(rows)


def micro(probs, labels, masks, thresholds, cols):
    keep = masks[:, cols] > 0
    y = labels[:, cols][keep].astype(int)
    pred = (probs[:, cols] >= np.asarray(thresholds)[cols])[keep].astype(int)
    return precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[2]


def macro(df, names=None):
    df = df[df["support"] > 0]
    if names is not None:
        df = df[df["category"].map(norm).isin({norm(n) for n in names})]
    return round(float(df["f1"].mean()), 3)


def error_analysis(test, thresholds, categories, chunks, top_n):
    rows = []
    for i, cat in enumerate(categories):
        keep = np.where(test["masks"][:, i] > 0)[0]
        y, p = test["labels"][keep, i], test["probs"][keep, i]
        fp = keep[(y == 0) & (p >= thresholds[i])]
        fn = keep[(y == 1) & (p < thresholds[i])]
        for kind, idx, order in [("false_positive", fp, -1), ("false_negative", fn, 1)]:
            idx = idx[np.argsort(order * test["probs"][idx, i])][:top_n]
            for j in idx:
                ch = chunks.iloc[int(test["chunk_index"][j])]
                rows.append({
                    "category": cat, "error_type": kind,
                    "n_errors_in_category": len(fp) if kind == "false_positive" else len(fn),
                    "probability": round(float(test["probs"][j, i]), 3),
                    "threshold": thresholds[i],
                    "contract_title": ch["contract_title"], "chunk_id": int(ch["chunk_id"]),
                    "true_labels_in_chunk": ch["full_containment_labels"],
                    "chunk_text": ch["chunk_text"],
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories from train_distilroberta.py")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "training/data"))
    ap.add_argument("--out", default=str(REPO_ROOT / "training/outputs/evaluation"))
    ap.add_argument("--min-val-positives", type=int, default=5)
    ap.add_argument("--top-n-errors", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = pd.read_csv(Path(args.data_dir) / "test_chunks.csv")
    risk_names = pd.read_csv(REPO_ROOT / "evaluation/risk_categories.csv")["category"].tolist()

    summary, comparison = [], []
    for run in args.runs:
        name = Path(run).name
        val, test = load(run, "val"), load(run, "test")
        categories = [str(c) for c in test["categories"]]
        all_cols = list(range(len(categories)))

        v_probs, v_labels, v_masks = contract_level(
            val["probs"], val["labels"], val["masks"], list(val["contract_title"]))
        val_contract = {"probs": v_probs, "labels": v_labels, "masks": v_masks}
        tuned = {
            "chunk": tune_thresholds(val, categories, args.min_val_positives),
            "contract": tune_thresholds(val_contract, categories, args.min_val_positives),
        }
        pd.concat([df.assign(level=level) for level, df in tuned.items()]).to_csv(
            out / f"thresholds_{name}.csv", index=False)

        c_probs, c_labels, c_masks = contract_level(
            test["probs"], test["labels"], test["masks"], list(test["contract_title"]))

        per_cat = []
        for scheme in ["0.5", "tuned"]:
            for level, (p, l, m) in {
                "chunk": (test["probs"], test["labels"], test["masks"]),
                "contract": (c_probs, c_labels, c_masks),
            }.items():
                th = [0.5] * len(categories) if scheme == "0.5" else tuned[level]["threshold"].tolist()
                df = metrics(p, l, m, categories, th)
                per_cat.append(df.assign(level=level, threshold_scheme=scheme))
                summary.append({
                    "run": name, "threshold_scheme": scheme, "level": level,
                    "micro_f1": round(micro(p, l, m, th, all_cols), 3),
                    "macro_f1_all": macro(df),
                    "macro_f1_risk19": macro(df, risk_names),
                    "macro_f1_tfidf7": macro(df, TFIDF_BASELINE_CATEGORIES),
                })
                if level == "contract":
                    for _, r in df[df["category"].map(norm).isin(map(norm, TFIDF_RESULTS))].iterrows():
                        comparison.append({"category": r["category"], "model": f"{name} ({scheme})",
                                           "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})

        pd.concat(per_cat).to_csv(out / f"per_category_{name}.csv", index=False)
        error_analysis(test, tuned["chunk"]["threshold"].tolist(), categories, chunks, args.top_n_errors).to_csv(
            out / f"error_analysis_{name}.csv", index=False)

    for cat, (p, r, f) in TFIDF_RESULTS.items():
        comparison.append({"category": cat, "model": "tfidf_baseline", "precision": p, "recall": r, "f1": f})
    summary.append({"run": "tfidf_baseline", "threshold_scheme": "-", "level": "contract",
                    "micro_f1": None, "macro_f1_all": None, "macro_f1_risk19": None,
                    "macro_f1_tfidf7": round(float(np.mean([v[2] for v in TFIDF_RESULTS.values()])), 3)})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "run_summary.csv", index=False)
    comp = pd.DataFrame(comparison).pivot(index="category", columns="model", values="f1")
    comp.to_csv(out / "tfidf_comparison.csv")

    print(summary_df.to_string(index=False))
    print("\nContract-level F1 vs TF-IDF baseline:")
    print(comp.to_string())
    print(f"\nWrote evaluation to {out}")


if __name__ == "__main__":
    main()
