#!/usr/bin/env python3
"""Rule-based contract risk scoring on top of the clause classifier.

Turns detected clauses into a clause register and a Low / Medium / High triage
bucket per contract. Every point of a contract's score traces back to a named
reason, so the ranking can be audited.

Four signals (weights in risk/risk_rules.json, category levels in
evaluation/risk_categories.csv):
  1. clauses present        level weight x confidence, summed over detected clauses
  2. most severe clause      a confident High-risk clause sets a minimum bucket
  3. missing protections     e.g. no Cap On Liability, or Uncapped Liability without one
  4. risky combinations      e.g. Exclusivity + Minimum Commitment

Buckets: cutoffs are percentiles of the scores of the 408 training contracts,
computed from CUAD annotations so they don't depend on the model or the test set.

Sources:
  --source gold    CUAD annotations (confidence 1). A perfect-classifier ceiling;
                   runs without a trained model.
  --source model   predictions_test.npz from training/train_distilroberta.py, with
                   per-category contract thresholds from training/evaluate.py.
                   Contract confidence = max chunk probability; evidence = that chunk.
                   Applies evidence rules from risk_rules.json and compares the
                   ranking with the gold one.

Writes to --out (default risk/outputs/<source>, gitignored):
  contract_scores.csv     rank, bucket, score and reasons per contract
  clause_register.csv     one row per detected clause with level, confidence,
                          points and evidence text
  sensitivity.csv         bucket changes and rank correlation under perturbed
                          cutoffs and weights
  model_vs_gold.csv       (model) rank correlation and bucket agreement with gold
    evidence_rules_check.csv (model) precision/recall of evidence backstops

Usage:
    python risk/score_contracts.py --source gold
    python risk/score_contracts.py --source model --run training/outputs/sqrt_weighted
"""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ["Low", "Medium", "High"]


def norm(name):
    return name.lower().replace(" ", "")


def get_category(qa):
    # Same parsing as the Week 3 chunker (Week 3/tokenizer_chunking_experiment.py).
    m = re.search(r'related to "([^"]+)"', qa.get("question", ""))
    if m:
        return m.group(1)
    return re.sub(r"_\d+$", "", qa["id"].split("__", 1)[1])


def load_gold(path):
    """{title: {category: entry}} from CUAD annotations; a category is present if it has any answer."""
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    contracts = {}
    for item in dataset["data"]:
        found = {}
        for p in item["paragraphs"]:
            for qa in p["qas"]:
                answers = qa.get("answers", [])
                cat = get_category(qa)
                if answers and cat not in found:
                    found[cat] = {"confidence": 1.0, "evidence": answers[0]["text"], "source": "annotation"}
        contracts[item["title"]] = found
    return contracts


def load_model(run, thresholds_path, chunks_path, scheme, evidence_rules):
    """{title: {category: entry}} from saved test predictions (max probability over chunks)."""
    pred = np.load(Path(run) / "predictions_test.npz")
    categories = [str(c) for c in pred["categories"]]
    probs, titles = pred["probs"], pred["contract_title"]
    chunks = pd.read_csv(chunks_path)
    texts = chunks["chunk_text"].iloc[pred["chunk_index"]].to_numpy()

    if scheme == "tuned":
        th = pd.read_csv(thresholds_path)
        th = th[th["level"] == "contract"].set_index("category")["threshold"]
        thresholds = [float(th[c]) for c in categories]
    else:
        thresholds = [0.5] * len(categories)

    rules_by_category = {norm(r["category"]): r for r in evidence_rules}

    contracts = {}
    for title in pd.unique(titles):
        rows = np.where(titles == title)[0]
        found = {}
        for i, cat in enumerate(categories):
            best = rows[probs[rows, i].argmax()]
            conf = float(probs[best, i])
            rule = rules_by_category.get(norm(cat))
            matching_rows = []
            if rule:
                matching_rows = [r for r in rows if re.search(rule["pattern"], texts[r])]
            if conf >= thresholds[i]:
                entry = {"confidence": conf, "evidence": texts[best], "source": "model"}
                if rule and matching_rows:
                    entry["evidence"] = texts[matching_rows[0]]
                    entry["rule_reason"] = f"Evidence rule matched: {rule['name']}"
                    entry["source"] = "model, evidence rule matched"
                elif rule:
                    entry["confidence"] = conf * rule["multiplier_if_missing"]
                    entry["rule_reason"] = f"Evidence rule not found: {rule['name']}"
                    entry["source"] = "model, evidence rule missing (downweighted)"
                found[cat] = entry
            elif rule and matching_rows and "backstop_confidence" in rule:
                found[cat] = {
                    "confidence": rule["backstop_confidence"],
                    "evidence": texts[matching_rows[0]],
                    "source": "keyword backstop",
                    "rule_reason": f"Evidence backstop matched: {rule['name']}",
                }
        contracts[str(title)] = found
    return contracts


def score_contract(found, levels, rules):
    """Score one contract. Returns (score, reasons, severity_floor, register rows)."""
    weights = rules["level_weights"]
    conf = {norm(c): e["confidence"] for c, e in found.items()}
    reasons, register = [], []

    # Signal 1: clauses present.
    for cat, entry in found.items():
        level = levels.get(norm(cat), (None, None))[0]
        points = weights[level] * entry["confidence"] if level else 0.0
        register.append({"category": levels.get(norm(cat), (None, cat))[1], "risk_level": level or "Unrated",
                 "confidence": round(entry["confidence"], 3), "points": round(points, 3),
                 "source": entry["source"], "rule_reason": entry.get("rule_reason", ""),
                 "evidence": entry["evidence"]})
        if points:
            reasons.append((f"{register[-1]['category']} ({level}, p={entry['confidence']:.2f})", points))

    # Signal 3: missing protections.
    for rule in rules["missing_protections"]:
        if conf.get(norm(rule["missing"]), 0) > 0:
            continue
        scale = conf.get(norm(rule["present"]), 0) if "present" in rule else 1.0
        if scale > 0:
            reasons.append((rule["name"], rule["points"] * scale))

    # Signal 4: risky combinations, scaled by the least confident clause.
    for rule in rules["combinations"]:
        scale = min(conf.get(norm(c), 0) for c in rule["all_of"])
        if scale > 0:
            reasons.append((f"{rule['name']} (combination)", rule["points"] * scale))

    # Signal 2: most severe clause.
    floor_conf = rules["severity_floor"]["min_confidence"]
    floor = any(levels.get(norm(c), (None,))[0] == "High" and e["confidence"] >= floor_conf
                for c, e in found.items())

    reasons.sort(key=lambda r: -r[1])
    return sum(p for _, p in reasons), reasons, floor, register


def score_all(contracts, levels, rules):
    rows, register = [], []
    for title, found in contracts.items():
        score, reasons, floor, reg = score_contract(found, levels, rules)
        n_high = sum(r["risk_level"] == "High" for r in reg)
        n_medium = sum(r["risk_level"] == "Medium" for r in reg)
        rows.append({"contract_title": title, "score": round(score, 3), "severity_floor": floor,
                     "high_risk_clauses": n_high, "medium_risk_clauses": n_medium,
                     "reasons": "; ".join(f"{name} +{p:.2f}" for name, p in reasons)})
        register += [{"contract_title": title, **r} for r in reg]
    return pd.DataFrame(rows), pd.DataFrame(register)


def cutoffs(train_scores, rules):
    b = rules["buckets"]
    high = np.percentile(train_scores, 100 - b["high_top_pct"])
    medium = np.percentile(train_scores, 100 - b["high_top_pct"] - b["medium_next_pct"])
    return medium, high


def assign_buckets(df, cuts, rules):
    medium, high = cuts
    floor = BUCKETS.index(rules["severity_floor"]["floor_bucket"])
    idx = np.where(df["score"] >= high, 2, np.where(df["score"] >= medium, 1, 0))
    idx = np.where(df["severity_floor"], np.maximum(idx, floor), idx)
    return pd.Series([BUCKETS[i] for i in idx], index=df.index)


def run(contracts, train_gold, levels, rules):
    """Score contracts with cutoffs taken from the training contracts under the same rules."""
    train_df, _ = score_all(train_gold, levels, rules)
    cuts = cutoffs(train_df["score"], rules)
    df, register = score_all(contracts, levels, rules)
    df["risk_bucket"] = assign_buckets(df, cuts, rules)
    return df, register, cuts


def rank_corr(a, b):
    return round(float(a.rank().corr(b.rank())), 3)


def sensitivity(contracts, train_gold, levels, rules, base):
    variants = {
        "High cutoff top 15%": ("buckets", "high_top_pct", 15),
        "High cutoff top 25%": ("buckets", "high_top_pct", 25),
        "Medium band 25%": ("buckets", "medium_next_pct", 25),
        "Medium band 35%": ("buckets", "medium_next_pct", 35),
        "High weight 2": ("level_weights", "High", 2),
        "High weight 4": ("level_weights", "High", 4),
        "Medium weight 0.5": ("level_weights", "Medium", 0.5),
        "Medium weight 1.5": ("level_weights", "Medium", 1.5),
    }
    rows = []
    for name, (section, key, value) in variants.items():
        r = copy.deepcopy(rules)
        r[section][key] = value
        df, _, _ = run(contracts, train_gold, levels, r)
        df = df.set_index("contract_title").loc[base.index]
        changed = df["risk_bucket"] != base["risk_bucket"]
        rows.append({"variant": name, "contracts_changing_bucket": int(changed.sum()),
                     "pct_changing_bucket": round(100 * changed.mean(), 1),
                     "rank_corr_with_default": rank_corr(df["score"], base["score"]),
                     **{f"n_{b.lower()}": int((df["risk_bucket"] == b).sum()) for b in BUCKETS}})
    return pd.DataFrame(rows)


def evidence_rules_check(model, gold, rules):
    """Contract-level precision/recall of each evidence backstop against annotations."""
    rows = []
    for rule in rules.get("evidence_rules", []):
        if "backstop_confidence" not in rule:
            continue
        cat = norm(rule["category"])
        truth = {t: any(norm(c) == cat for c in gold[t]) for t in model}
        variants = {
            "model only": lambda entry: entry["source"] != "keyword backstop",
            "model plus evidence backstop": lambda entry: True,
        }
        for label, keep in variants.items():
            pred = {
                title: any(norm(category) == cat and keep(entry)
                           for category, entry in found.items())
                for title, found in model.items()
            }
            true_positives = sum(pred[title] and truth[title] for title in model)
            rows.append({"category": rule["category"], "rule": rule["name"],
                         "variant": label, "detected": sum(pred.values()),
                         "true_positives": true_positives, "actual": sum(truth.values()),
                         "precision": round(true_positives / max(sum(pred.values()), 1), 3),
                         "recall": round(true_positives / max(sum(truth.values()), 1), 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gold", "model"], required=True)
    ap.add_argument("--run", help="run directory from train_distilroberta.py (model source)")
    ap.add_argument("--thresholds", help="thresholds_<run>.csv from evaluate.py "
                    "(default training/outputs/evaluation/thresholds_<run name>.csv)")
    ap.add_argument("--threshold-scheme", choices=["tuned", "0.5"], default="tuned")
    ap.add_argument("--chunks", default=str(REPO_ROOT / "training/data/test_chunks.csv"))
    ap.add_argument("--train", default=str(REPO_ROOT / "data/cuad/train_separate_questions.json"))
    ap.add_argument("--test", default=str(REPO_ROOT / "data/cuad/test.json"))
    ap.add_argument("--rubric", default=str(REPO_ROOT / "evaluation/risk_categories.csv"))
    ap.add_argument("--rules", default=str(REPO_ROOT / "risk/risk_rules.json"))
    ap.add_argument("--out", help="output directory (default risk/outputs/<source>)")
    args = ap.parse_args()

    rubric = pd.read_csv(args.rubric)
    levels = {norm(c): (lvl, c) for c, lvl in zip(rubric["category"], rubric["initial_risk"])}
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    train_gold = load_gold(args.train)
    test_gold = load_gold(args.test)

    if args.source == "gold":
        contracts = test_gold
        out = Path(args.out or REPO_ROOT / "risk/outputs/gold")
    else:
        if not args.run:
            ap.error("--run is required with --source model")
        name = Path(args.run).name
        thresholds = args.thresholds or REPO_ROOT / f"training/outputs/evaluation/thresholds_{name}.csv"
        contracts = load_model(args.run, thresholds, args.chunks, args.threshold_scheme, rules["evidence_rules"])
        out = Path(args.out or REPO_ROOT / f"risk/outputs/model_{name}")
    out.mkdir(parents=True, exist_ok=True)

    df, register, cuts = run(contracts, train_gold, levels, rules)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    cols = ["rank", "contract_title", "risk_bucket", "score", "severity_floor",
            "high_risk_clauses", "medium_risk_clauses", "reasons"]
    df[cols].to_csv(out / "contract_scores.csv", index=False)
    register.sort_values(["contract_title", "points"], ascending=[True, False]).to_csv(
        out / "clause_register.csv", index=False)

    base = df.set_index("contract_title")
    sens = sensitivity(contracts, train_gold, levels, rules, base)
    sens.to_csv(out / "sensitivity.csv", index=False)

    print(f"Source: {args.source} | contracts: {len(df)} | "
          f"cutoffs from train: Medium >= {cuts[0]:.2f}, High >= {cuts[1]:.2f}")
    print("Buckets:", df["risk_bucket"].value_counts().reindex(BUCKETS[::-1]).to_dict(),
          f"| severity floor applied to {int(df['severity_floor'].sum())}")
    print("\nTop 10 contracts:")
    for _, r in df.head(10).iterrows():
        print(f"  {r['rank']:>2}. [{r['risk_bucket']}] {r['score']:5.2f}  {r['contract_title'][:70]}")
        print(f"      {r['reasons'][:160]}")
    print("\nSensitivity:")
    print(sens.to_string(index=False))

    if args.source == "model":
        gold_df, _, _ = run(test_gold, train_gold, levels, rules)
        g = gold_df.set_index("contract_title").loc[base.index]
        agree = base["risk_bucket"] == g["risk_bucket"]
        gold_high = g["risk_bucket"] == "High"
        comparison = pd.DataFrame([{
            "rank_corr_with_gold": rank_corr(base["score"], g["score"]),
            "bucket_agreement": round(float(agree.mean()), 3),
            "gold_high_also_high": f"{int((gold_high & (base['risk_bucket'] == 'High')).sum())}/{int(gold_high.sum())}",
            "gold_high_flagged_low": int((gold_high & (base["risk_bucket"] == "Low")).sum()),
        }])
        comparison.to_csv(out / "model_vs_gold.csv", index=False)
        crosstab = pd.crosstab(g["risk_bucket"].rename("gold"), base["risk_bucket"].rename("model"))
        evidence = evidence_rules_check(contracts, test_gold, rules)
        evidence.to_csv(out / "evidence_rules_check.csv", index=False)
        print("\nModel vs gold (same rules and cutoffs):")
        print(comparison.to_string(index=False))
        print(crosstab.reindex(index=BUCKETS, columns=BUCKETS, fill_value=0).to_string())
        print("\nEvidence backstops vs annotations (contract level):")
        print(evidence.to_string(index=False))

    print(f"\nWrote risk scores to {out}")


if __name__ == "__main__":
    main()
