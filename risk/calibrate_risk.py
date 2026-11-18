#!/usr/bin/env python3
"""Select Change Of Control evidence settings on validation, then test once."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk.score_contracts import (
    BUCKETS,
    assign_buckets,
    cutoffs,
    evidence_rules_check,
    load_gold,
    load_model,
    norm,
    score_all,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIDENCES = [0.3, 0.4, 0.5, 0.6, 0.7]
MISSING_MULTIPLIERS = [0.1, 0.25, 0.5]
CHANGE_OF_CONTROL = "changeofcontrol"
BASELINE_CONFIDENCE = 0.5
BASELINE_MULTIPLIER = 0.25


def rules_for(base_rules, confidence, multiplier):
    rules = copy.deepcopy(base_rules)
    rule = next(r for r in rules["evidence_rules"] if norm(r["category"]) == CHANGE_OF_CONTROL)
    rule["backstop_confidence"] = confidence
    rule["multiplier_if_missing"] = multiplier
    return rules


def configured_model(base_model, base_multiplier, confidence, multiplier):
    """Apply candidate settings to cached detections without repeating regex scans."""
    model = copy.deepcopy(base_model)
    for found in model.values():
        entry = found.get("Change Of Control")
        if not entry:
            continue
        if entry["source"] == "keyword backstop":
            entry["confidence"] = confidence
        elif entry["source"] == "model, evidence rule missing (downweighted)":
            raw_confidence = entry["confidence"] / base_multiplier
            entry["confidence"] = raw_confidence * multiplier
    return model


def score_metrics(model_contracts, gold_contracts, levels, rules, bucket_cutoffs, gold_df=None,
                  baseline_buckets=None):
    model_df, _ = score_all(model_contracts, levels, rules)
    model_df["risk_bucket"] = assign_buckets(model_df, bucket_cutoffs, rules)
    if gold_df is None:
        gold_df, _ = score_all(gold_contracts, levels, rules)
        gold_df["risk_bucket"] = assign_buckets(gold_df, bucket_cutoffs, rules)
    model_df = model_df.set_index("contract_title")
    gold_df = gold_df.set_index("contract_title").reindex(model_df.index)
    if gold_df["risk_bucket"].isna().any():
        raise ValueError("Gold annotations are missing one or more scored contracts")

    gold_high = gold_df["risk_bucket"] == "High"
    model_high = model_df["risk_bucket"] == "High"
    model_low = model_df["risk_bucket"] == "Low"
    evidence = evidence_rules_check(model_contracts, gold_contracts, rules)
    coc = evidence[
        (evidence["category"].map(norm) == CHANGE_OF_CONTROL)
        & (evidence["variant"] == "model plus evidence backstop")
    ].iloc[0]
    result = {
        "rank_correlation": round(float(model_df["score"].rank().corr(gold_df["score"].rank())), 3),
        "bucket_agreement": round(float((model_df["risk_bucket"] == gold_df["risk_bucket"]).mean()), 3),
        "high_count": int((model_df["risk_bucket"] == "High").sum()),
        "medium_count": int((model_df["risk_bucket"] == "Medium").sum()),
        "low_count": int((model_df["risk_bucket"] == "Low").sum()),
        "gold_high_predicted_high": int((gold_high & model_high).sum()),
        "gold_high_total": int(gold_high.sum()),
        "gold_high_predicted_low": int((gold_high & model_low).sum()),
        "change_of_control_precision": float(coc["precision"]),
        "change_of_control_recall": float(coc["recall"]),
    }
    if baseline_buckets is not None:
        result["selection_score"] = round(
            (result["rank_correlation"] + result["bucket_agreement"]) / 2, 4
        )
        result["contracts_changing_bucket_from_baseline"] = int(
            (model_df["risk_bucket"] != baseline_buckets.reindex(model_df.index)).sum()
        )
    result["buckets"] = model_df["risk_bucket"]
    return result


def select_configuration(rows):
    """Preserve baseline rank correlation and avoid additional High-to-Low errors."""
    baseline = next(
        row for row in rows
        if row["backstop_confidence"] == BASELINE_CONFIDENCE
        and row["multiplier_if_missing"] == BASELINE_MULTIPLIER
    )
    eligible = [
        row for row in rows
        if row["rank_correlation"] >= baseline["rank_correlation"]
        and row["gold_high_predicted_low"] <= baseline["gold_high_predicted_low"]
    ]
    return max(
        eligible,
        key=lambda row: (
            row["selection_score"],
            -row["gold_high_predicted_low"],
            row["gold_high_predicted_high"],
            row["change_of_control_precision"],
            row["change_of_control_recall"],
            -row["contracts_changing_bucket_from_baseline"],
            -row["backstop_confidence"],
            -row["multiplier_if_missing"],
        ),
    )


def add_prefixed(target, prefix, metrics):
    for key, value in metrics.items():
        if key != "buckets":
            target[f"{prefix}_{key}"] = value


def write_grid(out, candidates):
    rows = [{key: value for key, value in row.items() if key != "buckets"}
            for row in candidates]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(REPO_ROOT / "training/outputs/sqrt_weighted"))
    parser.add_argument("--chunks", default=str(REPO_ROOT / "training/data"))
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation/risk_calibration_grid.csv"))
    parser.add_argument("--validation-only", action="store_true",
                        help="Write the validation grid and selection without opening test predictions")
    args = parser.parse_args()

    run_dir = Path(args.run)
    chunk_dir = Path(args.chunks)
    thresholds_path = run_dir / "thresholds_sqrt_weighted.csv"
    train_chunks_path = chunk_dir / "train_chunks.csv"
    test_chunks_path = chunk_dir / "test_chunks.csv"
    for required in [
        run_dir / "predictions_val.npz", thresholds_path, train_chunks_path,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    rubric = pd.read_csv(REPO_ROOT / "evaluation/risk_categories.csv")
    levels = {
        norm(category): (level, category)
        for category, level in zip(rubric["category"], rubric["initial_risk"])
    }
    base_rules = json.loads((REPO_ROOT / "risk/risk_rules.json").read_text(encoding="utf-8"))
    train_gold = load_gold(REPO_ROOT / "data/cuad/train_separate_questions.json")

    val_file = np.load(run_dir / "predictions_val.npz")
    val_indices = val_file["chunk_index"].astype(int)
    val_titles = val_file["contract_title"].astype(str)
    train_chunks = pd.read_csv(train_chunks_path)
    if (val_indices < 0).any() or (val_indices >= len(train_chunks)).any():
        raise ValueError("Validation chunk indices are outside train_chunks.csv")
    if not np.array_equal(train_chunks["contract_title"].iloc[val_indices].astype(str).to_numpy(), val_titles):
        raise ValueError("Validation chunk indices do not align with train_chunks.csv titles")
    val_titles_unique = list(dict.fromkeys(val_titles))
    missing_gold = set(val_titles_unique) - set(train_gold)
    if missing_gold:
        raise ValueError(f"Validation titles are missing from training gold: {sorted(missing_gold)[:3]}")
    validation_gold = {title: train_gold[title] for title in val_titles_unique}

    train_score_df, _ = score_all(train_gold, levels, base_rules)
    bucket_cutoffs = cutoffs(train_score_df["score"], base_rules)
    validation_gold_df, _ = score_all(validation_gold, levels, base_rules)
    validation_gold_df["risk_bucket"] = assign_buckets(validation_gold_df, bucket_cutoffs, base_rules)

    baseline_rules = rules_for(base_rules, BASELINE_CONFIDENCE, BASELINE_MULTIPLIER)
    cached_validation_model = load_model(
        run_dir, thresholds_path, train_chunks_path, "tuned",
        baseline_rules["evidence_rules"], split="val",
    )
    baseline_metrics = score_metrics(
        cached_validation_model, validation_gold, levels, baseline_rules, bucket_cutoffs,
        gold_df=validation_gold_df,
    )
    baseline_buckets = baseline_metrics["buckets"]

    candidates = []
    for confidence in CONFIDENCES:
        for multiplier in MISSING_MULTIPLIERS:
            rules = rules_for(base_rules, confidence, multiplier)
            model = configured_model(
                cached_validation_model, BASELINE_MULTIPLIER, confidence, multiplier,
            )
            metrics = score_metrics(
                model, validation_gold, levels, rules, bucket_cutoffs,
                gold_df=validation_gold_df, baseline_buckets=baseline_buckets,
            )
            candidates.append({
                "backstop_confidence": confidence,
                "multiplier_if_missing": multiplier,
                **metrics,
            })

    selected = select_configuration(candidates)
    for row in candidates:
        row["selected_on_validation"] = (
            row["backstop_confidence"] == selected["backstop_confidence"]
            and row["multiplier_if_missing"] == selected["multiplier_if_missing"]
        )

    out = Path(args.out)
    print(f"Validation contracts: {len(validation_gold)}", flush=True)
    print(f"Selected on validation: backstop_confidence={selected['backstop_confidence']}, "
          f"multiplier_if_missing={selected['multiplier_if_missing']}", flush=True)
    if args.validation_only:
        write_grid(out, candidates)
        return

    # Use the frozen settings directly through the production scoring path.
    selected_rules = rules_for(
        base_rules, selected["backstop_confidence"], selected["multiplier_if_missing"]
    )
    test_model = load_model(
        run_dir, thresholds_path, test_chunks_path, "tuned",
        selected_rules["evidence_rules"], split="test",
    )
    test_gold = load_gold(REPO_ROOT / "data/cuad/test.json")
    selected_test = score_metrics(test_model, test_gold, levels, selected_rules, bucket_cutoffs)

    baseline_summary = pd.read_csv(REPO_ROOT / "results/end_to_end_run1/model_vs_gold.csv").iloc[0]
    baseline_scores = pd.read_csv(REPO_ROOT / "results/end_to_end_run1/contract_scores.csv")
    baseline_counts = baseline_scores["risk_bucket"].value_counts()
    baseline_evidence = pd.read_csv(REPO_ROOT / "results/end_to_end_run1/evidence_rules_check.csv")
    baseline_coc = baseline_evidence[
        (baseline_evidence["category"].map(norm) == CHANGE_OF_CONTROL)
        & (baseline_evidence["variant"] == "model plus evidence backstop")
    ].iloc[0]
    selected_test["contracts_changing_bucket_from_committed_baseline"] = int(
        (selected_test["buckets"].reindex(baseline_scores.set_index("contract_title").index)
         != baseline_scores.set_index("contract_title")["risk_bucket"]).sum()
    )

    selected_row = next(row for row in candidates if row["selected_on_validation"])
    add_prefixed(selected_row, "test", selected_test)
    selected_row.update({
        "committed_baseline_test_rank_correlation": float(baseline_summary["rank_corr_with_gold"]),
        "committed_baseline_test_bucket_agreement": float(baseline_summary["bucket_agreement"]),
        "committed_baseline_test_gold_high_predicted_high": str(baseline_summary["gold_high_also_high"]),
        "committed_baseline_test_gold_high_predicted_low": int(baseline_summary["gold_high_flagged_low"]),
        "committed_baseline_test_high_count": int(baseline_counts.get("High", 0)),
        "committed_baseline_test_medium_count": int(baseline_counts.get("Medium", 0)),
        "committed_baseline_test_low_count": int(baseline_counts.get("Low", 0)),
        "committed_baseline_test_change_of_control_precision": float(baseline_coc["precision"]),
        "committed_baseline_test_change_of_control_recall": float(baseline_coc["recall"]),
    })

    write_grid(out, candidates)
    print(f"Wrote calibration grid to {out}")
    print("Selected configuration test metrics:")
    print({key: value for key, value in selected_test.items() if key != "buckets"})


if __name__ == "__main__":
    main()
