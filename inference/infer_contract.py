#!/usr/bin/env python3
"""Single-contract inference (Phase B of issue #23).

Takes one raw contract text file and runs it through the same steps the
'score' stage of run_pipeline.py runs on the whole CUAD test set, but for a
single contract on demand:

  1. chunk it with the Week 3 paragraph-aware token chunker (same strategy
     training/prepare_dataset.py trained the model with)
  2. classify every chunk with the trained model
  3. apply the saved per-category thresholds and the risk/risk_rules.json
     evidence-rule regex checks, by calling risk.score_contracts.load_model()
     with this one contract's chunks shaped the way it expects
  4. score it with risk.score_contracts.score_contract() and bucket it with
     risk.score_contracts.assign_buckets(), using cutoffs computed once from
     the 408 CUAD training contracts and then cached

This module holds the reusable pieces. run_pipeline.py's 'infer' subcommand
is the thin CLI wrapper around infer_contract().
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from risk.score_contracts import (
    assign_buckets,
    cutoffs as compute_cutoffs,
    load_gold,
    load_model,
    norm,
    score_all,
    score_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKER_PATH = REPO_ROOT / "Week 3" / "tokenizer_chunking_experiment.py"

# Same chunking strategy training/prepare_dataset.py trained the model with.
CHUNK_KIND, CHUNK_MAX_TOKENS, CHUNK_OVERLAP = "paragraph", 480, 64


def load_chunker():
    """Load Week 3's chunker module from its file path, the same way
    training/prepare_dataset.py does, since 'Week 3' isn't a Python package."""
    spec = importlib.util.spec_from_file_location("tokenizer_chunking_experiment", CHUNKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_contract(text, tokenizer, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP):
    """Paragraph-aware token chunks for one raw contract.

    paragraph_token_chunks() only needs (text, tokenizer, max_tokens, overlap) -
    it has no CUAD-specific state, so it works on any raw text, not just
    CUAD paragraphs.
    """
    chunker = load_chunker()
    chunk_fn = chunker.make_fn(tokenizer, CHUNK_KIND, max_tokens, overlap)
    return chunk_fn(text)


def pick_device():
    """mps if this machine has it, otherwise cpu. No cuda branch: inference here
    is one short-lived call, not worth the training script's cuda/amp machinery."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_contract_text(contract_path):
    """Read a raw contract file as text, failing clearly instead of on a
    confusing downstream error if it's missing, empty, or not text."""
    path = Path(contract_path)
    if not path.is_file():
        raise FileNotFoundError(f"Contract file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read {path} as UTF-8 text: {exc}") from exc
    if not text.strip():
        raise ValueError(f"Contract file is empty: {path}")
    return text


def load_categories(labels_path):
    """Category names in model-output order, from training/prepare_dataset.py's
    labels.json - not from model.config.id2label, so a mismatch between the
    model's checkpoint and the labels file it was trained with is caught
    explicitly in load_classifier() rather than silently mislabeling outputs."""
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"labels.json not found at {path}. Run 'python run_pipeline.py prepare' first, "
            "or check the manifest's 'prepare' stage data_dir."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_classifier(best_model_dir, categories):
    """Load the trained model and its own tokenizer from best_model_dir."""
    best_model_dir = Path(best_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    if not tokenizer.is_fast:
        raise ValueError(f"Fast tokenizer required for exact offset mapping (got a slow tokenizer from {best_model_dir}).")

    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    if model.config.num_labels != len(categories):
        raise ValueError(
            f"Model at {best_model_dir} has {model.config.num_labels} output labels, "
            f"but {len(categories)} categories were loaded from labels.json. "
            "The model checkpoint and labels.json must come from the same training run."
        )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def classify_chunks(model, tokenizer, chunk_texts, device, max_length=512, batch_size=16):
    """Sigmoid probability per category for every chunk, batched like
    train_distilroberta.py's predict()."""
    model.to(device)
    all_probs = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        all_probs.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(all_probs).numpy()


def detect_clauses(contract_title, categories, probs, chunk_texts, thresholds_csv, evidence_rules, scheme="tuned"):
    """Apply per-category thresholds and evidence-rule regex checks the same way
    risk.score_contracts.load_model() does for the full test set, by handing it
    this one contract's chunks shaped as the predictions npz + chunks csv it
    expects. This calls the real function instead of re-deriving its logic.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(
            tmp / "predictions_test.npz",
            categories=np.asarray(categories),
            probs=np.asarray(probs, dtype=float),
            contract_title=np.asarray([contract_title] * len(chunk_texts)),
            chunk_index=np.arange(len(chunk_texts)),
        )
        chunks_csv = tmp / "chunks.csv"
        pd.DataFrame({"chunk_text": chunk_texts}).to_csv(chunks_csv, index=False)
        contracts = load_model(tmp, thresholds_csv, chunks_csv, scheme, evidence_rules)
    return contracts.get(contract_title, {})


def get_or_compute_cutoffs(manifest, rules_path, rubric_path, train_json_path):
    """Bucket cutoffs are percentiles of the 408 CUAD training contracts' gold
    scores (risk/risk_rules.json) - they depend only on the rules/rubric, not on
    the model, so compute them once per manifest and cache them instead of
    rescoring 408 contracts on every 'infer' call.

    Returns (medium, high, info). info is None on a cache hit (nothing new to
    save); otherwise it's the dict the caller should record into the manifest's
    'cutoffs' stage.
    """
    rules_path, rubric_path, train_json_path = str(rules_path), str(rubric_path), str(train_json_path)
    cached = manifest.get("stages", {}).get("cutoffs")
    if (
        cached
        and cached.get("rules_path") == rules_path
        and cached.get("rubric_path") == rubric_path
        and cached.get("train_json") == train_json_path
    ):
        return cached["medium"], cached["high"], None

    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    rubric = pd.read_csv(rubric_path)
    levels = {norm(c): (lvl, c) for c, lvl in zip(rubric["category"], rubric["initial_risk"])}
    train_gold = load_gold(train_json_path)
    train_df, _ = score_all(train_gold, levels, rules)
    medium, high = compute_cutoffs(train_df["score"], rules)

    info = {
        "rules_path": rules_path, "rubric_path": rubric_path, "train_json": train_json_path,
        "medium": round(float(medium), 4), "high": round(float(high), 4),
    }
    return info["medium"], info["high"], info


def score_and_bucket(found, levels, rules, cuts):
    """Score one contract's detected clauses and assign its bucket, reusing
    risk.score_contracts.score_contract() and assign_buckets() exactly as the
    batch 'score' stage does."""
    score, reasons, floor, register = score_contract(found, levels, rules)
    one_row = pd.DataFrame([{"score": score, "severity_floor": floor}])
    bucket = assign_buckets(one_row, cuts, rules).iloc[0]
    return score, reasons, floor, register, bucket


def infer_contract(
    contract_path, best_model_dir, labels_path, thresholds_csv, rubric_path, rules_path, cuts,
    max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP, max_length=512, threshold_scheme="tuned",
):
    """Run the full single-contract pipeline and return a JSON-able result dict."""
    contract_path = Path(contract_path)
    text = read_contract_text(contract_path)
    title = contract_path.stem

    categories = load_categories(labels_path)
    device = pick_device()
    model, tokenizer = load_classifier(best_model_dir, categories)

    chunks = chunk_contract(text, tokenizer, max_tokens, overlap)
    if not chunks:
        raise ValueError(f"Contract produced no chunks after tokenization: {contract_path}")
    chunk_texts = [c["text"] for c in chunks]
    probs = classify_chunks(model, tokenizer, chunk_texts, device, max_length)

    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    rubric = pd.read_csv(rubric_path)
    levels = {norm(c): (lvl, c) for c, lvl in zip(rubric["category"], rubric["initial_risk"])}

    found = detect_clauses(title, categories, probs, chunk_texts, thresholds_csv, rules["evidence_rules"], threshold_scheme)
    score, reasons, floor, register, bucket = score_and_bucket(found, levels, rules, cuts)

    return {
        "contract_title": title,
        "contract_path": str(contract_path),
        "chunk_count": len(chunks),
        "score": round(float(score), 3),
        "risk_bucket": bucket,
        "severity_floor_applied": bool(floor),
        "reasons": [{"name": name, "points": round(float(p), 3)} for name, p in reasons],
        "clauses": register,
    }


def format_report(result):
    """Human-readable printout of an infer_contract() result."""
    lines = [
        f"Contract: {result['contract_title']}  ({result['contract_path']})",
        f"Chunks classified: {result['chunk_count']}",
        f"Risk bucket: {result['risk_bucket']}  (score {result['score']:.2f}"
        + (", severity floor applied" if result["severity_floor_applied"] else "") + ")",
        "",
        "Detected clauses:",
    ]
    clauses = sorted(result["clauses"], key=lambda r: -r["points"])
    if not clauses:
        lines.append("  (none)")
    for c in clauses:
        lines.append(
            f"  [{c['risk_level']:>8}] {c['category']:<45} conf={c['confidence']:.2f}  +{c['points']:.2f}  ({c['source']})"
        )
        evidence = " ".join(c["evidence"].split())
        lines.append(f"            evidence: {evidence[:160]}")

    lines.append("")
    lines.append("Top reasons:")
    for r in sorted(result["reasons"], key=lambda r: -r["points"])[:10]:
        lines.append(f"  {r['name']} +{r['points']:.2f}")
    return "\n".join(lines)
