#!/usr/bin/env python3
"""Task #2: fine-tune DistilRoBERTa for multi-label CUAD clause classification.

Baseline training loop (meant for a Google Colab GPU, also runs on CPU for smoke tests):
  - input: tokenized chunks from training/prepare_dataset.py
  - model: distilroberta-base with one sigmoid output per CUAD category
  - loss: BCE-with-logits, masking partial-overlap (ambiguous) chunk/category pairs
  - validation: 10% of the official train contracts, split by contract (no leakage)
  - test: the official CUAD test contracts (test.json), never used for model selection

Writes to --out (default training/outputs, gitignored):
  best_model/                     model + tokenizer from the best validation epoch (macro F1)
  predictions_val.npz / predictions_test.npz
                                  probabilities, labels, masks, chunk indices and contract
                                  titles, consumed by training/evaluate.py (thresholds,
                                  baseline comparison, error analysis)
  history.csv                     per-epoch train loss and validation micro/macro F1
  chunk_level_metrics.csv         per-category P/R/F1 on test chunks
  contract_level_metrics.csv      per-category P/R/F1 on test contracts (max over chunks),
                                  comparable with the TF-IDF baseline notebook
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Categories reported by the TF-IDF baseline (notebooks/naeem_tfidf_baseline_risk_rules.ipynb).
TFIDF_BASELINE_CATEGORIES = [
    "Anti-Assignment",
    "Change Of Control",
    "Insurance",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Audit Rights",
    "Termination For Convenience",
]


class ChunkDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
            "label_mask": self.data["label_mask"][idx],
        }


def make_collate(tokenizer):
    pad = DataCollatorWithPadding(tokenizer)

    def collate(batch):
        out = pad([{"input_ids": b["input_ids"]} for b in batch])
        out["labels"] = torch.stack([b["labels"] for b in batch])
        out["label_mask"] = torch.stack([b["label_mask"] for b in batch])
        return out

    return collate


def masked_bce(logits, labels, label_mask, pos_weight=None):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight
    )
    return (loss * label_mask).sum() / label_mask.sum().clamp(min=1.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs, labels, masks = [], [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(batch["labels"])
        masks.append(batch["label_mask"])
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(masks).numpy()


def per_category_metrics(probs, labels, masks, categories, threshold):
    rows = []
    for i, cat in enumerate(categories):
        keep = masks[:, i] > 0
        y_true = labels[keep, i].astype(int)
        y_pred = (probs[keep, i] >= threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "Category": cat,
                "Support": int(y_true.sum()),
                "Predicted": int(y_pred.sum()),
                "Precision": round(float(prec), 3),
                "Recall": round(float(rec), 3),
                "F1_Score": round(float(f1), 3),
            }
        )
    return pd.DataFrame(rows)


def micro_f1(probs, labels, masks, threshold):
    keep = masks > 0
    _, _, f1, _ = precision_recall_fscore_support(
        labels[keep].astype(int), (probs[keep] >= threshold).astype(int),
        average="binary", zero_division=0,
    )
    return float(f1)


def contract_level(probs, labels, masks, titles):
    # A contract has a category if any chunk fully contains it or partially overlaps it;
    # the contract-level score is the max chunk probability.
    df_titles = pd.Series(titles)
    contract_probs, contract_labels = [], []
    for _, idx in df_titles.groupby(df_titles).groups.items():
        idx = np.asarray(idx)
        contract_probs.append(probs[idx].max(0))
        contract_labels.append(np.maximum(labels[idx], 1 - masks[idx]).max(0))
    probs_c = np.stack(contract_probs)
    labels_c = np.stack(contract_labels)
    return probs_c, labels_c, np.ones_like(labels_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "training/data"))
    ap.add_argument("--out", default=str(REPO_ROOT / "training/outputs"))
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pos-weight-mode",
        choices=["none", "raw", "sqrt"],
        default="none",
        help="Per-category positive weighting: raw = neg/pos, sqrt = sqrt(neg/pos) (Task #3).",
    )
    ap.add_argument(
        "--pos-weight-cap",
        type=float,
        default=30.0,
        help="Upper bound on any category's positive weight; raw neg/pos reaches ~600.",
    )
    ap.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Deprecated alias for --pos-weight-mode raw.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N train/test chunks (quick smoke test).",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(data_dir / "train_tokenized.pt")
    test_data = torch.load(data_dir / "test_tokenized.pt")
    categories = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))

    n_train = len(train_data["input_ids"]) if args.limit is None else args.limit
    n_test = len(test_data["input_ids"]) if args.limit is None else args.limit

    # Contract-level train/validation split so chunks of one contract stay together.
    train_titles = np.array(train_data["contract_title"][:n_train])
    unique_titles = sorted(set(train_titles))
    rng = np.random.default_rng(args.seed)
    if len(unique_titles) < 2:
        raise ValueError("Need at least 2 train contracts for a train/validation split; raise --limit.")
    n_val = min(len(unique_titles) - 1, max(1, int(len(unique_titles) * args.val_frac)))
    val_titles = set(rng.choice(unique_titles, size=n_val, replace=False))
    val_idx = [i for i, t in enumerate(train_titles) if t in val_titles]
    tr_idx = [i for i, t in enumerate(train_titles) if t not in val_titles]
    print(f"Train chunks: {len(tr_idx)} | val chunks: {len(val_idx)} ({len(val_titles)} contracts) | test chunks: {n_test}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | categories: {len(categories)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(categories),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(categories)),
        label2id={c: i for i, c in enumerate(categories)},
    ).to(device)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(ChunkDataset(train_data, tr_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ChunkDataset(train_data, val_idx), batch_size=args.batch_size * 2, collate_fn=collate)
    test_loader = DataLoader(ChunkDataset(test_data, range(n_test)), batch_size=args.batch_size * 2, collate_fn=collate)

    if args.use_pos_weight:
        args.pos_weight_mode = "raw"
    pos_weight = None
    if args.pos_weight_mode != "none":
        labels_tr = train_data["labels"][tr_idx]
        mask_tr = train_data["label_mask"][tr_idx]
        pos = (labels_tr * mask_tr).sum(0)
        neg = ((1 - labels_tr) * mask_tr).sum(0)
        ratio = neg / pos.clamp(min=1.0)
        if args.pos_weight_mode == "sqrt":
            ratio = ratio.sqrt()
        pos_weight = ratio.clamp(min=1.0, max=args.pos_weight_cap).to(device)
        print(f"pos_weight ({args.pos_weight_mode}, cap {args.pos_weight_cap}): "
              f"min {pos_weight.min():.1f}, median {pos_weight.median():.1f}, max {pos_weight.max():.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
            loss = masked_bce(
                logits.float(), batch["labels"].to(device), batch["label_mask"].to(device), pos_weight
            )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        val_probs, val_labels, val_masks = predict(model, val_loader, device, use_amp)
        val_micro = micro_f1(val_probs, val_labels, val_masks, args.threshold)
        val_macro = per_category_metrics(val_probs, val_labels, val_masks, categories, args.threshold)["F1_Score"].mean()
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader),
                "val_micro_f1": round(val_micro, 4),
                "val_macro_f1": round(float(val_macro), 4),
                "minutes": round((time.time() - start) / 60, 2),
            }
        )
        print(f"Epoch {epoch}: {history[-1]}")

        # Select on macro F1 so frequent, easy categories (Document Name,
        # Parties, ...) don't dominate checkpoint choice.
        if val_macro > best_f1:
            best_f1 = val_macro
            model.save_pretrained(out / "best_model")
            tokenizer.save_pretrained(out / "best_model")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Final test evaluation with the best validation checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(out / "best_model").to(device)
    test_probs, test_labels, test_masks = predict(model, test_loader, device, use_amp)
    val_probs, val_labels, val_masks = predict(model, val_loader, device, use_amp)
    for name, probs, labels, masks, idx, data in [
        ("val", val_probs, val_labels, val_masks, val_idx, train_data),
        ("test", test_probs, test_labels, test_masks, list(range(n_test)), test_data),
    ]:
        np.savez_compressed(
            out / f"predictions_{name}.npz",
            probs=probs, labels=labels, masks=masks,
            chunk_index=np.array(idx),
            contract_title=np.array([data["contract_title"][i] for i in idx]),
            categories=np.array(categories),
        )

    chunk_df = per_category_metrics(test_probs, test_labels, test_masks, categories, args.threshold)
    chunk_df.to_csv(out / "chunk_level_metrics.csv", index=False)

    titles = test_data["contract_title"][:n_test]
    c_probs, c_labels, c_masks = contract_level(test_probs, test_labels, test_masks, titles)
    contract_df = per_category_metrics(c_probs, c_labels, c_masks, categories, args.threshold)
    contract_df.to_csv(out / "contract_level_metrics.csv", index=False)

    print("\nTest chunk-level micro F1:", round(micro_f1(test_probs, test_labels, test_masks, args.threshold), 3))
    print("Test chunk-level macro F1:", round(chunk_df["F1_Score"].mean(), 3))
    print("\nContract-level metrics on the TF-IDF baseline categories:")
    print(contract_df[contract_df["Category"].isin(TFIDF_BASELINE_CATEGORIES)].to_string(index=False))
    print(f"\nSaved model and metrics to {out}")


if __name__ == "__main__":
    main()
