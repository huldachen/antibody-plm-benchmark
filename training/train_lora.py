"""
LoRA fine-tuning script for ESM-2 on antibody CDR-H3 fitness prediction.

Custom training loop (not Trainer) for transparency over
LoRA application and per-epoch metric logging. Saves adapter weights only.

Usage:
    python training/train_lora.py --dataset mason --task classification \\
        --lora-rank 8 --lr 1e-4 --epochs 10 --output-dir training/checkpoints/mason_r8

Smoke test (5 steps on 32 samples, asserts loss decrease):
    python training/train_lora.py --dataset mason --task classification --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EsmForSequenceClassification,
)

# Local
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from training.dataloader import (
    CDRH3Dataset,
    label_summary,
    load_dataframe,
    stratified_split,
    distance_split,
)


DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
LORA_TARGETS_ESM2 = ["query", "key", "value", "dense"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_labels: int, problem_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, TaskType, get_peft_model  # local import for optional dep

    base = EsmForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGETS_ESM2,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    return model


def compute_metrics(preds: np.ndarray, labels: np.ndarray, is_classification: bool) -> dict:
    if is_classification:
        # preds are logits of shape (N, 2)
        probs = torch.softmax(torch.from_numpy(preds), dim=-1).numpy()[:, 1]
        pred_class = (probs >= 0.5).astype(int)
        out = {
            "accuracy": float(accuracy_score(labels, pred_class)),
            "mcc": float(matthews_corrcoef(labels, pred_class)),
        }
        if len(np.unique(labels)) == 2:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        return out
    # regression: preds shape (N, 1)
    flat = preds.reshape(-1)
    return {
        "mse": float(mean_squared_error(labels, flat)),
        "rmse": float(np.sqrt(mean_squared_error(labels, flat))),
        "r2": float(r2_score(labels, flat)),
    }


@torch.no_grad()
def evaluate(model, loader, device, is_classification: bool, return_preds: bool = False):
    model.eval()
    all_preds, all_labels, total_loss, n = [], [], 0.0, 0
    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out = model(**inputs, labels=labels)
        all_preds.append(out.logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        bs = labels.shape[0]
        total_loss += float(out.loss.detach().cpu()) * bs
        n += bs
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(preds, labels, is_classification)
    metrics["loss"] = total_loss / max(n, 1)
    # The metrics dict stays scalars-only: the per-epoch logger rounds every
    # value, so smuggling arrays through it breaks training (it did). Callers
    # that want the raw per-sample outputs -- to keep a run comparable later
    # without the adapter weights -- ask for them explicitly.
    if return_preds:
        return metrics, preds, labels
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune ESM-2 on Mason/Absci CDR-H3")
    p.add_argument("--dataset", choices=["mason", "absci"], required=True)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["random", "distance"], default="random",
                   help="random stratified (optimistic) or leave-cluster-out")
    p.add_argument("--d-link", type=int, default=2,
                   help="Hamming linkage threshold for --split distance")
    p.add_argument("--output-dir", required=False)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="Compute device. 'auto' picks cuda > mps > cpu.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 5 steps on 32 samples and assert loss decrease, then exit.")
    return p.parse_args()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    print(f"[init] device={device} model={args.model_name}")

    # Data
    df, spec = load_dataframe(args.dataset, args.task)
    print(f"[data] {args.dataset}/{args.task} loaded: {label_summary(df, spec)}")

    if args.split == "distance":
        train_df, val_df, test_df = distance_split(
            df, spec, d_link=args.d_link, seed=args.seed)
    else:
        train_df, val_df, test_df = stratified_split(df, spec, seed=args.seed)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Smoke-test reduction: keep 32 train + 16 val
    if args.smoke_test:
        train_df = train_df.sample(n=min(32, len(train_df)), random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(16, len(val_df)), random_state=args.seed).reset_index(drop=True)
        args.epochs = 1
        print(f"[smoke] reduced to train={len(train_df)} val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = CDRH3Dataset(train_df, tokenizer, spec, max_length=args.max_length)
    val_ds = CDRH3Dataset(val_df, tokenizer, spec, max_length=args.max_length)
    test_ds = CDRH3Dataset(test_df, tokenizer, spec, max_length=args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Model
    if spec.is_classification:
        num_labels = int(df[spec.label_col].nunique())
        problem_type = "single_label_classification"
    else:
        num_labels = 1
        problem_type = "regression"

    model = build_model(
        args.model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={trainable:,} ({100*trainable/total:.2f}% of {total:,})")

    # Verify the adapter actually attached. PEFT silently ignores any entry in
    # target_modules that matches no module, and `esm` is not in PEFT's built-in
    # mapping -- so a config copied from a Llama/Mistral example
    # (target_modules=["q_proj","v_proj"]) binds to NOTHING on ESM-2, whose
    # attention projections are named query/key/value. Training then quietly
    # degrades to frozen-embeddings-plus-head, and every downstream comparison
    # is an artifact. Counting the adapters is the only honest confirmation.
    adapted = sorted({n.split(".lora_")[0] for n, _ in model.named_parameters()
                      if "lora_" in n})
    head_only = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and "lora_" not in n)
    print(f"[lora] adapters attached to {len(adapted)} modules; "
          f"non-adapter trainable (head etc.) = {head_only:,}")
    if not adapted:
        raise RuntimeError(
            f"LoRA attached to NO modules. target_modules={LORA_TARGETS_ESM2} "
            f"matched nothing in {args.model_name}. ESM-2 uses "
            f"query/key/value, not q_proj/v_proj."
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # ----- Smoke test path -----
    if args.smoke_test:
        model.train()
        losses = []
        it = iter(train_loader)
        for step in range(5):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.detach().cpu()))
            print(f"[smoke] step={step} loss={losses[-1]:.4f}")
        ok = losses[-1] < losses[0]
        print(f"[smoke] first={losses[0]:.4f} last={losses[-1]:.4f} decreased={ok}")
        if not ok:
            print("[smoke] FAIL: loss did not decrease over 5 steps", file=sys.stderr)
            return 2
        print("[smoke] PASS")
        return 0

    # ----- Real training -----
    if not args.output_dir:
        print("[error] --output-dir is required for real training", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    log_fh = open(log_path, "w", newline="")
    log_writer: Optional[csv.DictWriter] = None

    best_val_metric = None
    best_epoch = -1
    best_metric_key = "roc_auc" if spec.is_classification else "rmse"
    best_better = (lambda new, cur: new > cur) if spec.is_classification else (lambda new, cur: new < cur)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            bs = labels.shape[0]
            running += float(out.loss.detach().cpu()) * bs
            n_seen += bs
        train_loss = running / max(n_seen, 1)

        val_metrics = evaluate(model, val_loader, device, spec.is_classification)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": round(train_loss, 6), "elapsed_s": round(elapsed, 1)}
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):      # scalars only; never assume
                row[f"val_{k}"] = round(v, 6)
        if log_writer is None:
            log_writer = csv.DictWriter(log_fh, fieldnames=list(row.keys()))
            log_writer.writeheader()
        log_writer.writerow(row)
        log_fh.flush()
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} "
              + " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
              + f" t={elapsed:.1f}s")

        cur = val_metrics.get(best_metric_key)
        if cur is not None and (best_val_metric is None or best_better(cur, best_val_metric)):
            best_val_metric = cur
            best_epoch = epoch
            model.save_pretrained(os.path.join(args.output_dir, "best_adapter"))

    log_fh.close()

    # Final test eval using last-epoch weights (best is also saved separately)
    test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, device, spec.is_classification, return_preds=True)
    print(f"[test] {test_metrics}")

    # Persist per-sample test predictions BEFORE anything else: this is what
    # makes the run comparable to other models later (paired DeLong), and it
    # survives deletion of the adapter.
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    if spec.is_classification:
        score = torch.softmax(torch.from_numpy(test_preds), dim=-1).numpy()[:, 1]
    else:
        score = test_preds.reshape(-1)
    pd.DataFrame({
        spec.seq_col: test_df[spec.seq_col].to_numpy(),
        "label": test_labels,
        "score": score,
    }).to_csv(pred_path, index=False)
    print(f"[test] per-sample predictions -> {pred_path}")

    # Save run config + final adapter
    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_metric": best_val_metric,
                "best_metric_key": best_metric_key,
                # Recorded so downstream reporting never has to hardcode it:
                # the count depends on lora_rank and the base model.
                "trainable_params": trainable,
                "total_params": total,
                "test_metrics": test_metrics,
                "label_summary": {
                    "train": label_summary(train_df, spec),
                    "val": label_summary(val_df, spec),
                    "test": label_summary(test_df, spec),
                },
            },
            f,
            indent=2,
        )
    print(f"[done] best epoch={best_epoch} {best_metric_key}={best_val_metric}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
