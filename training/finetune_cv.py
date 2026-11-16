#!/usr/bin/env python3
"""
Cross-fitted LoRA scores for every variant, so improver retrieval is testable.

One held-out Absci split contains 2 strict and 7 near improvers — far too few to
answer the question this benchmark exists to ask: *does fine-tuning fix the
anti-prediction of improvers?* A binder-classification AUROC on 124 variants does
not answer it either; binder retrieval and improver retrieval are different tasks,
which is the whole point of finding 1.

K-fold cross-fitting scores every variant with a model that never saw it, putting
all 5 strict and 42 near improvers in play at once. The scores are out-of-fold, so
they are honest held-out predictions — but they come from K *different* models, so
they are a property of the training procedure, not of one fitted model. That is the
right object here: the claim under test is about fine-tuning as a method.

Run (lora env):
    python training/finetune_cv.py --dataset absci --folds 5 --epochs 5

Output: results/<ds>/<ds>_finetune_oof.csv — sequence, out-of-fold score, fold.
Evaluate it with experiments/e5_finetune_improver.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from evaluation.paths import result_path  # noqa: E402
from training.dataloader import CDRH3Dataset, load_dataframe  # noqa: E402
from training.train_lora import (  # noqa: E402
    DEFAULT_MODEL, build_model, evaluate, pick_device, set_seed)


def parse_args() -> argparse.Namespace:
    """Command-line arguments for cross-fitted fine-tuning."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset", choices=["mason", "absci"], required=True)
    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p.parse_args()


def main() -> int:
    """Train one model per fold and write out-of-fold scores for every variant."""
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    is_cls = args.task == "classification"

    df, spec = load_dataframe(args.dataset, args.task)
    print(f"[data] {args.dataset}/{args.task}: n={len(df)}")
    y = df[spec.label_col].to_numpy()
    strata = y if is_cls else pd.qcut(y, 4, labels=False, duplicates="drop").to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    oof = np.full(len(df), np.nan)
    fold_of = np.full(len(df), -1, dtype=int)

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(df)), strata), start=1):
        set_seed(args.seed + k)          # fresh init per fold, reproducible
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        te_df = df.iloc[te_idx].reset_index(drop=True)
        model = build_model(args.model_name, 2 if is_cls else 1,
                            "single_label_classification" if is_cls else "regression",
                            args.lora_rank, args.lora_alpha, args.lora_dropout).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr)
        tr_loader = DataLoader(CDRH3Dataset(tr_df, tokenizer, spec, args.max_length),
                               batch_size=args.batch_size, shuffle=True,
                               collate_fn=collator)
        te_loader = DataLoader(CDRH3Dataset(te_df, tokenizer, spec, args.max_length),
                               batch_size=args.batch_size, shuffle=False,
                               collate_fn=collator)
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for batch in tr_loader:
                labels = batch["labels"].to(device)
                inputs = {kk: v.to(device) for kk, v in batch.items() if kk != "labels"}
                out = model(**inputs, labels=labels)
                opt.zero_grad()
                out.loss.backward()
                opt.step()
                tot += float(out.loss.detach().cpu()) * labels.shape[0]
            print(f"   fold {k}/{args.folds} epoch {ep+1}/{args.epochs} "
                  f"train_loss={tot/len(tr_df):.4f}", flush=True)
        metrics, preds, _ = evaluate(model, te_loader, device, is_cls, return_preds=True)
        # Higher = predicted fitter, matching every other scorer in the benchmark.
        score = (preds[:, 1] - preds[:, 0]) if is_cls else preds.ravel()
        oof[te_idx] = score
        fold_of[te_idx] = k
        print(f"[fold {k}] held-out {len(te_idx)}: "
              + " ".join(f"{m}={v:.4f}" for m, v in metrics.items()), flush=True)
        del model, opt
        if device == "mps":
            torch.mps.empty_cache()

    assert not np.isnan(oof).any(), "some variant never landed in a held-out fold"
    out = pd.DataFrame({spec.seq_col: df[spec.seq_col].values,
                        "oof_score": oof, "fold": fold_of})
    path = result_path(f"{args.dataset}_finetune_oof.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} out-of-fold scores -> "
          f"{os.path.relpath(path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
