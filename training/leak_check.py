"""
Diagnostic: place a fine-tuned PLM result in context by checking for
data-leakage modes and benchmarking trivial / non-linear baselines on the
*same* train/val/test split (seed 42, identical to train_lora.py).

Usage:
    python training/leak_check.py --dataset mason
    python training/leak_check.py --dataset absci

Checks (per dataset):
  1. Exact-duplicate CDR-H3 sequences across train/val/test (basic correctness).
  2. Hamming-distance distribution from each test seq to its nearest train seq
     (near-duplicate leakage).
  3. Logistic regression on [NumMutations] alone — does mutation count carry
     most of the signal?
  4. Logistic regression on flat one-hot CDR-H3 — linear baseline.
  5. MLP on the same one-hot — non-linear baseline (controls for the
     possibility that the 0.90 ceiling is a linear-model artefact).

Output: prints a comparison table to stdout. No GPU needed; runs in seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from evaluation.paths import result_path  # noqa: E402
from training.dataloader import (  # noqa: E402
    load_dataframe, stratified_split, distance_split)


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA2INT = {a: i for i, a in enumerate(AA_VOCAB)}


def encode_int(seqs: list[str], length: int) -> np.ndarray:
    """Encode list of equal-length AA strings as (N, length) int array."""
    out = np.zeros((len(seqs), length), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            out[i, j] = AA2INT.get(c, -1)
    return out


def encode_onehot(seqs: list[str], length: int) -> np.ndarray:
    """Encode list of equal-length AA strings as (N, length*20) one-hot."""
    int_enc = encode_int(seqs, length)  # (N, L)
    n = int_enc.shape[0]
    oh = np.zeros((n, length * 20), dtype=np.float32)
    rows = np.repeat(np.arange(n), length)
    cols = (np.arange(length) * 20)[None, :].repeat(n, axis=0).ravel() + int_enc.ravel()
    valid = int_enc.ravel() >= 0
    oh[rows[valid], cols[valid]] = 1.0
    return oh


def hamming_min_to_train(test_int: np.ndarray, train_int: np.ndarray) -> np.ndarray:
    """For each test row, return the min Hamming distance to any train row.

    Vectorised: chunk test rows to keep memory bounded.
    """
    n_test = test_int.shape[0]
    out = np.zeros(n_test, dtype=np.int32)
    chunk = 256
    for i in range(0, n_test, chunk):
        block = test_int[i:i + chunk]                          # (b, L)
        # broadcast (b, 1, L) vs (1, T, L), sum mismatches across L
        diffs = (block[:, None, :] != train_int[None, :, :]).sum(axis=2)  # (b, T)
        out[i:i + chunk] = diffs.min(axis=1)
    return out


def fit_and_score(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict:
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "probs": probs,
        "label": label,
        "n_features": X_train.shape[1],
        "roc_auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "mcc": matthews_corrcoef(y_test, preds),
    }


def _trainable_params(run_dir: str, cfg: dict) -> int | None:
    """Trainable-parameter count for a run — read, never hardcoded.

    The count depends on `lora_rank` and the base model, so a literal would go
    silently wrong the moment either changes. Prefers the value recorded by
    `train_lora.py`; falls back to parsing `run.log` for runs that predate that
    field. Returns None when neither is available, so a missing count shows up
    as missing rather than as a wrong number.

    Parameters
    ----------
    run_dir : str
        Checkpoint directory for the run.
    cfg : dict
        Parsed `config.json` for the run.

    Returns
    -------
    int or None
        Trainable parameters, or None if not recoverable.
    """
    n = cfg.get("trainable_params")
    if isinstance(n, int):
        return n
    log_path = os.path.join(run_dir, "run.log")
    if os.path.isfile(log_path):
        with open(log_path, errors="ignore") as fh:
            m = re.search(r"trainable=([\d,]+)", fh.read())
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def lora_test_metrics(dataset: str, test_seqs=None, seq_col=None) -> dict | None:
    """Read LoRA test metrics from the most recent matching checkpoint dir."""
    base = os.path.join(os.path.dirname(__file__), "checkpoints")
    if not os.path.isdir(base):
        return None
    candidates = sorted(
        d for d in os.listdir(base)
        if d.startswith(f"{dataset}_cls_") and os.path.isfile(os.path.join(base, d, "config.json"))
    )
    if not candidates:
        return None
    # Several runs may exist under different splits. Picking sorted()[-1] once
    # reported a random-split AUC beside distance-split baselines, because
    # "_rerun" sorts after "_dist2". Select the run whose SAVED PREDICTIONS match
    # the test set actually in hand; anything else is not comparable.
    if test_seqs is not None:
        want = set(test_seqs)
        matched = None
        for d in candidates:
            pf = os.path.join(base, d, "test_predictions.csv")
            if os.path.isfile(pf):
                col = pd.read_csv(pf, nrows=0).columns[0]
                if set(pd.read_csv(pf)[col]) == want:
                    matched = d
                    break
        if matched is None:
            return None
        candidates = [matched]
    run_dir = os.path.join(base, candidates[-1])
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    args = cfg.get("args", {})
    tm = cfg.get("test_metrics", {})
    return {
        "label": f"ESM-2 LoRA r={args.get('lora_rank')} lr={args.get('lr'):g} ({candidates[-1]})",
        "n_features": _trainable_params(run_dir, cfg),
        "roc_auc": tm.get("roc_auc"),
        "accuracy": tm.get("accuracy"),
        "mcc": tm.get("mcc"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Leak / baseline diagnostic for v3 fine-tuning")
    p.add_argument("--dataset", choices=["mason", "absci"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["random", "distance"], default="random",
                   help="random stratified (optimistic) or leave-cluster-out")
    p.add_argument("--d-link", type=int, default=2,
                   help="Hamming linkage threshold for --split distance")
    args = p.parse_args()

    print(f"[load] {args.dataset}/classification, same seed as training ({args.seed})")
    df, spec = load_dataframe(args.dataset, "classification")
    if args.split == "distance":
        train_df, val_df, test_df = distance_split(
            df, spec, d_link=args.d_link, seed=args.seed)
    else:
        train_df, val_df, test_df = stratified_split(df, spec, seed=args.seed)
    print(f"       train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    seq_col = spec.seq_col
    label_col = spec.label_col
    cdr_len = len(train_df[seq_col].iloc[0])
    print(f"       seq_col={seq_col}  label_col={label_col}  CDR-H3 length={cdr_len}")

    # ---- 1. Exact-duplicate check across splits ----
    train_set = set(train_df[seq_col])
    val_set = set(val_df[seq_col])
    test_set = set(test_df[seq_col])
    cnt_train = Counter(train_df[seq_col])
    cnt_test = Counter(test_df[seq_col])
    n_dup_train_internal = sum(c - 1 for c in cnt_train.values() if c > 1)
    n_dup_test_internal = sum(c - 1 for c in cnt_test.values() if c > 1)
    print("\n[check 1] exact-duplicate CDR-H3 sequences across splits")
    print(f"       unique seqs    train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    print(f"       intra-split duplicates  train={n_dup_train_internal}  test={n_dup_test_internal}")
    print(f"       train ∩ val   = {len(train_set & val_set)}")
    print(f"       train ∩ test  = {len(train_set & test_set)}")
    print(f"       val   ∩ test  = {len(val_set & test_set)}")
    leak_total = len(train_set & test_set)
    print(f"       LEAK CHECK: train∩test = {leak_total} {'(CLEAN)' if leak_total == 0 else '(LEAK!)'}")

    # ---- 2. Nearest-neighbour Hamming distance from test to train ----
    print("\n[check 2] for each test seq, Hamming distance to nearest train seq")
    train_int = encode_int(train_df[seq_col].tolist(), cdr_len)
    test_int = encode_int(test_df[seq_col].tolist(), cdr_len)
    min_d = hamming_min_to_train(test_int, train_int)
    bins = np.bincount(min_d, minlength=cdr_len + 1)
    for d, c in enumerate(bins):
        if c > 0:
            pct = 100 * c / len(min_d)
            print(f"       d={d:2d} : {c:5d} test seqs ({pct:5.1f}%)")
    print(f"       median nearest-train Hamming = {np.median(min_d):.1f}")
    print(f"       fraction of test with d<=1 from a train seq: {(min_d <= 1).mean():.3f}")

    # ---- 3. LR on NumMutations alone ----
    print("\n[check 3] logistic regression on [NumMutations] alone")
    X_train_mut = train_df[["NumMutations"]].to_numpy(dtype=np.float32)
    X_test_mut = test_df[["NumMutations"]].to_numpy(dtype=np.float32)
    y_train = train_df[label_col].to_numpy(dtype=np.int64)
    y_test = test_df[label_col].to_numpy(dtype=np.int64)
    res_mut = fit_and_score(
        LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        X_train_mut, y_train, X_test_mut, y_test, "LR(MutCount)",
    )
    print(f"       {res_mut}")

    # ---- 4. LR on one-hot CDR-H3 ----
    n_oh = cdr_len * 20
    print(f"\n[check 4] logistic regression on one-hot CDR-H3 ({n_oh} features)")
    X_train_oh = encode_onehot(train_df[seq_col].tolist(), cdr_len)
    X_test_oh = encode_onehot(test_df[seq_col].tolist(), cdr_len)
    res_oh = fit_and_score(
        LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        X_train_oh, y_train, X_test_oh, y_test, "LR(one-hot)",
    )
    print(f"       {res_oh}")

    # ---- 5. MLP on one-hot CDR-H3 (non-linear baseline) ----
    print(f"\n[check 5] MLP(128) on one-hot CDR-H3 (non-linear baseline)")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        max_iter=200,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    res_mlp = fit_and_score(mlp, X_train_oh, y_train, X_test_oh, y_test, "MLP(128) one-hot")
    print(f"       {res_mlp}  (n_iter={mlp.n_iter_})")

    # ---- LoRA result if checkpoint exists ----
    res_lora = lora_test_metrics(args.dataset, test_df[seq_col], seq_col)
    if res_lora is None:
        print(f"\n[lora] no checkpoint whose saved predictions match this "
              f"'{args.split}' split - excluding the LoRA row")

    # ---- Summary ----
    title = f"SUMMARY — {args.dataset} classification, test n={len(test_df)}, seed={args.seed}"
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    print(f"{'Method':<40s}  {'features':>10s}  {'ROC-AUC':>8s}  {'acc':>6s}  {'mcc':>6s}")
    for r in (res_mut, res_oh, res_mlp):
        print(f"{r['label']:<40s}  {r['n_features']:>10d}  "
              f"{r['roc_auc']:>8.4f}  {r['accuracy']:>6.4f}  {r['mcc']:>6.4f}")
    if res_lora is not None:
        nf = res_lora["n_features"]
        nf_s = f"{nf:,}" if nf is not None else "unknown"
        print(f"{res_lora['label']:<40s}  {nf_s:>10s}  "
              f"{res_lora['roc_auc']:>8.4f}  {res_lora['accuracy']:>6.4f}  {res_lora['mcc']:>6.4f}")
    else:
        print(f"{'(no LoRA checkpoint for this dataset yet)':<40s}")

    # ---- Paired DeLong vs the LoRA run, when its predictions exist ----
    # Per-model intervals do not test a difference between two correlated AUCs,
    # and every model here is scored on the SAME test split. This needs only the
    # saved test_predictions.csv, not the adapter weights -- which is the point
    # of saving them.
    delong_rows = []
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    pred_file = None
    if os.path.isdir(ckpt_dir):
        want = set(test_df[seq_col])
        for d in sorted(os.listdir(ckpt_dir)):
            pf = os.path.join(ckpt_dir, d, "test_predictions.csv")
            if d.startswith(f"{args.dataset}_cls_") and os.path.isfile(pf):
                col = pd.read_csv(pf, nrows=0).columns[0]
                if set(pd.read_csv(pf)[col]) == want:
                    pred_file = pf
                    break

    if pred_file is None:
        print("\n[delong] no saved LoRA test_predictions.csv - skipping paired tests")
    else:
        from evaluation.delong import delong_test
        lora = pd.read_csv(pred_file).rename(columns={"score": "lora_score"})
        frame = pd.DataFrame({seq_col: test_df[seq_col].to_numpy(),
                              "y": test_df[label_col].to_numpy(int),
                              "LR(MutCount)": res_mut["probs"],
                              "LR(one-hot)": res_oh["probs"],
                              "MLP(128) one-hot": res_mlp["probs"]})
        merged = frame.merge(lora[[lora.columns[0], "lora_score"]],
                             left_on=seq_col, right_on=lora.columns[0], how="inner")
        if len(merged) != len(frame):
            print(f"\n[delong] WARNING aligned {len(merged)}/{len(frame)} rows - "
                  f"split mismatch, skipping")
        else:
            y = merged["y"].to_numpy(int)
            lora_s = merged["lora_score"].to_numpy(float)
            print(f"\n[delong] LoRA vs each baseline (paired, n={len(y)}, "
                  f"n_pos={int(y.sum())}):")
            for name in ("LR(MutCount)", "LR(one-hot)", "MLP(128) one-hot"):
                d = delong_test(y, lora_s, merged[name].to_numpy(float))
                sig = ("***" if d["p"] < 1e-3 else "**" if d["p"] < 0.01
                       else "*" if d["p"] < 0.05 else "n.s.")
                delong_rows.append({"comparison": f"LoRA vs {name}",
                                    "auc_lora": d["auc_a"], "auc_other": d["auc_b"],
                                    "delta": d["delta"], "se_delta": d["se_delta"],
                                    "p_delong": d["p"], "significant_95": d["p"] < 0.05})
                print(f"    LoRA {d['auc_a']:.4f}  vs  {name:<20} {d['auc_b']:.4f}   "
                      f"delta={d['delta']:+.4f}  se={d['se_delta']:.4f}  "
                      f"p={d['p']:.3e}  {sig}")
            pd.DataFrame(delong_rows).to_csv(
                result_path(f"{args.dataset}_finetune_delong"
                f"{'' if args.split=='random' else '_distance'}.csv"), index=False)

    # ---- Persist: no headline number may live only in stdout (DEC-012 rule 3).
    # Every row carries the split diagnostics, so the metric can never be read
    # apart from the caveat that earns it: this is a RANDOM stratified split, the
    # optimistic regime this benchmark exists to warn about.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    common = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split_type": ("random_stratified" if args.split == "random"
                       else f"distance_aware_d{args.d_link}"),
        "train_inter_test": leak_total,
        "median_nearest_train_hamming": float(np.median(min_d)),
        "frac_test_d_le_1": float((min_d <= 1).mean()),
    }
    scored = [res_mut, res_oh, res_mlp] + ([res_lora] if res_lora is not None else [])
    out_rows = [
        {**common, "method": r["label"], "n_features": r["n_features"],
         "roc_auc": r["roc_auc"], "accuracy": r["accuracy"], "mcc": r["mcc"]}
        for r in scored
    ]
    out_csv = result_path(f"{args.dataset}_finetune_baselines"
                               f"{'' if args.split=='random' else '_distance'}.csv")
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_rows)} rows -> "
          f"{os.path.relpath(out_csv, project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
