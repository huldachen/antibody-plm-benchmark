#!/usr/bin/env python3
"""
E6 — does fine-tuning help on a panel that is NOT a combinatorial library?

KyDab is the generalisation check: natural antibodies from four immunisations,
not mutants of one parent. That breaks the benchmark's usual comparisons on
purpose, and the breakage is the finding worth stating:

  * there is **no parent**, so "mutation count from WT" does not exist and
    neither does the improver tier (Kd < parent). Findings 1-4 cannot be
    rerun here;
  * CDR-H3 length **varies** (8-24 residues), so the one-hot baseline that
    carries finding 4 does not apply either;
  * the four antigens are different targets. A CDR-H3-only model cannot know
    which target it is predicting for, so each antigen is fitted separately.

What remains well-posed is affinity *ranking within an antigen*. The honest
comparator for a supervised fine-tune is another supervised model on the same
folds -- here a ridge on 3-mer counts, which handles variable length -- with the
published zero-shot scores alongside for reference.

Significance uses a paired bootstrap over variants: two Spearman correlations on
the same rows are correlated, so independent intervals cannot test their
difference (the same reason finding 4 needed DeLong).

Run (plm env): python experiments/e6_kydab_finetune.py
Output: results/kydab/kydab_finetune_eval.csv, stdout.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from evaluation.paths import result_path  # noqa: E402

ANTIGENS = ["typhi-01", "pertussis"]
ZERO_SHOT = ["esm2_pll", "ablang_pll", "antiberty_pll", "ablang2_pll",
             "cdrh3_length", "random"]
N_BOOT = 2000
SEED = 0


def kmer_features(seqs, k: int = 3) -> np.ndarray:
    """Count 3-mers per sequence — a fixed-width encoding for variable length.

    One-hot by position is undefined when sequences differ in length, which is
    the case across every KyDab antigen. K-mer counts are the cheapest encoding
    that still carries local sequence identity.
    """
    vocab = sorted({seqs[i][j:j + k] for i in range(len(seqs))
                    for j in range(max(len(seqs[i]) - k + 1, 0))})
    index = {m: i for i, m in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=float)
    for r, s in enumerate(seqs):
        for m, c in Counter(s[j:j + k] for j in range(len(s) - k + 1)).items():
            X[r, index[m]] = c
    return X


def paired_bootstrap_delta(y, a, b, n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap the difference between two Spearman correlations on the same rows.

    Returns
    -------
    tuple of float
        (delta, lo, hi, two-sided p) for rho(a, y) - rho(b, y).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = spearmanr(a, y)[0] - spearmanr(b, y)[0]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 3:
            deltas[i] = np.nan
            continue
        deltas[i] = spearmanr(a[idx], y[idx])[0] - spearmanr(b[idx], y[idx])[0]
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return obs, lo, hi, min(p, 1.0)


def main() -> int:
    """Score the cross-fitted fine-tune per antigen against same-fold baselines."""
    scores = pd.read_csv(result_path("kydab_scores.csv"))
    rows = []
    for antigen in ANTIGENS:
        oof_path = result_path(f"kydab_{antigen.replace('-', '')}_finetune_oof.csv")
        if not os.path.exists(oof_path):
            print(f"[skip] {antigen}: no out-of-fold scores at {oof_path}")
            continue
        oof = pd.read_csv(oof_path)
        s = scores[scores["project"] == antigen]
        # One row per unique CDR-H3, matching the dedup the fine-tune used --
        # replicates of the same sequence must not straddle folds.
        s = s.groupby("heavy_cdr3", as_index=False).agg(
            {"neg_log_Kd": "mean", **{c: "mean" for c in ZERO_SHOT}})
        m = s.merge(oof, on="heavy_cdr3", how="inner")
        y = m["neg_log_Kd"].to_numpy()
        print(f"\n=== {antigen}: {len(m)} unique CDR-H3 with scores "
              f"(affinity sd {y.std():.2f}) ===")

        # Supervised comparator, cross-fitted on the SAME folds as the fine-tune.
        X = kmer_features(m["heavy_cdr3"].tolist())
        kmer_oof = np.full(len(m), np.nan)
        for k in sorted(m["fold"].unique()):
            te = m["fold"].to_numpy() == k
            est = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
            est.fit(X[~te], y[~te])
            kmer_oof[te] = est.predict(X[te])

        cand = {"ESM-2 LoRA (cross-fitted)": m["oof_score"].to_numpy(),
                "Ridge 3-mer (cross-fitted, same folds)": kmer_oof}
        cand.update({c: m[c].to_numpy() for c in ZERO_SHOT})

        print(f"   {'scorer':<40}{'Spearman rho':>14}{'p':>10}")
        for name, sc in cand.items():
            rho, p = spearmanr(sc, y)
            rows.append({"antigen": antigen, "n": len(m), "scorer": name,
                         "rho": rho, "p": p, "delta_vs_lora": np.nan,
                         "delta_lo": np.nan, "delta_hi": np.nan, "delta_p": np.nan})
            print(f"   {name:<40}{rho:>+14.3f}{p:>10.3g}")

        ref = cand["ESM-2 LoRA (cross-fitted)"]
        print("   paired bootstrap, LoRA minus baseline:")
        for other in ("Ridge 3-mer (cross-fitted, same folds)", "esm2_pll",
                      "cdrh3_length"):
            d, lo, hi, p = paired_bootstrap_delta(y, ref, cand[other])
            rows.append({"antigen": antigen, "n": len(m),
                         "scorer": f"LoRA vs {other}", "rho": np.nan, "p": np.nan,
                         "delta_vs_lora": d, "delta_lo": lo, "delta_hi": hi,
                         "delta_p": p})
            verdict = "n.s." if lo <= 0 <= hi else "differs"
            print(f"      vs {other:<38} delta={d:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3g}  {verdict}")

    if not rows:
        print("no antigens evaluated")
        return 1
    out = result_path("kydab_finetune_eval.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {os.path.relpath(out, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
