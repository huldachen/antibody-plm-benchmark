#!/usr/bin/env python3
"""
E5 — does fine-tuning fix the anti-prediction of improvers?

This is the question the benchmark exists to ask, and a binder-classification
AUROC does not answer it: binder retrieval and improver retrieval are different
tasks, which is finding 1. A single held-out Absci split holds 2 strict and 7
near improvers, so the test has to be cross-fitted — `training/finetune_cv.py`
writes an out-of-fold LoRA score for every variant, putting all 5 strict and 42
near improvers in play.

Fairness matters more than usual here. Comparing a *supervised* fine-tune against
*unsupervised* zero-shot scores would be rigged, so a logistic regression on
one-hot CDR-H3 is cross-fitted on the SAME folds (read from the OOF file, not
re-drawn) and carries the real argument. Mutation count needs no fitting and is
the floor everything must clear.

Run (plm env): python experiments/e5_finetune_improver.py
Output: results/absci/absci_finetune_improver.csv, stdout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from evaluation.paths import result_path  # noqa: E402
from evaluation.delong import delong_test  # noqa: E402
from evaluation.improver_metrics import (  # noqa: E402
    improver_mask, near_improver_mask, beat_parent_enrichment)
from experiments.absci_fewshot_common import one_hot  # noqa: E402
from experiments.e1c_label_definition import auroc_ci  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main() -> int:
    """Score the cross-fitted fine-tune against cross-fitted and trivial baselines."""
    df = pd.read_csv(result_path("absci_all_scored.csv"))
    oof = pd.read_csv(result_path("absci_finetune_oof.csv"))

    # The fine-tune sees HCDR3 only, so its scores key on HCDR3. Every strict and
    # near improver varies in HCDR3 alone (0/5 and 0/42 vary outside it), so the
    # join cannot drop an improver -- checked below rather than assumed.
    merged = df.merge(oof, on="HCDR3", how="inner")
    fit = merged["neg_log_Kd"].values
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])
    strict, near = improver_mask(fit, parent), near_improver_mask(fit, 0.10)
    full_strict = improver_mask(df["neg_log_Kd"].values, parent)
    full_near = near_improver_mask(df["neg_log_Kd"].values, 0.10)
    print(f"[join] {len(merged)} of {len(df)} variants carry an out-of-fold score; "
          f"improvers kept: strict {strict.sum()}/{full_strict.sum()}, "
          f"near {near.sum()}/{full_near.sum()}")

    # Cross-fitted one-hot baseline, on the folds the fine-tune actually used.
    X = one_hot(merged["HCDR3"].values)
    y_binder = (merged["Binder"].values == 1).astype(int)
    lr_oof = np.full(len(merged), np.nan)
    for k in sorted(merged["fold"].unique()):
        te = merged["fold"].values == k
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[~te], y_binder[~te])
        lr_oof[te] = clf.decision_function(X[te])
    assert not np.isnan(lr_oof).any()

    scorers = {
        "ESM-2 LoRA (cross-fitted)": merged["oof_score"].values,
        "LR one-hot (cross-fitted)": lr_oof,
        "MutCount": -merged["NumMutations"].values.astype(float),
        "BLOSUM62": merged["BLOSUM62_score"].values,
        "ESM-2 zero-shot": merged["ESM2_score"].values,
        "AbLang zero-shot": merged["AbLang_score"].values,
    }

    rows = []
    print(f"\n{'scorer':<28}{'near AUROC (n=%d)' % near.sum():>26}"
          f"{'strict AUROC (n=%d)' % strict.sum():>26}{'near MW p':>12}")
    for name, sc in scorers.items():
        a_near, (lo_n, hi_n) = roc_auc_score(near, sc), auroc_ci(near, sc)
        a_str, (lo_s, hi_s) = roc_auc_score(strict, sc), auroc_ci(strict, sc)
        mw = beat_parent_enrichment(sc, fit, parent, positive_mask=near)["MW_p_enrich"]
        rows += [{"scorer": name, "tier": "near", "n_pos": int(near.sum()),
                  "auroc": a_near, "lo": lo_n, "hi": hi_n, "mw_p_enrich": mw},
                 {"scorer": name, "tier": "strict", "n_pos": int(strict.sum()),
                  "auroc": a_str, "lo": lo_s, "hi": hi_s, "mw_p_enrich": np.nan}]
        print(f"{name:<28}{a_near:>8.3f} [{lo_n:.3f},{hi_n:.3f}]"
              f"{a_str:>9.3f} [{lo_s:.3f},{hi_s:.3f}]{mw:>12.2e}")

    print("\nPaired DeLong on the near tier (same variants, same labels):")
    ref = scorers["ESM-2 LoRA (cross-fitted)"]
    for other in ("LR one-hot (cross-fitted)", "MutCount"):
        d = delong_test(near.astype(int), ref, scorers[other])
        sig = ("***" if d["p"] < 1e-3 else "**" if d["p"] < 0.01
               else "*" if d["p"] < 0.05 else "n.s.")
        rows.append({"scorer": f"LoRA vs {other}", "tier": "near_delong",
                     "n_pos": int(near.sum()), "auroc": d["delta"],
                     "lo": np.nan, "hi": np.nan, "mw_p_enrich": d["p"]})
        print(f"   LoRA {d['auc_a']:.3f} vs {other:<26} {d['auc_b']:.3f}  "
              f"delta={d['delta']:+.3f} se={d['se_delta']:.3f} p={d['p']:.3g}  {sig}")

    out = result_path("absci_finetune_improver.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {os.path.relpath(out, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
