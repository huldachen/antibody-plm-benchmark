#!/usr/bin/env python3
"""
E1b (M4) — does the zero-shot anti-ranking pattern replicate on a second assay?

Mason 2021 (trastuzumab scFv, same target, BINARY FACS binder/non-binder, no Kd)
is a second, independent assay. No Kd => no improver definition; the task is
BINDER RETRIEVAL (positive = AgClass==1, n=154 of 500). We ask only the E1
question: do zero-shot PLMs anti-rank the binders (rank them below random) while
the trivial mutation-count and BLOSUM62 baselines enrich them? A yes means the
zero-shot pattern is not an Absci artifact.

Runs in plm env on the existing results/mason_all_scored.csv (no model re-run).
Output: results/mason_binder_retrieval.csv, stdout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import (  # noqa: E402
    beat_parent_enrichment, improver_retrieval_at_k,
)

SCORED = os.path.join(PROJECT_ROOT, "results", "mason_all_scored.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "mason_binder_retrieval.csv")
KS = (30, 100)
MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("ESM-2", "ESM2_score"),
    ("AbLang", "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
]


def main() -> None:
    df = pd.read_csv(SCORED)
    df["MutCount_score"] = -df["NumMutations"].astype(float)
    binder = (df["AgClass"].values == 1)
    fit = df["LogEnrichment"].values.astype(float)   # continuous, secondary
    n = len(df)
    n_pos = int(binder.sum())
    print(f"Mason: n={n}, binders={n_pos} ({100*n_pos/n:.1f}%), "
          f"pool median rank={n/2:.0f}, random P@K≈{n_pos/n:.3f}\n")

    rows = []
    for name, col in MODELS:
        s = df[col].values.astype(float)
        enr = beat_parent_enrichment(s, fit, 0.0, ks=KS, positive_mask=binder)
        row = {"Model": name, "median_rank": enr["median_rank"],
               "MW_p_enrich": enr["MW_p_enrich"], "MW_p_anti": enr["MW_p_anti"]}
        for k in KS:
            r = improver_retrieval_at_k(s, fit, 0.0, k, positive_mask=binder)
            row[f"P@{k}"] = r["precision"]["mean"]
            row[f"Recall@{k}"] = r["recall"]["mean"]
        # secondary: Spearman vs continuous LogEnrichment
        row["spearman_logE"] = spearmanr(s, fit)[0]
        # is the enrichment a mutation-count proxy? rho vs mutation count.
        row["rho_vs_mutcount"] = spearmanr(s, df["NumMutations"].values)[0]
        rows.append(row)

    res = pd.DataFrame(rows)[["Model", "median_rank", "P@30", "Recall@100",
                              "MW_p_enrich", "MW_p_anti", "spearman_logE",
                              "rho_vs_mutcount"]]
    res.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}",
                           "display.width", 200):
        print(res.to_string(index=False))
    print("\nReading: median_rank < 250 & MW_p_enrich small => enriches binders; "
          "median_rank > 250 & MW_p_anti small => anti-ranks (below random).")
    print(f"Saved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
