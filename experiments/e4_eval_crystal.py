#!/usr/bin/env python3
"""
E4.0 crystal-arm evaluation — does AntiFold (crystal structure IF) beat counting?

Runs in the `plm` env on the AntiFold crystal scores produced (in the `antifold`
env) by structures/score_antifold_crystal.py. Puts AntiFold-crystal into the E1
improver-retrieval table alongside the zero-shot sequence scorers and the
mutation-count baseline, on both the strict (n=5) and near (n=42) tiers.

Prerequisite: results/absci_antifold_crystal.csv (from the antifold env).

Run:
    python experiments/e4_eval_crystal.py
Outputs:
    results/absci_e4_crystal_metrics.csv, stdout table.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import (  # noqa: E402
    evaluate_improver_retrieval, near_improver_mask,
)

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
ANTIFOLD_CSV = os.path.join(PROJECT_ROOT, "results", "absci_antifold_crystal.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e4_crystal_metrics.csv")

FITNESS_COL = "neg_log_Kd"
KS = (10, 30, 100)
MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("ESM-2", "ESM2_score"),
    ("AbLang", "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
    ("AntiFold-crystal", "antifold_crystal_score"),
]


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    df["MutCount_score"] = -df["NumMutations"].astype(float)

    af = pd.read_csv(ANTIFOLD_CSV)
    if af["HCDR3"].duplicated().any() or df["HCDR3"].duplicated().any():
        # Fall back to positional join (both written in absci_all_scored order).
        if len(af) != len(df):
            raise SystemExit("AntiFold rows != pool rows; cannot align")
        df["antifold_crystal_score"] = af["antifold_crystal_score"].values
    else:
        df = df.merge(af[["HCDR3", "antifold_crystal_score"]], on="HCDR3",
                      how="left", validate="one_to_one")
    if df["antifold_crystal_score"].isna().any():
        raise SystemExit("Missing AntiFold scores after join")

    fit = df[FITNESS_COL].values
    parent = float(df.loc[df["NumMutations"] == 0, FITNESS_COL].iloc[0])
    near = near_improver_mask(fit, top_frac=0.10)
    n_strict = int(np.nansum(fit > parent))
    n_near = int(near.sum())
    print(f"Absci HER2: n={len(df)}, strict_improvers={n_strict}, "
          f"near_improvers={n_near}, parent neg_log_Kd={parent:.4f}\n")

    rows = []
    for tier, pmask in [("strict", None), ("near", near)]:
        for name, col in MODELS:
            r = evaluate_improver_retrieval(df[col].values, fit, parent, ks=KS,
                                            positive_mask=pmask)
            rows.append({
                "tier": tier, "Model": name, "median_rank": r["median_rank"],
                "P@30": r["P@30"], "Recall@100": r["Recall@100"],
                "MW_p_enrich": r["MW_p_enrich"], "MW_p_anti": r["MW_p_anti"],
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    for tier in ("strict", "near"):
        sub = res[res["tier"] == tier].drop(columns="tier")
        n = n_strict if tier == "strict" else n_near
        print(f"=== {tier.upper()} tier (n={n}) ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.4g}",
                               "display.width", 200):
            print(sub.to_string(index=False))
        print()
    print(f"Saved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
