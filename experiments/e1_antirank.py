#!/usr/bin/env python3
"""
E1 -- Reproduce anti-ranking through the improver-retrieval lens.

Hypothesis (pre-registered, entries/2026-07-01.md): on Absci HER2, zero-shot
PLMs rank the 5 improvers BELOW the pool median (anti-ranking), while the
trivial mutation-count baseline enriches them into the top-K. This reframes
the v2/v3 anti-prediction finding using the v4 metric suite.

Run:
    python experiments/e1_antirank.py

Outputs:
    results/absci_improver_metrics.csv   (the locked metric table)
    stdout table
No model is run; this reads the existing scored CSV (results/absci_all_scored.csv).
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import (  # noqa: E402
    evaluate_improver_retrieval,
    random_baseline_metrics,
    near_improver_mask,
)

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_improver_metrics.csv")
OUT_CSV_NEAR = os.path.join(PROJECT_ROOT, "results",
                            "absci_near_improver_metrics.csv")

FITNESS_COL = "neg_log_Kd"
MUTATION_COL = "NumMutations"
KS = (10, 30, 100)

# (display name, score column). MutCount is synthesized below.
MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("ESM-2", "ESM2_score"),
    ("AbLang", "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
]


def main() -> None:
    df = pd.read_csv(SCORED_CSV)

    # MutCount scorer: fewer mutations = higher score (closer to parent).
    df["MutCount_score"] = -df[MUTATION_COL].astype(float)

    # Parent = the NumMutations==0 row; resolve its fitness at runtime so every
    # model shares the exact same threshold (avoids the rounding bug documented
    # in reevaluate.resolve_wt_fitness).
    wt_rows = df[df[MUTATION_COL] == 0]
    if len(wt_rows) != 1:
        raise ValueError(f"Expected exactly one WT row; found {len(wt_rows)}")
    parent_fitness = float(wt_rows[FITNESS_COL].iloc[0])

    fit = df[FITNESS_COL].values
    n_total = len(df)
    n_improvers = int(np.nansum(fit > parent_fitness))

    # Secondary "near-improver" tier: top-decile neg_log_Kd among binders.
    near_mask = near_improver_mask(fit, top_frac=0.10)
    n_near = int(near_mask.sum())
    print(f"Absci HER2: n_total={n_total}, parent neg_log_Kd={parent_fitness:.4f} "
          f"(Kd=1.94 nM)")
    print(f"  strict improvers (Kd < parent): {n_improvers}")
    print(f"  near-improvers (top-decile neg_log_Kd among binders): {n_near}\n")

    _run_tier(df, fit, parent_fitness, positive_mask=None,
              n_pos=n_improvers, n_total=n_total, out_csv=OUT_CSV,
              tier="STRICT improvers (Kd < parent; n=%d)" % n_improvers)
    print()
    _run_tier(df, fit, parent_fitness, positive_mask=near_mask,
              n_pos=n_near, n_total=n_total, out_csv=OUT_CSV_NEAR,
              tier="NEAR-improvers (top-decile binders; n=%d) [secondary]" % n_near)


def _run_tier(df, fit, parent_fitness, positive_mask, n_pos, n_total, out_csv,
              tier) -> None:
    """Compute + save + print the metric suite for one positive-set tier."""
    print(f"=== {tier} ===")
    rows = []
    for name, col in MODELS:
        if col not in df.columns:
            print(f"  [skip] {name}: column {col} absent")
            continue
        r = evaluate_improver_retrieval(df[col].values, fit, parent_fitness,
                                        ks=KS, positive_mask=positive_mask)
        r["Model"] = name
        rows.append(r)

    # Random null baseline (chance level for every K, for this positive set).
    rnd = random_baseline_metrics(fit, parent_fitness, ks=KS,
                                  positive_mask=positive_mask)
    rnd_row = {"Model": "Random", "median_rank": n_total / 2.0,
               "MW_p_enrich": np.nan, "MW_p_enrich_shuf": np.nan,
               "MW_p_anti": np.nan, "improver_midranks": [],
               "improver_tie_ranges": [], "n_improvers": n_pos,
               "n_total": n_total}
    for k in KS:
        rnd_row[f"P@{k}"] = rnd[f"P@{k}"]
        rnd_row[f"Recall@{k}"] = rnd[f"Recall@{k}"]
        rnd_row[f"Fold@{k}"] = np.nan
        rnd_row[f"P@{k}_ties"] = np.nan
    rows.append(rnd_row)

    res = pd.DataFrame(rows)

    # P@K_ties kept so the tie structure driving tiebreak-averaged P@K is
    # auditable from the artifact. MW_p_enrich = tie-corrected (midrank);
    # MW_p_enrich_shuf = shuffle-mean cross-check (DEC-007).
    lead = ["Model", "n_improvers", "median_rank"]
    kcols = []
    for k in KS:
        kcols += [f"P@{k}", f"Recall@{k}", f"Fold@{k}", f"P@{k}_ties"]
    tail = ["MW_p_enrich", "MW_p_enrich_shuf", "MW_p_anti",
            "improver_midranks", "improver_tie_ranges", "n_total"]
    res = res[[c for c in lead + kcols + tail if c in res.columns]]
    res.to_csv(out_csv, index=False)

    show = res.drop(columns=["improver_midranks", "improver_tie_ranges",
                             "n_total"])
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 220,
                           "display.max_columns", 40):
        print(show.to_string(index=False))
    print(f"Saved -> {os.path.relpath(out_csv, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
