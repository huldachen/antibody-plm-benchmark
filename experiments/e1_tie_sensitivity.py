#!/usr/bin/env python3
"""
E1 tie-sensitivity diagnostic — the reproducible evidence behind DEC-007.

Records, for the beat-parent Mann-Whitney enrichment on Absci HER2, the three
ways of handling the heavy mutation-count ties:

  (1) deterministic  — MW on stable-mergesort positional ranks (the OLD method;
                       hands MutCount's NumMut=2 improvers the luckiest tie
                       assignment, ranks 3-4).
  (2) midrank        — MW on raw scores, so scipy applies its midrank tie
                       correction. The canonical, principled number (DEC-007).
  (3) tie-shuffle    — 20 noise-broken rankings, MW on positional ranks each;
                       report mean / min / max / fraction < 0.05.

(2) and (3) must agree (~0.001); the point is that both differ from (1), and for
MutCount the corrected enrichment crosses back over p=0.05 (0.028 -> 0.075).

Also records the exact near-improver-tier enrichment p-values (the 3e-7 etc.
reported in entries/2026-07-02.md), which were previously computed ad hoc.

Run:
    python experiments/e1_tie_sensitivity.py
Output:
    results/absci_e1_tie_sensitivity.csv  + stdout tables
No model is run; reads results/absci_all_scored.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import (  # noqa: E402
    improver_mask, near_improver_mask,
)

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e1_tie_sensitivity.csv")

MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("ESM-2", "ESM2_score"),
    ("AbLang", "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
]
N_SHUFFLES = 20
SEED = 42


def three_way_enrichment(s, is_pos, n_shuffles=N_SHUFFLES, seed=SEED):
    """Deterministic, midrank, and tie-shuffle enrichment p for one scorer."""
    n = s.size
    # (1) deterministic positional ranks (stable mergesort; rank 1 = best).
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    _, p_det = mannwhitneyu(ranks[is_pos], ranks[~is_pos], alternative="less")

    # (2) midrank: MW on raw scores (scipy midrank tie correction).
    _, p_mid = mannwhitneyu(s[is_pos], s[~is_pos], alternative="greater")

    # (3) tie-shuffle distribution.
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(n_shuffles):
        noise = rng.uniform(-1e-9, 1e-9, size=n)
        o = np.argsort(-(s + noise), kind="mergesort")
        r = np.empty(n, dtype=int)
        r[o] = np.arange(1, n + 1)
        _, pp = mannwhitneyu(r[is_pos], r[~is_pos], alternative="less")
        ps.append(pp)
    ps = np.asarray(ps)
    return {
        "p_deterministic": float(p_det),
        "p_midrank": float(p_mid),
        "p_shuffle_mean": float(ps.mean()),
        "p_shuffle_min": float(ps.min()),
        "p_shuffle_max": float(ps.max()),
        "shuffle_frac_sig": float(np.mean(ps < 0.05)),
    }


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    df["MutCount_score"] = -df["NumMutations"].astype(float)
    fit = df["neg_log_Kd"].values.astype(float)
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])

    strict = improver_mask(fit, parent)
    near = near_improver_mask(fit, top_frac=0.10)

    rows = []
    print(f"Absci HER2: n={len(df)}, strict_improvers={int(strict.sum())}, "
          f"near_improvers={int(near.sum())}\n")
    print("STRICT tier — beat-parent MW enrichment p, three tie treatments:")
    print(f"{'model':<10}{'determ':>9}{'midrank':>9}{'shuf_mean':>10}"
          f"{'shuf_min':>9}{'shuf_max':>9}{'%<.05':>7}   {'near_p':>9}")
    for name, col in MODELS:
        s = df[col].values.astype(float)
        strict_res = three_way_enrichment(s, strict)
        # near-improver exact enrichment p (midrank on scores).
        _, near_p = mannwhitneyu(s[near], s[~near], alternative="greater")
        row = {"Model": name, **{f"strict_{k}": v for k, v in strict_res.items()},
               "near_p_enrich": float(near_p)}
        rows.append(row)
        print(f"{name:<10}{strict_res['p_deterministic']:>9.4f}"
              f"{strict_res['p_midrank']:>9.4f}{strict_res['p_shuffle_mean']:>10.4f}"
              f"{strict_res['p_shuffle_min']:>9.4f}{strict_res['p_shuffle_max']:>9.4f}"
              f"{100*strict_res['shuffle_frac_sig']:>6.0f}%   {near_p:>9.2e}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nKey point (DEC-007): MutCount strict enrichment goes "
          f"{rows[0]['strict_p_deterministic']:.3f} (determ, artifact) -> "
          f"{rows[0]['strict_p_midrank']:.3f} (corrected) — crosses 0.05.")
    print(f"Saved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
