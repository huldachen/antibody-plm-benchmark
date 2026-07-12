#!/usr/bin/env python3
"""
E4.0 mechanism — is AntiFold-crystal just a mutation-count proxy?

The crystal backbone was solved for the WT sequence, so AntiFold's inverse-
folding likelihood should favour WT-backbone-compatible (low-mutation) sequences
— the same mutation-count/germline confound seen throughout the benchmark. This
tests it directly: partial Spearman of each scorer vs fitness (neg_log_Kd) among
binders, controlling for mutation count. If the partial correlation collapses to
~0, the scorer adds nothing beyond counting.

Run (plm env):  python experiments/e4_partial_correlation.py
Output: results/absci_e4_partial_corr.csv, stdout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
ANTIFOLD_CSV = os.path.join(PROJECT_ROOT, "results", "absci_antifold_crystal.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e4_partial_corr.csv")

MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("AntiFold-crystal", "antifold_crystal_score"),
    ("ESM-2", "ESM2_score"),
]


def _resid_on(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of a after linear regression on b (with intercept)."""
    B = np.c_[np.ones_like(b), b]
    coef, _, _, _ = np.linalg.lstsq(B, a, rcond=None)
    return a - B @ coef


def partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z = Pearson on rank-residuals."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex, ey = _resid_on(rx, rz), _resid_on(ry, rz)
    r, p = pearsonr(ex, ey)
    return r, p


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    df["MutCount_score"] = -df["NumMutations"].astype(float)
    af = pd.read_csv(ANTIFOLD_CSV)
    df["antifold_crystal_score"] = af["antifold_crystal_score"].values

    # Binders only (fitness defined). Control = mutation count.
    b = df[df["neg_log_Kd"].notna()].copy()
    fit = b["neg_log_Kd"].values
    mut = b["NumMutations"].values.astype(float)
    print(f"Binders with measured Kd: n={len(b)}\n")

    rows = []
    for name, col in MODELS:
        s = b[col].values.astype(float)
        raw_r, raw_p = spearmanr(s, fit)
        par_r, par_p = partial_spearman(s, fit, mut)
        conf_r, _ = spearmanr(s, mut)         # scorer vs mutation count
        rows.append({"Model": name, "raw_rho": raw_r, "raw_p": raw_p,
                     "partial_rho_ctrl_mut": par_r, "partial_p": par_p,
                     "rho_vs_mutcount": conf_r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}",
                           "display.width", 200):
        print(res.to_string(index=False))
    print("\nReading: partial_rho_ctrl_mut ~ 0 => scorer adds nothing beyond "
          "mutation count. rho_vs_mutcount near +/-1 => scorer IS ~ a mutation "
          "counter.")
    print(f"Saved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
