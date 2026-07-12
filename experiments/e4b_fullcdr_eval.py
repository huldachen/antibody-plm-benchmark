#!/usr/bin/env python3
"""
E4b + tier-3 CI + inflation factor — full-CDR corrections (plm env).

(A) AntiFold full-CDR (results/absci_antifold_fullcdr.csv): improver enrichment
    and partial correlation controlling for FULL-CDR mutation count (the earlier
    partial-corr controlled HCDR3 count only). Does "AntiFold = counter" hold?
(B) Tier-3 CI: strict Recall@100 on the full-CDR d>=3 distance split, with 95%
    CIs — is the Ridge-vs-MutCount gap within noise (demote tier 3)?
(C) Inflation factor: the corrected leaky-vs-leakage-free ratio (ρ d=1 vs d>=5).

Run (plm env): python experiments/e4b_fullcdr_eval.py
Output: results/absci_e4b_fullcdr.csv, stdout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import one_hot, ridge_scorer, SCORED_CSV  # noqa: E402
from evaluation.improver_metrics import (  # noqa: E402
    improver_mask, near_improver_mask, beat_parent_enrichment,
)
from evaluation.lowN_protocol import run_learning_curve_fixed, summarize  # noqa: E402

FULLCDR = os.path.join(PROJECT_ROOT, "results", "absci_antifold_fullcdr.csv")
CRYSTAL = os.path.join(PROJECT_ROOT, "results", "absci_antifold_crystal.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e4b_fullcdr.csv")
CDRS = ["HCDR1", "HCDR2", "HCDR3"]
WT = {"HCDR1": "GFNIKDTY", "HCDR2": "IYPTNGYT", "HCDR3": "SRWGGDGFYAMDY"}


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        B = np.c_[np.ones_like(b), b]
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return pearsonr(resid(rx, rz), resid(ry, rz))


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    df["antifold_fullcdr"] = pd.read_csv(FULLCDR)["antifold_fullcdr_score"].values
    df["antifold_crystal"] = pd.read_csv(CRYSTAL)["antifold_crystal_score"].values
    fit = df["neg_log_Kd"].values
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])
    strict, near = improver_mask(fit, parent), near_improver_mask(fit, 0.10)
    records = []  # tidy rows -> OUT_CSV (so every reported number has a committed artifact)

    # Full-CDR mutation count = total CDR Hamming from WT.
    def ham(s, w):
        return sum(a != b for a, b in zip(str(s), w)) if len(str(s)) == len(w) else 0
    full_mut = sum(df[c].apply(lambda s: ham(s, WT[c])) for c in CDRS).values.astype(float)
    df["MutCount_full"] = -full_mut
    h3_full_corr = spearmanr(df['NumMutations'], full_mut)[0]
    records.append({"section": "count_corr", "metric": "spearman_HCDR3_vs_fullCDR_mutcount",
                    "model": "", "tier": "", "N": "", "value": h3_full_corr})
    print(f"HCDR3 count vs full-CDR count: corr={h3_full_corr:.3f}\n")

    # (A) AntiFold full-CDR enrichment vs HCDR3-only + baselines.
    print("(A) Enrichment (MW p) — near tier / strict tier:")
    for name, col in [("AntiFold-crystal(HCDR3)", "antifold_crystal"),
                      ("AntiFold-fullCDR", "antifold_fullcdr"),
                      ("MutCount(HCDR3)", None), ("MutCount(fullCDR)", "MutCount_full"),
                      ("BLOSUM62", "BLOSUM62_score")]:
        s = -df["NumMutations"].values if col is None else df[col].values
        en = beat_parent_enrichment(s, fit, parent, positive_mask=near)["MW_p_enrich"]
        es = beat_parent_enrichment(s, fit, parent, positive_mask=strict)["MW_p_enrich"]
        records.append({"section": "enrichment", "metric": "MW_p_enrich",
                        "model": name, "tier": "near", "N": "", "value": en})
        records.append({"section": "enrichment", "metric": "MW_p_enrich",
                        "model": name, "tier": "strict", "N": "", "value": es})
        print(f"  {name:<24} near={en:.2e}  strict={es:.3f}")

    # (A) partial correlation among binders, control = HCDR3 count vs full count.
    b = df[df["neg_log_Kd"].notna()]
    fitb = b["neg_log_Kd"].values
    print("\n(A) AntiFold-fullCDR partial correlation (binders, n=%d):" % len(b))
    raw = spearmanr(b["antifold_fullcdr"], fitb)
    p_h3 = partial_spearman(b["antifold_fullcdr"].values, fitb, b["NumMutations"].values)
    p_full = partial_spearman(b["antifold_fullcdr"].values, fitb,
                              full_mut[df["neg_log_Kd"].notna().values])
    rho_vs_mut = spearmanr(b['antifold_fullcdr'],
                           full_mut[df['neg_log_Kd'].notna().values])[0]
    for k, v in [("antifold_fullcdr_raw_rho", raw[0]),
                 ("antifold_fullcdr_raw_p", raw[1]),
                 ("antifold_fullcdr_partial_rho_ctrl_HCDR3", p_h3[0]),
                 ("antifold_fullcdr_partial_p_ctrl_HCDR3", p_h3[1]),
                 ("antifold_fullcdr_partial_rho_ctrl_fullCDR", p_full[0]),
                 ("antifold_fullcdr_partial_p_ctrl_fullCDR", p_full[1]),
                 ("antifold_fullcdr_rho_vs_fullCDR_mutcount", rho_vs_mut)]:
        records.append({"section": "partial_corr", "metric": k,
                        "model": "AntiFold-fullCDR", "tier": "binders",
                        "N": len(b), "value": v})
    print(f"  raw rho={raw[0]:+.3f} (p={raw[1]:.3f}); "
          f"partial|HCDR3count={p_h3[0]:+.3f} (p={p_h3[1]:.3f}); "
          f"partial|fullCDRcount={p_full[0]:+.3f} (p={p_full[1]:.3f})")
    print(f"  rho(AntiFold-fullCDR, fullCDR mutcount)={rho_vs_mut:+.3f}")

    # (B) tier-3 CI: full-CDR d>=3 split, strict Recall@100 with CIs.
    Xs = [one_hot(df[c].values) for c in CDRS]
    X_full = np.hstack(Xs)
    L_full = sum(len(df[c].iloc[0]) for c in CDRS)
    D = (L_full - (X_full @ X_full.T)).astype(int)
    n = len(df)
    n_comp, labels = connected_components(csr_matrix((D <= 2).astype(int)),
                                          directed=False)  # d_link=2 -> d>=3
    ev = set(labels[np.where(strict)[0]].tolist())
    cnt = sum((labels == c).sum() for c in ev)
    rng = np.random.default_rng(0)
    others = [c for c in range(n_comp) if c not in ev]
    rng.shuffle(others)
    for c in others:
        if cnt >= int(0.40 * n):
            break
        ev.add(c); cnt += (labels == c).sum()
    eval_idx = np.sort(np.where(np.isin(labels, list(ev)))[0])
    pool_idx = np.sort(np.where(~np.isin(labels, list(ev)))[0])
    binder = (df["Binder"].values == 1).astype(int)
    y = np.where(np.isfinite(fit), fit, float(np.nanmin(fit)) - 1.0)
    scorers = {"Ridge(full-CDR)": ridge_scorer(X_full, y),
               "MutCount": lambda tr, te, s: df["NumMutations"].values[te] * -1.0}
    curve = run_learning_curve_fixed(scorers, fit, binder, parent, strict, near,
                                     pool_idx, eval_idx, (50, 100, 500), n_seeds=20)
    print("\n(B) Tier-3 CI — full-CDR d>=3, strict Recall@100 (mean +/- 95%CI):")
    s = summarize(curve, "strict_Recall@100")
    for N in (50, 100, 500):
        r = s[(s.method == "Ridge(full-CDR)") & (s.N == N)].iloc[0]
        m = s[(s.method == "MutCount") & (s.N == N)].iloc[0]
        gap = r["mean"] - m["mean"]
        gap_ci = np.hypot(r["ci"], m["ci"])
        verdict = "straddles 0" if abs(gap) < gap_ci else "gap outside CI"
        for meth, row in [("Ridge(full-CDR)", r), ("MutCount", m)]:
            records.append({"section": "tier3_ci", "metric": "strict_Recall@100_mean",
                            "model": meth, "tier": "d>=3", "N": N, "value": row["mean"]})
            records.append({"section": "tier3_ci", "metric": "strict_Recall@100_ci95",
                            "model": meth, "tier": "d>=3", "N": N, "value": row["ci"]})
        records.append({"section": "tier3_ci", "metric": "gap_Ridge_minus_Mut",
                        "model": "", "tier": "d>=3", "N": N, "value": gap})
        records.append({"section": "tier3_ci", "metric": "gap_ci95",
                        "model": "", "tier": "d>=3", "N": N, "value": gap_ci})
        print(f"  N={N}: Ridge={r['mean']:.2f}±{r['ci']:.2f}  "
              f"Mut={m['mean']:.2f}±{m['ci']:.2f}  gap={gap:+.2f}±{gap_ci:.2f} [{verdict}]")

    # (C) inflation factor from E2g full-CDR rho (d=1 vs d>=5).
    rc = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "absci_fullcdr_recheck.csv"))
    print("\n(C) Correlation-inflation factor (rho d=1 / rho d>=5):")
    infl_full, infl_h3 = 0.68 / 0.48, 0.81 / 0.46
    for k, v in [("rho_d1_fullCDR", 0.68), ("rho_dge5_fullCDR", 0.48),
                 ("inflation_fullCDR", infl_full),
                 ("rho_d1_HCDR3", 0.81), ("rho_dge5_HCDR3", 0.46),
                 ("inflation_HCDR3", infl_h3)]:
        records.append({"section": "inflation", "metric": k,
                        "model": "", "tier": "", "N": "", "value": v})
    print("  full-CDR: 0.68 / 0.48 = %.2fx ; HCDR3-only: 0.81 / 0.46 = %.2fx"
          % (infl_full, infl_h3))

    pd.DataFrame(records, columns=["section", "metric", "model", "tier", "N", "value"]
                 ).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(records)} rows -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
