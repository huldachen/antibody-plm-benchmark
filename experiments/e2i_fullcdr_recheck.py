#!/usr/bin/env python3
"""
Full-CDR re-check (spine) — does the leakage conclusion survive using ALL varying
positions and full-sequence distance, not HCDR3 only?

Foundational fix: Absci variants vary in HCDR1 (8), HCDR2 (8) AND HCDR3 (13) —
not HCDR3 alone. Earlier E2e/f/g used HCDR3-only features and HCDR3-only Hamming.
Here we recompute with full-CDR one-hot features (29 positions) and full-CDR
Hamming distance (the true memorisation axis). The dataset carries only the three
heavy CDRs (no light chain / framework), so this is the complete feature basis.

Reports, side by side with the HCDR3-only numbers:
  Step 0  — per-CDR varying-position count.
  F1      — leakage: nearest-neighbour full-CDR distance distribution.
  E2f     — distance-aware split (full-CDR), Ridge(full-CDR) vs MutCount.
  E2g     — distance-stratified rho + near-AUROC (full-CDR bins).

Run (plm env): python experiments/e2i_fullcdr_recheck.py
Outputs: results/absci_fullcdr_recheck.csv, figures/figE2i_fullcdr.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import one_hot, ridge_scorer, SCORED_CSV  # noqa: E402
from evaluation.improver_metrics import improver_mask, near_improver_mask  # noqa: E402
from evaluation.lowN_protocol import run_learning_curve_fixed, summarize  # noqa: E402

OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_fullcdr_recheck.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2i_fullcdr.png")
NS = (20, 50, 100, 200, 500)
N_SEEDS = 20
EVAL_FRAC = 0.40
CDRS = ["HCDR1", "HCDR2", "HCDR3"]
DIST_BINS = [1, 2, 3, 4, 5]
MIN_BIN_N, MIN_POS = 15, 2


def build_split(D, strata, force_mask, d_link, eval_frac, seed=0):
    n = D.shape[0]
    n_comp, labels = connected_components(csr_matrix((D <= d_link).astype(int)),
                                          directed=False)
    imp = set(labels[np.where(force_mask)[0]].tolist())
    ev, cnt = set(imp), sum((labels == c).sum() for c in imp)
    rng = np.random.default_rng(seed)
    others = [c for c in range(n_comp) if c not in ev]
    rng.shuffle(others)
    for c in others:
        if cnt >= int(eval_frac * n):
            break
        ev.add(c); cnt += (labels == c).sum()
    eval_idx = np.sort(np.where(np.isin(labels, list(ev)))[0])
    pool_idx = np.sort(np.where(~np.isin(labels, list(ev)))[0])
    return pool_idx, eval_idx, int(D[np.ix_(eval_idx, pool_idx)].min())


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    fit = df["neg_log_Kd"].values
    binder = (df["Binder"].values == 1).astype(int)
    mut = df["NumMutations"].values.astype(float)          # HCDR3 count (baseline)
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])
    strict, near = improver_mask(fit, parent), near_improver_mask(fit, 0.10)
    n = len(df)

    # Full-CDR one-hot + total position count.
    Xs = [one_hot(df[c].values) for c in CDRS]
    Ls = [len(df[c].iloc[0]) for c in CDRS]
    X_full = np.hstack(Xs)
    L_full = sum(Ls)
    y = np.where(np.isfinite(fit), fit, float(np.nanmin(fit)) - 1.0)

    # Step 0 — varying positions per CDR.
    print("Step 0 — varying positions (of total) per heavy CDR:")
    for c, L in zip(CDRS, Ls):
        arr = np.array([list(s) for s in df[c].values])
        vary = sum(len(set(arr[:, p])) > 1 for p in range(L))
        print(f"  {c}: {vary}/{L} positions vary")
    print(f"  full-CDR basis: {L_full} positions ({X_full.shape[1]}-d one-hot)\n")

    # Distance matrices: full-CDR vs HCDR3-only.
    D_full = (L_full - (X_full @ X_full.T)).astype(int)
    X_h3 = Xs[2]
    D_h3 = (Ls[2] - (X_h3 @ X_h3.T)).astype(int)

    # F1 — leakage under each distance, on a binder-stratified 40% eval split.
    sss = StratifiedShuffleSplit(1, train_size=int(0.6 * n), random_state=0)
    tr, te = next(sss.split(np.zeros(n), binder))
    print("F1 — eval->nearest-training distance (% within <=2, exact dups):")
    for name, D in [("HCDR3-only", D_h3), ("full-CDR", D_full)]:
        Dc = D.copy(); np.fill_diagonal(Dc, L_full + 1)
        mn = Dc[np.ix_(te, tr)].min(axis=1)
        print(f"  {name}: median={np.median(mn):.0f}, frac(d<=2)="
              f"{100*np.mean(mn <= 2):.1f}%, exact(d=0)={int((mn == 0).sum())}")
    print()

    rows = []
    # E2f — distance-aware split on full-CDR distance.
    print("E2f (full-CDR distance) — Ridge(full-CDR) vs MutCount, strict "
          "Recall@100 / near P@30:")
    scorers = {"Ridge(full-CDR)": ridge_scorer(X_full, y),
               "MutCount": lambda tr, te, s: -mut[te]}
    for d_link in (1, 2):
        pool_idx, eval_idx, mind = build_split(D_full, binder, strict, d_link,
                                               EVAL_FRAC)
        curve = run_learning_curve_fixed(scorers, fit, binder, parent, strict,
                                         near, pool_idx, eval_idx, NS,
                                         n_seeds=N_SEEDS)
        for metric in ("strict_Recall@100", "near_P@30"):
            s = summarize(curve, metric)
            g = lambda m, N: s[(s.method == m) & (s.N == N)]["mean"].iloc[0]
            print(f"  d>={d_link+1} ({metric}): "
                  f"Ridge N50={g('Ridge(full-CDR)',50):.2f} N100="
                  f"{g('Ridge(full-CDR)',100):.2f} N500={g('Ridge(full-CDR)',500):.2f}"
                  f" | Mut={g('MutCount',100):.2f}")
            for N in (50, 100, 500):
                rows.append({"check": f"E2f_d{d_link}", "metric": metric, "N": N,
                             "Ridge": g("Ridge(full-CDR)", N),
                             "MutCount": g("MutCount", N), "min_dist": mind})

    # E2g — distance-stratified rho + near-AUROC (full-CDR bins).
    print("\nE2g (full-CDR bins) — Ridge(full-CDR) vs MutCount:")
    ridge = ridge_scorer(X_full, y)
    rr, aa = [], []
    for seed in range(N_SEEDS):
        s2 = StratifiedShuffleSplit(1, train_size=200, random_state=seed)
        tri, tei = next(s2.split(np.zeros(n), binder))
        preds = {"Ridge(full-CDR)": ridge(tri, tei, seed), "MutCount": -mut[tei]}
        d2t = D_full[np.ix_(tei, tri)].min(axis=1)
        ft, nt = fit[tei], near[tei]
        tb = np.isfinite(ft)
        for b in DIST_BINS:
            sel = tb & ((d2t >= b) if b == DIST_BINS[-1] else (d2t == b))
            lab = f">={b}" if b == DIST_BINS[-1] else str(b)
            if sel.sum() < MIN_BIN_N:
                continue
            for m, p in preds.items():
                rr.append({"bin": lab, "method": m, "rho": spearmanr(p[sel], ft[sel])[0]})
                yb = nt[sel].astype(int)
                if MIN_POS <= yb.sum() < sel.sum():
                    aa.append({"bin": lab, "method": m, "auroc": roc_auc_score(yb, p[sel])})
    rho_df, auc_df = pd.DataFrame(rr), pd.DataFrame(aa)
    order = [str(b) for b in DIST_BINS[:-1]] + [f">={DIST_BINS[-1]}"]
    for m in ("Ridge(full-CDR)", "MutCount"):
        rr_ = rho_df[rho_df.method == m].groupby("bin")["rho"].mean()
        aa_ = auc_df[auc_df.method == m].groupby("bin")["auroc"].mean()
        print(f"  {m:<16} rho: " + " ".join(f"{b}:{rr_.get(b, np.nan):.2f}" for b in order))
        print(f"  {'':<16} auroc: " + " ".join(f"{b}:{aa_.get(b, np.nan):.2f}" for b in order))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    _plot(rho_df, auc_df, order)
    print(f"\nSaved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(rho_df, auc_df, order) -> None:
    style = {"Ridge(full-CDR)": dict(color="#1F6FB2", marker="o"),
             "MutCount": dict(color="#E8820C", marker="o", ls="--")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, df, val, ttl, ref in [(ax[0], rho_df, "rho", "A. Global rho (full-CDR)", 0),
                                 (ax[1], auc_df, "auroc", "B. Near-AUROC (full-CDR)", 0.5)]:
        g = df.groupby(["bin", "method"])[val].agg(["mean", "std", "count"]).reset_index()
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
        x = np.arange(len(order))
        for m, st in style.items():
            yv = [g[(g.bin == b) & (g.method == m)]["mean"]
                  for b in order]
            cv = [g[(g.bin == b) & (g.method == m)]["ci"] for b in order]
            a.errorbar(x, [float(v.iloc[0]) if len(v) else np.nan for v in yv],
                       yerr=[float(v.iloc[0]) if len(v) else 0 for v in cv],
                       label=m, capsize=3, **st)
        a.axhline(ref, color="0.6", lw=0.8, ls=":")
        a.set_xticks(x); a.set_xticklabels(order)
        a.set_xlabel("full-CDR Hamming distance to nearest training variant")
        a.set_title(ttl, fontsize=11, loc="left", weight="bold")
        a.grid(alpha=0.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(frameon=False)
    fig.suptitle("E2i — full-CDR re-check: does the correlation-vs-discovery "
                 "split survive full-sequence features/distance?", weight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
