#!/usr/bin/env python3
"""
E2g — distance-stratified: correlation vs discovery, on the SAME strata.

The reconciliation the paper hinges on. Random train/test splits; bin each test
binder by CDR-H3 Hamming distance to the nearest training variant; within each
bin (SAME binder subset) compute BOTH:
  (A) global fitness ranking  — Spearman rho(score, neg_log_Kd)
  (B) discovery              — near-improver retrieval AUROC (near-improver vs
                               other binders, ranked by score)
for Ridge(compact) and MutCount, with per-bin n and CIs, plus the within-bin
mutation-count range (the mechanism).

Expected story: Ridge keeps a global-rho edge that decays with distance but stays
above MutCount even far away — YET that edge does NOT convert into an
improver-retrieval edge (panel B), where Ridge ~ MutCount. "Labels generalize
for correlation, not for discovery." Mechanism: far from training every variant
is high-mutation, so MutCount has ~no variance (rho collapses) while Ridge keeps
position-identity signal — but that signal ranks the bulk, not the rare top.

Run (plm env): python experiments/e2g_distance_stratified.py
Outputs: results/absci_e2g_dist_stratified.csv, figures/figE2g_distance_stratified.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import load_absci, ridge_scorer  # noqa: E402

OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e2g_dist_stratified.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2g_distance_stratified.png")

N_TRAIN = 200
N_SEEDS = 20
L = 13
DIST_BINS = [1, 2, 3, 4, 5]        # last is ">=5"
MIN_BIN_N = 15                      # binders needed to score a bin
MIN_POS = 2                         # near-improvers needed for AUROC in a bin


def main() -> None:
    data = load_absci()
    X1, Xc = data["X_1hot"], data["X_compact"]
    fit, binder, mut, y, near = (data["fit"], data["binder"], data["mutcount"],
                                 data["y"], data["near"])
    n = len(fit)
    D = (L - (X1 @ X1.T)).astype(int)
    ridge = ridge_scorer(Xc, y)

    rho_rec, auc_rec, mut_rec = [], [], []
    for seed in range(N_SEEDS):
        sss = StratifiedShuffleSplit(1, train_size=N_TRAIN, random_state=seed)
        tr, te = next(sss.split(np.zeros(n), binder))
        preds = {"Ridge(compact)": ridge(tr, te, seed), "MutCount": -mut[te]}
        d_to_train = D[np.ix_(te, tr)].min(axis=1)
        fit_te, near_te, mut_te = fit[te], near[te], mut[te]
        te_binder = np.isfinite(fit_te)

        for b in DIST_BINS:
            last = (b == DIST_BINS[-1])
            in_bin = (d_to_train >= b) if last else (d_to_train == b)
            blabel = f">={b}" if last else str(b)
            sel = te_binder & in_bin                 # binders in this bin
            if sel.sum() < MIN_BIN_N:
                continue
            mut_rec.append({"dist_bin": blabel, "seed": seed,
                            "n_binders": int(sel.sum()),
                            "mut_mean": float(mut_te[sel].mean()),
                            "mut_min": int(mut_te[sel].min()),
                            "mut_max": int(mut_te[sel].max()),
                            "n_near": int(near_te[sel].sum())})
            for mname, pred in preds.items():
                p = pred[sel]
                rho, _ = spearmanr(p, fit_te[sel])
                rho_rec.append({"dist_bin": blabel, "method": mname, "rho": rho})
                # Discovery: near-improver vs other binders, SAME subset.
                yb = near_te[sel].astype(int)
                if MIN_POS <= yb.sum() < sel.sum():
                    auc_rec.append({"dist_bin": blabel, "method": mname,
                                    "auroc": roc_auc_score(yb, p)})

    rho_df, auc_df, mut_df = (pd.DataFrame(rho_rec), pd.DataFrame(auc_rec),
                              pd.DataFrame(mut_rec))

    def agg(df, val):
        g = df.groupby(["dist_bin", "method"])[val].agg(["mean", "std", "count"])
        g = g.reset_index()
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
        return g

    A, B = agg(rho_df, "rho"), agg(auc_df, "auroc")
    A["metric"], B["metric"] = "spearman_rho", "improver_auroc"
    pd.concat([A, B], ignore_index=True).to_csv(OUT_CSV, index=False)

    order = [str(b) for b in DIST_BINS[:-1]] + [f">={DIST_BINS[-1]}"]
    order = [b for b in order if b in A["dist_bin"].values]
    mb = mut_df.groupby("dist_bin")
    print(f"Train N={N_TRAIN}, {N_SEEDS} seeds. Per test-to-train Hamming bin:\n")
    print("bin  n_bind  mut(min-max,mean)  n_near | rho_Ridge rho_Mut | AUROC_Ridge AUROC_Mut")
    for b in order:
        nb = int(mb.get_group(b)["n_binders"].mean())
        mn = int(mb.get_group(b)["mut_min"].min())
        mx = int(mb.get_group(b)["mut_max"].max())
        mm = mb.get_group(b)["mut_mean"].mean()
        nn = mb.get_group(b)["n_near"].mean()

        def cell(df, val, b, m):
            r = df[(df.dist_bin == b) & (df.method == m)]
            return f"{r['mean'].iloc[0]:.2f}±{r['ci'].iloc[0]:.2f}" if len(r) else "  -  "
        print(f"{b:>4} {nb:>6} {mn}-{mx}, {mm:>4.1f}       {nn:>4.1f} | "
              f"{cell(A,'rho',b,'Ridge(compact)')} {cell(A,'rho',b,'MutCount')} | "
              f"{cell(B,'auroc',b,'Ridge(compact)')} {cell(B,'auroc',b,'MutCount')}")

    _plot(A, B, order)
    print(f"\nSaved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(A, B, order) -> None:
    style = {"Ridge(compact)": dict(color="#1F6FB2", marker="o", lw=2.4),
             "MutCount": dict(color="#E8820C", marker="o", lw=2.4, ls="--")}
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    def draw(ax, df, val, ylab, title, ref=None):
        for m, st in style.items():
            ys = [df[(df.dist_bin == b) & (df.method == m)]["mean"]
                  for b in order]
            cs = [df[(df.dist_bin == b) & (df.method == m)]["ci"] for b in order]
            yv = [float(v.iloc[0]) if len(v) else np.nan for v in ys]
            cv = [float(v.iloc[0]) if len(v) else 0 for v in cs]
            ax.errorbar(x, yv, yerr=cv, label=m, capsize=3, **st)
        if ref is not None:
            ax.axhline(ref, color="0.6", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_xlabel("CDR-H3 Hamming distance to nearest training variant",
                      fontsize=10)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    draw(axes[0], A, "rho", "Spearman ρ (score vs fitness)",
         "A. Global fitness ranking — labels generalize (ρ 0.46 > 0.20 at d≥5)",
         ref=0)
    draw(axes[1], B, "auroc", "Near-improver AUROC",
         "B. Stringent discovery (near-AUROC) — edge collapses to counting",
         ref=0.5)
    axes[0].legend(fontsize=10, frameon=False, loc="upper right")
    fig.suptitle("E2g — the advantage that GENERALIZES (broad correlation) is "
                 "not the same as the one that DISCOVERS (stringent top-K)",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
