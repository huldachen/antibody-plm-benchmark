#!/usr/bin/env python3
"""
Figures F1-F4 (full-CDR basis).

Reads the committed results CSVs and writes the four figures next to this file,
in figures/. Run from the repo root in the analysis (plm) env:

    python figures/make_figures.py

  F1  leakage characterization        -> figures/F1_leakage.png
  F2  zero-shot two-prior divergence  -> figures/F2_zeroshot_two_prior.png
  F3  structure reduces to counting   -> figures/F3_structure_counting.png
  F4  correlation vs discovery        -> figures/F4_correlation_vs_discovery.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import one_hot, ridge_scorer  # noqa: E402
from evaluation.improver_metrics import (  # noqa: E402
    improver_mask, near_improver_mask, beat_parent_enrichment,
)

SCORED = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
AF_FULL = os.path.join(PROJECT_ROOT, "results", "absci_antifold_fullcdr.csv")
CDRS = ["HCDR1", "HCDR2", "HCDR3"]
WT = {"HCDR1": "GFNIKDTY", "HCDR2": "IYPTNGYT", "HCDR3": "SRWGGDGFYAMDY"}
ORANGE, BLUE, GREY, GREEN, RED, PURPLE = (
    "#E8820C", "#1F6FB2", "#9AA0A6", "#2E8B57", "#C0392B", "#7B4FB5")


def _load():
    df = pd.read_csv(SCORED)
    df["antifold_fullcdr"] = pd.read_csv(AF_FULL)["antifold_fullcdr_score"].values
    fit = df["neg_log_Kd"].values
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])
    X_full = np.hstack([one_hot(df[c].values) for c in CDRS])
    L_full = sum(len(df[c].iloc[0]) for c in CDRS)
    return df, fit, parent, X_full, L_full


def fig1_leakage(df, X_full, L_full):
    n = len(df)
    D = (L_full - (X_full @ X_full.T)).astype(int)
    X_h3 = one_hot(df["HCDR3"].values)
    D_h3 = (13 - (X_h3 @ X_h3.T)).astype(int)
    binder = (df["Binder"].values == 1).astype(int)
    tr, te = next(StratifiedShuffleSplit(1, train_size=int(0.6 * n),
                                         random_state=0).split(np.zeros(n), binder))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for D_, lab, color, off in [(D_h3, "CDR-H3 only (incomplete)", GREY, -0.19),
                                (D, "full CDR (correct)", BLUE, 0.19)]:
        Dc = D_.copy(); np.fill_diagonal(Dc, L_full + 1)
        mn = Dc[np.ix_(te, tr)].min(axis=1)
        vals, cnts = np.unique(mn[mn <= 9], return_counts=True)
        ax.bar(vals + off, 100 * cnts / len(mn), width=0.38, label=lab,
               color=color, alpha=0.85)
        frac2 = 100 * np.mean(mn <= 2)
        ax.annotate(f"{lab.split(' (')[0]}: {frac2:.0f}% ≤2, "
                    f"{int((mn==0).sum())} exact",
                    (0.97, 0.86 if color == BLUE else 0.93), xycoords="axes fraction",
                    ha="right", fontsize=9, color=color)
    ax.set_xlabel("min Hamming distance from held-out variant to nearest "
                  "training variant", fontsize=10)
    ax.set_ylabel("% of held-out variants", fontsize=10.5)
    ax.set_title("F1. Sequence-overlap leakage, correctly measured on the full "
                 "CDR basis\n(CDR-H3-only distance overstates near-duplication)",
                 fontsize=11, loc="left", weight="bold")
    ax.legend(fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "F1_leakage.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_two_prior(df, fit, parent):
    near = near_improver_mask(fit, 0.10)
    models = [("MutCount", -df["NumMutations"].values, ORANGE),
              ("BLOSUM62", df["BLOSUM62_score"].values, GREEN),
              ("ESM-2", df["ESM2_score"].values, RED),
              ("AbLang", df["AbLang_score"].values, BLUE),
              ("AntiBERTy", df["AntiBERTy_score"].values, PURPLE)]
    names, signed, aurocs, colors = [], [], [], []
    for nm, s, c in models:
        e = beat_parent_enrichment(s, fit, parent, positive_mask=near)
        p_e, p_a = e["MW_p_enrich"], e["MW_p_anti"]
        signed.append(-np.log10(max(p_e, 1e-12)) if p_e <= p_a
                      else np.log10(max(p_a, 1e-12)))
        aurocs.append(roc_auc_score(near.astype(int), s))
        names.append(nm); colors.append(c)
    y = np.arange(len(names))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a.barh(y, signed, color=colors, alpha=0.85)
    a.axvline(0, color="0.3", lw=1)
    for x in (np.log10(0.05) * -1, np.log10(0.05)):
        a.axvline(x, ls=":", color="0.5", lw=1)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=10)
    a.set_xlabel("signed −log10(MW p)  (right = enriches improvers)", fontsize=10)
    a.set_title("A. Near-improver enrichment", fontsize=11, loc="left", weight="bold")
    b.barh(y, aurocs, color=colors, alpha=0.85)
    b.axvline(0.5, color="0.4", lw=1, ls="--")
    b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(0.35, 0.85)
    b.set_xlabel("near-improver AUROC  (0.5 = chance)", fontsize=10)
    b.set_title("B. Improver retrieval (AUROC)", fontsize=11, loc="left", weight="bold")
    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.2, axis="x")
    fig.suptitle("F2. Two edit-distance-like priors: parent-distance baselines "
                 "enrich improvers; the PLMs do not", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "F2_zeroshot_two_prior.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def _partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def r(a, b):
        B = np.c_[np.ones_like(b), b]
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return pearsonr(r(rx, rz), r(ry, rz))[0]


def fig3_structure(df):
    b = df[df["neg_log_Kd"].notna()]
    fitb, mut = b["neg_log_Kd"].values, b["NumMutations"].values.astype(float)
    rows = [("MutCount", -b["NumMutations"].values), ("BLOSUM62", b["BLOSUM62_score"].values),
            ("AntiFold\n(full-CDR)", b["antifold_fullcdr"].values), ("ESM-2", b["ESM2_score"].values)]
    names, raw, part = [], [], []
    for nm, s in rows:
        names.append(nm); raw.append(spearmanr(s, fitb)[0]); part.append(_partial(s, fitb, mut))
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for yi, r_, p_ in zip(y, raw, part):
        ax.plot([r_, p_], [yi, yi], color="0.7", lw=2, zorder=1)
    ax.scatter(raw, y, s=90, color=BLUE, label="raw ρ (vs affinity)", zorder=3)
    ax.scatter(part, y, s=90, facecolor="white", edgecolor=RED, linewidth=2,
               label="partial ρ (control mutation count)", zorder=3)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Spearman ρ with affinity (among binders, n=420)", fontsize=10.5)
    ax.set_title("F3. Fixed-backbone inverse folding reduces to counting\n"
                 "(raw correlation collapses to n.s. once mutation count is "
                 "controlled)", fontsize=11, loc="left", weight="bold")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "F3_structure_counting.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig4_corr_vs_discovery(df, fit, parent, X_full, L_full):
    n = len(df)
    D = (L_full - (X_full @ X_full.T)).astype(int)
    binder = (df["Binder"].values == 1).astype(int)
    near = near_improver_mask(fit, 0.10)
    mut = df["NumMutations"].values.astype(float)
    y = np.where(np.isfinite(fit), fit, float(np.nanmin(fit)) - 1.0)
    ridge = ridge_scorer(X_full, y)
    BINS = [1, 2, 3, 4, 5]
    rho_rec, auc_rec = [], []
    for seed in range(20):
        tr, te = next(StratifiedShuffleSplit(1, train_size=200, random_state=seed
                                             ).split(np.zeros(n), binder))
        preds = {"Ridge": ridge(tr, te, seed), "MutCount": -mut[te]}
        d2t = D[np.ix_(te, tr)].min(axis=1)
        ft, nt = fit[te], near[te]
        tb = np.isfinite(ft)
        for bb in BINS:
            sel = tb & ((d2t >= bb) if bb == BINS[-1] else (d2t == bb))
            lab = f"≥{bb}" if bb == BINS[-1] else str(bb)
            if sel.sum() < 15:
                continue
            for m, p in preds.items():
                rho_rec.append({"bin": lab, "m": m, "v": spearmanr(p[sel], ft[sel])[0]})
                yb = nt[sel].astype(int)
                if 2 <= yb.sum() < sel.sum():
                    auc_rec.append({"bin": lab, "m": m, "v": roc_auc_score(yb, p[sel])})
    order = [str(b) for b in BINS[:-1]] + [f"≥{BINS[-1]}"]
    style = {"Ridge": dict(color=BLUE, marker="o"),
             "MutCount": dict(color=ORANGE, marker="o", ls="--")}

    def agg(rec):
        d = pd.DataFrame(rec).groupby(["bin", "m"])["v"].agg(["mean", "std", "count"]).reset_index()
        d["ci"] = 1.96 * d["std"] / np.sqrt(d["count"].clip(lower=1))
        return d
    A, B = agg(rho_rec), agg(auc_rec)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, dd, ttl, ref, ylab in [
            (ax[0], A, "A. Global fitness ranking — labels generalize", 0.0,
             "Spearman ρ (score vs affinity)"),
            (ax[1], B, "B. Improver retrieval (decision-relevant) — ≈ counting", 0.5,
             "near-improver AUROC")]:
        x = np.arange(len(order))
        for m, st in style.items():
            yv = [dd[(dd.bin == bn) & (dd.m == m)]["mean"] for bn in order]
            cv = [dd[(dd.bin == bn) & (dd.m == m)]["ci"] for bn in order]
            a.errorbar(x, [float(v.iloc[0]) if len(v) else np.nan for v in yv],
                       yerr=[float(v.iloc[0]) if len(v) else 0 for v in cv],
                       label=m, capsize=3, lw=2.3, **st)
        a.axhline(ref, color="0.6", lw=0.8, ls=":")
        a.set_xticks(x); a.set_xticklabels(order)
        a.set_xlabel("CDR Hamming distance to nearest training variant", fontsize=10)
        a.set_ylabel(ylab, fontsize=10.5)
        a.set_title(ttl, fontsize=11, loc="left", weight="bold")
        a.grid(alpha=0.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=10, frameon=False)
    fig.suptitle("F4. The advantage that generalizes (correlation) is not the "
                 "one that discovers (improver retrieval)", fontsize=12,
                 weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "F4_correlation_vs_discovery.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(HERE, exist_ok=True)
    df, fit, parent, X_full, L_full = _load()
    print("F1 leakage ...");            fig1_leakage(df, X_full, L_full)
    print("F2 two-prior ...");          fig2_two_prior(df, fit, parent)
    print("F3 structure=counting ...."); fig3_structure(df)
    print("F4 correlation vs discovery ..."); fig4_corr_vs_discovery(df, fit, parent, X_full, L_full)
    print("Saved F1-F4 to figures/")


if __name__ == "__main__":
    main()
