#!/usr/bin/env python3
"""
E1c (mechanism) — is PLM enrichment set by the positive set's low-mutation skew?

A controlled test. Four conditions vary in how strongly their positive set
skews toward low mutation count: Mason binders, Absci binders, Absci
near-improvers, Absci strict improvers. For each we measure, on the SAME pool and
scores:
  x = MutCount AUROC(positive vs rest)  — how strongly positives skew low-mutation
  y = PLM AUROC(positive vs rest)       — how much the PLM enriches positives
The three Absci conditions share dataset AND scores (only the positive
DEFINITION changes), so they are internally controlled; Mason extends the range
(confounds dataset with task — noted). If y tracks x monotonically through
(0.5, 0.5), PLM "enrichment" is a low-mutation-count proxy whose sign is set by
the target, not a task-independent signal.

Run (plm env): python experiments/e1c_mechanism.py
Outputs: results/absci_mason_mechanism.csv, figures/figMech_plm_vs_mutcount.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import improver_mask, near_improver_mask  # noqa: E402

ABSCI = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
MASON = os.path.join(PROJECT_ROOT, "results", "mason_all_scored.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_mason_mechanism.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figMech_plm_vs_mutcount.png")
PLMS = [("ESM-2", "ESM2_score", "#C0392B"),
        ("AbLang", "AbLang_score", "#1F6FB2"),
        ("AntiBERTy", "AntiBERTy_score", "#2E8B57")]


def auroc(pos_mask, score):
    return roc_auc_score(pos_mask.astype(int), score)


def main() -> None:
    ab = pd.read_csv(ABSCI)
    ma = pd.read_csv(MASON)
    ab_fit = ab["neg_log_Kd"].values
    ab_parent = float(ab.loc[ab["NumMutations"] == 0, "neg_log_Kd"].iloc[0])

    conditions = [
        ("Mason binders", ma, (ma["AgClass"].values == 1), ma["NumMutations"].values),
        ("Absci binders", ab, (ab["Binder"].values == 1), ab["NumMutations"].values),
        ("Absci near-improvers", ab, near_improver_mask(ab_fit, 0.10),
         ab["NumMutations"].values),
        ("Absci strict improvers", ab, improver_mask(ab_fit, ab_parent),
         ab["NumMutations"].values),
    ]

    rows = []
    for name, df, pos, mut in conditions:
        x = auroc(pos, -mut)                       # MutCount AUROC (low-mut skew)
        row = {"condition": name, "n_pos": int(pos.sum()),
               "mutcount_auroc": x}
        for pname, col, _ in PLMS:
            row[f"{pname}_auroc"] = auroc(pos, df[col].values)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 200):
        print(res.to_string(index=False))

    # correlation of PLM AUROC with MutCount AUROC across the 4 conditions
    print("\nSpearman(PLM AUROC, MutCount AUROC) across conditions:")
    for pname, _, _ in PLMS:
        from scipy.stats import spearmanr
        r = spearmanr(res["mutcount_auroc"], res[f"{pname}_auroc"])[0]
        print(f"  {pname}: rho = {r:+.2f}")

    _plot(res)
    print(f"\nSaved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(res) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.axvline(0.5, color="0.7", lw=0.8, ls=":")
    ax.plot([0.3, 0.85], [0.3, 0.85], color="0.8", lw=1.0, ls="--", zorder=0)
    for pname, _, color in PLMS:
        ax.plot(res["mutcount_auroc"], res[f"{pname}_auroc"], "o-", color=color,
                label=pname, ms=8, lw=1.5, alpha=0.85)
    for _, r in res.iterrows():
        ax.annotate(f"{r['condition']}\n(n={int(r['n_pos'])})",
                    (r["mutcount_auroc"], r["AbLang_auroc"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", color="0.35")
    ax.set_xlabel("MutCount AUROC — how strongly the positive set skews "
                  "low-mutation", fontsize=10)
    ax.set_ylabel("PLM AUROC — how much the PLM enriches the positive set",
                  fontsize=10)
    ax.set_title("PLM enrichment tracks the target's low-mutation skew\n"
                 "(above 0.5 = enriches; the sign is set by mutation count, not "
                 "the model)", fontsize=11, loc="left", weight="bold")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
