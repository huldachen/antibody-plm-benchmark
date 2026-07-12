#!/usr/bin/env python3
"""
E1 figure -- "PLMs anti-rank improvers; mutation count enriches them."

Two panels, both driven off the same harness as the E1 table so the figure
can never drift from results/absci_improver_metrics.csv:

  A. Improver-rank landscape: where each of the 5 Absci improvers falls in the
     1266-variant ranking, per model, against the pool-median (random) line.
  B. Beat-parent enrichment: signed -log10(Mann-Whitney p), green = enriches
     improvers, red = anti-ranks them, with the p=0.05 significance threshold.

Run:
    python experiments/e1_figure.py
Output:
    figures/figE1_improver_antirank.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import beat_parent_enrichment  # noqa: E402

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE1_improver_antirank.png")

FITNESS_COL = "neg_log_Kd"
MUTATION_COL = "NumMutations"

# Narrative order top->bottom: enricher, neutral, then the three anti-rankers.
MODELS = [
    ("MutCount", "MutCount_score"),
    ("BLOSUM62", "BLOSUM62_score"),
    ("ESM-2", "ESM2_score"),
    ("AbLang", "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
]

ORANGE = "#E8820C"   # mutation-count highlight (repo convention)
NEUTRAL = "#6B7280"  # BLOSUM / neutral
BLUE = "#1F6FB2"     # PLMs (Absci blue family)
GREEN = "#2E8B57"
RED = "#C0392B"


def model_color(name: str) -> str:
    if name == "MutCount":
        return ORANGE
    if name == "BLOSUM62":
        return NEUTRAL
    return BLUE


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    df["MutCount_score"] = -df[MUTATION_COL].astype(float)
    parent = float(df.loc[df[MUTATION_COL] == 0, FITNESS_COL].iloc[0])
    fit = df[FITNESS_COL].values
    n_total = len(df)
    pool_median = n_total / 2.0

    stats = []
    n_imp = 0
    for name, col in MODELS:
        enr = beat_parent_enrichment(df[col].values, fit, parent,
                                     ks=(10, 30, 100))
        n_imp = enr["n_improvers"]
        # Signed enrichment (tie-corrected p): + if improvers score better than
        # chance, - if worse.
        p_enr, p_anti = enr["MW_p_enrich"], enr["MW_p_anti"]
        if p_enr <= p_anti:
            signed = -np.log10(max(p_enr, 1e-12))     # enrichment (positive)
        else:
            signed = np.log10(max(p_anti, 1e-12))      # anti-ranking (negative)
        stats.append({"name": name, "mid": enr["improver_midranks"],
                      "ranges": enr["improver_tie_ranges"],
                      "median": enr["median_rank"], "signed": signed})

    ys = np.arange(len(MODELS))[::-1]  # first model at top

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.5, 1.0]})

    # ---- Panel A: improver-rank landscape ---------------------------------
    axA.axvspan(0, pool_median, color=GREEN, alpha=0.05)
    axA.axvspan(pool_median, n_total, color=RED, alpha=0.05)
    axA.axvline(pool_median, color="0.4", ls="--", lw=1.2, zorder=1)
    axA.text(pool_median + n_total * 0.01, len(MODELS) - 0.5,
             f"random (median rank {int(pool_median)})",
             ha="left", va="top", fontsize=8.5, color="0.35",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                       alpha=0.85))

    for y, s in zip(ys, stats):
        c = model_color(s["name"])
        axA.plot([1, n_total], [y, y], color="0.85", lw=1.0, zorder=1)
        # Score-tie spans: each improver could sit anywhere in this rank range.
        for lo, hi in s["ranges"]:
            if hi > lo:
                axA.plot([lo, hi], [y, y], color=c, lw=5, alpha=0.28,
                         solid_capstyle="round", zorder=2)
        axA.scatter(s["mid"], [y] * len(s["mid"]), s=70, color=c,
                    edgecolor="white", linewidth=0.8, zorder=3)
        axA.scatter([s["median"]], [y], marker="|", s=650, color=c,
                    linewidth=2.2, zorder=2)
        axA.text(n_total * 1.02, y, f"med {s['median']:.0f}", va="center",
                 ha="left", fontsize=8.5, color=c)

    axA.set_yticks(ys)
    axA.set_yticklabels([s["name"] for s in stats], fontsize=11)
    axA.set_xlim(0, n_total * 1.14)
    axA.set_ylim(-0.6, len(MODELS) - 0.4)
    axA.set_xlabel(f"Rank in the {n_total}-variant pool  (1 = best predicted)",
                   fontsize=10.5)
    axA.set_title(f"A. Where the {n_imp} improvers land in each model's ranking",
                  fontsize=11.5, loc="left", weight="bold")
    axA.text(1, len(MODELS) - 0.5, "shaded bar = score-tie rank span",
             fontsize=8, color="0.45", va="top", ha="left")
    axA.text(n_total * 0.24, -0.55, "← enriched (better than random)",
             fontsize=8.5, color=GREEN, ha="center")
    axA.text(n_total * 0.80, -0.55, "anti-ranked (worse than random) →",
             fontsize=8.5, color=RED, ha="center")
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ---- Panel B: signed enrichment significance --------------------------
    thr = -np.log10(0.05)
    for y, s in zip(ys, stats):
        val = s["signed"]
        axB.barh(y, val, color=(GREEN if val > 0 else RED), alpha=0.85,
                 height=0.6, edgecolor="white")
    axB.axvline(0, color="0.3", lw=1.0)
    for x in (thr, -thr):
        axB.axvline(x, color="0.5", ls=":", lw=1.1)
    axB.text(thr, len(MODELS) - 0.45, "p=0.05", rotation=90, fontsize=7.5,
             va="top", ha="right", color="0.4")
    axB.text(-thr, len(MODELS) - 0.45, "p=0.05", rotation=90, fontsize=7.5,
             va="top", ha="left", color="0.4")
    axB.set_yticks(ys)
    axB.set_yticklabels([])
    axB.tick_params(left=False)
    axB.set_ylim(-0.6, len(MODELS) - 0.4)
    axB.set_xlabel("signed  -log10(MW p)", fontsize=10.5)
    axB.set_title("B. Beat-parent enrichment (tie-corrected)", fontsize=11.5,
                  loc="left", weight="bold")
    axB.legend(handles=[Patch(color=GREEN, label="enriches improvers"),
                        Patch(color=RED, label="anti-ranks improvers")],
               fontsize=8.5, loc="lower right", frameon=False)
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)

    fig.suptitle(
        "Zero-shot PLMs anti-rank the rare improvers; the mutation counter "
        "only trends the other way (n.s.)  —  Absci HER2, MW p tie-corrected",
        fontsize=12.0, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved -> {os.path.relpath(OUT_PNG, PROJECT_ROOT)}")
    for s in stats:
        print(f"  {s['name']:<10s} median_rank={s['median']:>7.1f}  "
              f"signed_log10p={s['signed']:+.2f}")


if __name__ == "__main__":
    main()
