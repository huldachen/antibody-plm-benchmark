#!/usr/bin/env python3
"""
E2 -- Few-shot learning curves: does adding labels beat counting mutations?

Honest framing: mutation count is the explicit
baseline-to-beat. E1 showed it *significantly* enriches good binders
(near-improver p=3e-7) with ZERO labels. E2 asks whether spending N wet-lab
labels on a supervised few-shot model buys anything over just counting
mutations — and if so, from what N.

Scope of this first cut (labels axis):
  baselines : MutCount (0 labels), Random
  few-shot  : Ridge on one-hot CDR-H3; Ridge on compact (one-hot + mut count)
Regression target is censored neg_log_Kd (DEC-008). Structure scorers
(AntiFold/ESM-IF) and PLM-embedding few-shot are the E4 / Group-B(i,ii)
follow-ups, not here.

Run:
    python experiments/e2_learning_curve.py
Outputs:
    results/absci_e2_lowN.csv            (long-form: method, N, seed, metrics)
    results/absci_e2_lowN_summary.csv    (mean + 95% CI per method x N)
    figures/figE2_learning_curves.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.lowN_protocol import run_learning_curve, summarize  # noqa: E402
from experiments.absci_fewshot_common import load_absci, base_scorers  # noqa: E402

OUT_LONG = os.path.join(PROJECT_ROOT, "results", "absci_e2_lowN.csv")
OUT_SUM = os.path.join(PROJECT_ROOT, "results", "absci_e2_lowN_summary.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2_learning_curves.png")

NS = (20, 50, 100, 200, 500)
N_SEEDS = 20

ORANGE = "#E8820C"
GREY = "#6B7280"
BLUE = "#1F6FB2"
TEAL = "#2E8B8B"


def main() -> None:
    data = load_absci()
    fit, binder, parent = data["fit"], data["binder"], data["parent"]
    strict_mask, near_mask = data["strict"], data["near"]

    print(f"Absci HER2: n={len(data['df'])}, binders={binder.sum()}, "
          f"strict_improvers={int(strict_mask.sum())}, "
          f"near_improvers={int(near_mask.sum())}, censor_floor={data['floor']:.3f}")

    scorers = base_scorers(data)

    print(f"Running learning curve: N={NS}, seeds={N_SEEDS} ...")
    curve = run_learning_curve(scorers, fit, binder, parent, strict_mask,
                               near_mask, NS, n_seeds=N_SEEDS)
    curve.to_csv(OUT_LONG, index=False)

    metrics = ["near_P@30", "strict_Recall@100"]
    summ = pd.concat([summarize(curve, m).assign(metric=m) for m in metrics],
                     ignore_index=True)
    summ.to_csv(OUT_SUM, index=False)

    # ---- console summary at the extremes ----
    for m in metrics:
        print(f"\n{m}  (mean ± 95%CI):")
        s = summarize(curve, m)
        for method in scorers:
            sub = s[s["method"] == method].sort_values("N")
            cells = "  ".join(f"N={int(r.N)}:{r.mean:.3f}±{r.ci:.3f}"
                              for r in sub.itertuples())
            print(f"  {method:<20s} {cells}")

    _plot(summ, metrics)
    print(f"\nSaved -> {os.path.relpath(OUT_LONG, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_SUM, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(summ: pd.DataFrame, metrics) -> None:
    style = {
        "MutCount (0 labels)": dict(color=ORANGE, lw=2.6, ls="--", marker="o"),
        "Random": dict(color=GREY, lw=1.6, ls=":", marker=""),
        "Ridge(one-hot)": dict(color=BLUE, lw=2.2, ls="-", marker="o"),
        "Ridge(compact)": dict(color=TEAL, lw=2.2, ls="-", marker="s"),
    }
    titles = {"near_P@30": "A. Near-improver Precision@30 (n=42 tier)",
              "strict_Recall@100": "B. Strict improver Recall@100 (n=5 tier)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, m in zip(axes, metrics):
        s = summ[summ["metric"] == m]
        for method, st in style.items():
            sub = s[s["method"] == method].sort_values("N")
            if sub.empty:
                continue
            ax.plot(sub["N"], sub["mean"], label=method, **st)
            ax.fill_between(sub["N"], sub["lo"], sub["hi"],
                            color=st["color"], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks(list(NS))
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("N labeled variants (log scale)", fontsize=10.5)
        ax.set_ylabel(m, fontsize=10.5)
        ax.set_title(titles[m], fontsize=11.5, loc="left", weight="bold")
        ax.grid(alpha=0.25, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.suptitle("E2 — Does spending N labels beat counting mutations? "
                 "(Absci HER2; mutation count = baseline-to-beat)",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
