#!/usr/bin/env python3
"""
E2b -- Fixed-held-out-test robustness check for E2 (DEC-009).

The E2 first cut (sliding window) let strict improvers migrate into the training
draw, shrinking the held-out denominator. Here the evaluation set is FIXED and
holds ALL 5 strict improvers (never trained on), so the strict-tier denominator
is constant and this becomes a clean generalization test: can a model trained
on non-improver labels rank improvers it has never seen — and does it still beat
counting mutations?

Run:
    python experiments/e2b_fixed_test.py
Outputs:
    results/absci_e2b_fixed.csv, results/absci_e2b_fixed_summary.csv
    figures/figE2b_fixed_test.png
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

from evaluation.lowN_protocol import (  # noqa: E402
    fixed_eval_split, run_learning_curve_fixed, summarize,
)
from experiments.absci_fewshot_common import load_absci, base_scorers  # noqa: E402

OUT_LONG = os.path.join(PROJECT_ROOT, "results", "absci_e2b_fixed.csv")
OUT_SUM = os.path.join(PROJECT_ROOT, "results", "absci_e2b_fixed_summary.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2b_fixed_test.png")

NS = (20, 50, 100, 200, 500)
N_SEEDS = 20
EVAL_FRAC = 0.40
SPLIT_SEED = 0

ORANGE = "#E8820C"
GREY = "#6B7280"
BLUE = "#1F6FB2"
TEAL = "#2E8B8B"


def main() -> None:
    data = load_absci()
    fit, binder, parent = data["fit"], data["binder"], data["parent"]
    strict_mask, near_mask = data["strict"], data["near"]

    pool_idx, eval_idx = fixed_eval_split(binder, strict_mask, EVAL_FRAC,
                                          SPLIT_SEED)
    n_strict_eval = int(strict_mask[eval_idx].sum())
    n_near_eval = int(near_mask[eval_idx].sum())
    print(f"Fixed split (DEC-009): |pool|={pool_idx.size} (0 strict improvers), "
          f"|eval|={eval_idx.size}")
    print(f"  eval set holds {n_strict_eval}/5 strict improvers (all of them) "
          f"and {n_near_eval} near-improvers")
    print(f"  training pool binders={int(binder[pool_idx].sum())}\n")

    scorers = base_scorers(data)
    print(f"Running fixed-test learning curve: N={NS}, seeds={N_SEEDS} ...")
    curve = run_learning_curve_fixed(scorers, fit, binder, parent, strict_mask,
                                     near_mask, pool_idx, eval_idx, NS,
                                     n_seeds=N_SEEDS)
    curve.to_csv(OUT_LONG, index=False)

    metrics = ["near_P@30", "strict_Recall@100"]
    summ = pd.concat([summarize(curve, m).assign(metric=m) for m in metrics],
                     ignore_index=True)
    summ.to_csv(OUT_SUM, index=False)

    for m in metrics:
        print(f"\n{m}  (mean ± 95%CI):")
        s = summarize(curve, m)
        for method in scorers:
            sub = s[s["method"] == method].sort_values("N")
            cells = "  ".join(f"N={int(r.N)}:{r.mean:.3f}±{r.ci:.3f}"
                              for r in sub.itertuples())
            print(f"  {method:<20s} {cells}")

    _plot(summ, metrics, n_strict_eval, n_near_eval)
    print(f"\nSaved -> {os.path.relpath(OUT_LONG, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_SUM, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(summ, metrics, n_strict_eval, n_near_eval) -> None:
    style = {
        "MutCount (0 labels)": dict(color=ORANGE, lw=2.6, ls="--", marker="o"),
        "Random": dict(color=GREY, lw=1.6, ls=":", marker=""),
        "Ridge(one-hot)": dict(color=BLUE, lw=2.2, ls="-", marker="o"),
        "Ridge(compact)": dict(color=TEAL, lw=2.2, ls="-", marker="s"),
    }
    titles = {"near_P@30": f"A. Near-improver Precision@30 (n={n_near_eval} in eval)",
              "strict_Recall@100": f"B. Strict improver Recall@100 (n={n_strict_eval}, all held out)"}
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
        ax.set_xlabel("N labeled variants from training pool (log scale)",
                      fontsize=10.5)
        ax.set_ylabel(m, fontsize=10.5)
        ax.set_title(titles[m], fontsize=11.0, loc="left", weight="bold")
        ax.grid(alpha=0.25, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=9, loc="best", frameon=False)
    fig.suptitle("E2b — Fixed held-out test, improvers never trained on: "
                 "does supervision still beat counting? (Absci HER2)",
                 fontsize=12.0, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
