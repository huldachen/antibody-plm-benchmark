#!/usr/bin/env python3
"""
E2c -- Group B GP + Group C (our C1 ranker) vs the baselines.

Follow-up (b). Same fixed-held-out-test protocol as E2b (DEC-009: all 5 strict
improvers held out of training), so results are directly comparable. Question:
do a GP, a parent-anchored ridge, or the C1 pairwise-ranking method beat plain
Ridge and the mutation counter — especially at LOW N, where data-efficiency is
the whole pitch of C1?

Methods: MutCount (0 labels), Random, Ridge(compact) [best Group B so far],
GP(compact), Ridge(parent-Δ), C1-Rank(parent-Δ) [Group C, our method].

Run:
    python experiments/e2c_group_c.py
Outputs:
    results/absci_e2c_groupc.csv, results/absci_e2c_groupc_summary.csv
    figures/figE2c_group_c.png
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
from experiments.absci_fewshot_common import (  # noqa: E402
    load_absci, base_scorers, group_c_scorers,
)

OUT_LONG = os.path.join(PROJECT_ROOT, "results", "absci_e2c_groupc.csv")
OUT_SUM = os.path.join(PROJECT_ROOT, "results", "absci_e2c_groupc_summary.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2c_group_c.png")

NS = (20, 50, 100, 200, 500)
N_SEEDS = 20
EVAL_FRAC = 0.40
SPLIT_SEED = 0

STYLE = {
    "MutCount (0 labels)": dict(color="#E8820C", lw=2.6, ls="--", marker="o"),
    "Random": dict(color="#9AA0A6", lw=1.4, ls=":", marker=""),
    "Ridge(compact)": dict(color="#1F6FB2", lw=2.0, ls="-", marker="o"),
    "GP(compact)": dict(color="#7B4FB5", lw=2.0, ls="-", marker="^"),
    "Ridge(parent-Δ)": dict(color="#2E8B8B", lw=2.0, ls="-", marker="s"),
    "C1-Rank(parent-Δ)": dict(color="#C0392B", lw=2.6, ls="-", marker="D"),
}


def main() -> None:
    data = load_absci()
    fit, binder, parent = data["fit"], data["binder"], data["parent"]
    strict_mask, near_mask = data["strict"], data["near"]

    pool_idx, eval_idx = fixed_eval_split(binder, strict_mask, EVAL_FRAC,
                                          SPLIT_SEED)
    print(f"Fixed split (DEC-009): |pool|={pool_idx.size} (0 improvers), "
          f"|eval|={eval_idx.size}, strict_in_eval="
          f"{int(strict_mask[eval_idx].sum())}, near_in_eval="
          f"{int(near_mask[eval_idx].sum())}")

    base = base_scorers(data)
    scorers = {
        "MutCount (0 labels)": base["MutCount (0 labels)"],
        "Random": base["Random"],
        "Ridge(compact)": base["Ridge(compact)"],
        **group_c_scorers(data),
    }

    print(f"Running Group-C learning curve: N={NS}, seeds={N_SEEDS}, "
          f"{len(scorers)} methods ...")
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

    _plot(summ, metrics)
    print(f"\nSaved -> {os.path.relpath(OUT_LONG, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_SUM, PROJECT_ROOT)}, "
          f"{os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(summ, metrics) -> None:
    titles = {"near_P@30": "A. Near-improver Precision@30",
              "strict_Recall@100": "B. Strict improver Recall@100 (n=5, held out)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, m in zip(axes, metrics):
        s = summ[summ["metric"] == m]
        for method, st in STYLE.items():
            sub = s[s["method"] == method].sort_values("N")
            if sub.empty:
                continue
            ax.plot(sub["N"], sub["mean"], label=method, **st)
            ax.fill_between(sub["N"], sub["lo"], sub["hi"],
                            color=st["color"], alpha=0.10)
        ax.set_xscale("log")
        ax.set_xticks(list(NS))
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("N labeled variants (log scale)", fontsize=10.5)
        ax.set_ylabel(m, fontsize=10.5)
        ax.set_title(titles[m], fontsize=11.0, loc="left", weight="bold")
        ax.grid(alpha=0.25, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8.5, loc="best", frameon=False)
    fig.suptitle("E2c — GP + our C1 ranker vs baselines (fixed test, improvers "
                 "held out; Absci HER2)", fontsize=12.0, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
