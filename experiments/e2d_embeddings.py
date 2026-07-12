#!/usr/bin/env python3
"""
E2d -- PLM-embedding few-shot vs cheap compact features (Group B i/ii; spec E3).

Follow-up (c). Same fixed-held-out-test protocol (DEC-009). Question (spec E3):
does a simple model on cheap hand features (one-hot + mutation count) match or
beat few-shot ridge on expensive ESM-2 / AbLang embeddings at low N? If the
20-feature-ish compact model rivals a 1280-d PLM embedding, that is the
"you don't need the PLM" result.

Methods: MutCount (0 labels), Random, Ridge(compact), Ridge(ESM-2 emb),
Ridge(AbLang emb). Embeddings come from experiments/extract_embeddings.py.

Run:
    python experiments/e2d_embeddings.py
Outputs:
    results/absci_e2d_emb.csv, results/absci_e2d_emb_summary.csv
    figures/figE2d_embeddings.png
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
    load_absci, base_scorers, embedding_scorers,
)

OUT_LONG = os.path.join(PROJECT_ROOT, "results", "absci_e2d_emb.csv")
OUT_SUM = os.path.join(PROJECT_ROOT, "results", "absci_e2d_emb_summary.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figE2d_embeddings.png")

NS = (20, 50, 100, 200, 500)
N_SEEDS = 20
EVAL_FRAC = 0.40
SPLIT_SEED = 0

STYLE = {
    "MutCount (0 labels)": dict(color="#E8820C", lw=2.6, ls="--", marker="o"),
    "Random": dict(color="#9AA0A6", lw=1.4, ls=":", marker=""),
    "Ridge(compact)": dict(color="#1F6FB2", lw=2.6, ls="-", marker="o"),
    "Ridge(ESM-2 emb)": dict(color="#C0392B", lw=2.0, ls="-", marker="^"),
    "Ridge(AbLang emb)": dict(color="#2E8B57", lw=2.0, ls="-", marker="s"),
}


def main() -> None:
    data = load_absci()
    if data.get("X_esm2") is None or data.get("X_ablang") is None:
        raise SystemExit("Embeddings missing — run experiments/extract_embeddings.py first")
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
        **embedding_scorers(data),
    }
    print(f"Running embedding learning curve: N={NS}, seeds={N_SEEDS}, "
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
    fig.suptitle("E2d — Cheap compact features vs PLM-embedding few-shot "
                 "(fixed test, improvers held out; Absci HER2)",
                 fontsize=12.0, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
