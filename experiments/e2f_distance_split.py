#!/usr/bin/env python3
"""
E2f — distance-aware (leave-cluster-out) split: does Ridge still beat counting?

The #1 validity check. E2b held improvers out but not near-duplicates:
39% of eval variants sit within Hamming<=2 of a training variant, and the 5
improvers are only 2-4 mutations from the nearest training variant (see
e2e_leakage_diagnostic). So the Ridge>MutCount headline may be near-memorisation.

Here we cluster CDR-H3s into connected components at Hamming <= D_LINK (near-
duplicate graph) and assign WHOLE components to eval vs training pool, with all
strict improvers forced into eval. Single-linkage components guarantee every
eval variant is >= D_LINK+1 mutations from every pool variant — no near-
duplicate leakage. Then re-run the fixed-test learning curve (Ridge(compact) vs
MutCount) on this distance-separated split.

Run (plm env): python experiments/e2f_distance_split.py
Outputs: results/absci_e2f_distance.csv, results/absci_e2f_distance_summary.csv,
         figures/figE2f_distance_split.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import load_absci, base_scorers  # noqa: E402
from evaluation.lowN_protocol import run_learning_curve_fixed, summarize  # noqa: E402

NS = (20, 50, 100, 200, 500)
N_SEEDS = 20
EVAL_FRAC = 0.40
# D_LINK: components merge if CDR-H3 Hamming <= D_LINK -> eval is >= D_LINK+1
# mutations from any training variant. CLI arg overrides (default 2 = strictest).
D_LINK = int(sys.argv[1]) if len(sys.argv) > 1 else 2
L = 13
OUT_LONG = os.path.join(PROJECT_ROOT, "results", f"absci_e2f_distance_d{D_LINK}.csv")
OUT_SUM = os.path.join(PROJECT_ROOT, "results",
                       f"absci_e2f_distance_d{D_LINK}_summary.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures",
                       f"figE2f_distance_split_d{D_LINK}.png")
STYLE = {
    "MutCount (0 labels)": dict(color="#E8820C", lw=2.6, ls="--", marker="o"),
    "Random": dict(color="#9AA0A6", lw=1.4, ls=":", marker=""),
    "Ridge(compact)": dict(color="#1F6FB2", lw=2.4, ls="-", marker="o"),
}


def build_distance_split(X, binder, strict, d_link, eval_frac, seed=0):
    """Connected components at Hamming<=d_link; whole components to eval/pool,
    improvers forced to eval. Returns (pool_idx, eval_idx, min_cross_dist)."""
    n = X.shape[0]
    D = (L - (X @ X.T)).astype(int)
    adj = csr_matrix((D <= d_link).astype(int))
    n_comp, labels = connected_components(adj, directed=False)

    comps = [np.where(labels == c)[0] for c in range(n_comp)]
    sizes = np.array([len(c) for c in comps])
    print(f"D_LINK={d_link}: {n_comp} components; sizes max={sizes.max()}, "
          f"median={np.median(sizes):.0f}, singletons={(sizes==1).sum()}")

    imp_comp_ids = set(labels[np.where(strict)[0]].tolist())
    eval_ids = set(imp_comp_ids)
    eval_count = sum(sizes[c] for c in eval_ids)
    target = int(eval_frac * n)
    # Add remaining components (deterministic order) until ~eval_frac.
    rng = np.random.default_rng(seed)
    others = [c for c in range(n_comp) if c not in eval_ids]
    rng.shuffle(others)
    for c in others:
        if eval_count >= target:
            break
        eval_ids.add(c)
        eval_count += sizes[c]

    eval_idx = np.sort(np.concatenate([comps[c] for c in eval_ids]))
    pool_idx = np.sort(np.array([i for i in range(n) if labels[i] not in eval_ids],
                                dtype=int))
    min_cross = int(D[np.ix_(eval_idx, pool_idx)].min())
    return pool_idx, eval_idx, min_cross


def main() -> None:
    data = load_absci()
    fit, binder, parent = data["fit"], data["binder"], data["parent"]
    strict, near = data["strict"], data["near"]

    pool_idx, eval_idx, min_cross = build_distance_split(
        data["X_1hot"], binder, strict, D_LINK, EVAL_FRAC)
    print(f"Distance split: |pool|={pool_idx.size}, |eval|={eval_idx.size}, "
          f"strict_in_eval={int(strict[eval_idx].sum())}, "
          f"near_in_eval={int(near[eval_idx].sum())}")
    print(f"min CDR-H3 Hamming(eval, pool) = {min_cross}  "
          f"(>= {D_LINK+1} by construction: no near-duplicate leakage)\n")

    scorers = {k: base_scorers(data)[k]
               for k in ("MutCount (0 labels)", "Random", "Ridge(compact)")}
    curve = run_learning_curve_fixed(scorers, fit, binder, parent, strict, near,
                                     pool_idx, eval_idx, NS, n_seeds=N_SEEDS)
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
            print(f"  {method:<20s} " + "  ".join(
                f"N={int(r.N)}:{r.mean:.3f}±{r.ci:.3f}" for r in sub.itertuples()))

    _plot(summ, metrics, int(strict[eval_idx].sum()), int(near[eval_idx].sum()),
          min_cross)
    print(f"\nSaved -> {os.path.relpath(OUT_PNG, PROJECT_ROOT)}")


def _plot(summ, metrics, n_strict, n_near, min_cross) -> None:
    titles = {"near_P@30": f"A. Near-improver P@30 (n={n_near} in eval)",
              "strict_Recall@100": f"B. Strict Recall@100 (n={n_strict}, held out)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, m in zip(axes, metrics):
        s = summ[summ["metric"] == m]
        for method, st in STYLE.items():
            sub = s[s["method"] == method].sort_values("N")
            if sub.empty:
                continue
            ax.plot(sub["N"], sub["mean"], label=method, **st)
            ax.fill_between(sub["N"], sub["lo"], sub["hi"], color=st["color"],
                            alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks(list(NS)); ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("N labeled variants (log scale)", fontsize=10.5)
        ax.set_ylabel(m, fontsize=10.5)
        ax.set_title(titles[m], fontsize=11.0, loc="left", weight="bold")
        ax.grid(alpha=0.25, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=9, loc="best", frameon=False)
    fig.suptitle("E2f — Distance-aware split (eval >= %d mutations from any "
                 "training variant): does Ridge still beat counting?" % min_cross,
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
