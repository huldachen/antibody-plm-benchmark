#!/usr/bin/env python3
"""
Sensitivity checks (reviewer m4/m5) — near-tier cutoff and censoring floor.

(1) Near-improver cutoff: does the zero-shot enrichment conclusion hold at
    top_frac in {5%, 10%, 20%}? (MutCount/BLOSUM enrich; PLMs null/anti.)
(2) Censoring floor (DEC-008): does the leakage-free supervised conclusion
    (Ridge strict Recall@100 > MutCount on the d>=2 distance split) hold for
    floors {min-0.5, min-1, min-2} log units?

Run (plm env): python experiments/e2h_sensitivity.py
Output: results/absci_sensitivity.csv, stdout.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import load_absci, ridge_scorer  # noqa: E402
from evaluation.improver_metrics import (  # noqa: E402
    beat_parent_enrichment, near_improver_mask,
)
from evaluation.lowN_protocol import (  # noqa: E402
    fixed_eval_split, run_learning_curve_fixed, summarize,
)

OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_sensitivity.csv")
ZERO_MODELS = [("MutCount", "MutCount_score"), ("BLOSUM62", "BLOSUM62_score"),
               ("ESM-2", "ESM2_score"), ("AbLang", "AbLang_score"),
               ("AntiBERTy", "AntiBERTy_score")]


def main() -> None:
    data = load_absci()
    df, fit, parent = data["df"], data["fit"], data["parent"]
    df["MutCount_score"] = -df["NumMutations"].astype(float)
    rows = []

    # (1) near-tier cutoff sensitivity — zero-shot enrichment p per cutoff.
    print("(1) Near-tier cutoff — MW enrich p (lower=enriches):")
    print(f"{'cutoff':>8} " + " ".join(f"{m:>10}" for m, _ in ZERO_MODELS))
    for frac in (0.05, 0.10, 0.20):
        mask = near_improver_mask(fit, top_frac=frac)
        ps = {}
        for name, col in ZERO_MODELS:
            e = beat_parent_enrichment(df[col].values, fit, parent,
                                       positive_mask=mask)
            ps[name] = e["MW_p_enrich"]
            rows.append({"check": "near_cutoff", "param": frac, "model": name,
                         "value": e["MW_p_enrich"], "n_pos": int(mask.sum())})
        print(f"{frac:>8.0%} " + " ".join(f"{ps[m]:>10.2e}" for m, _ in ZERO_MODELS))

    # (2) censoring-floor sensitivity — Ridge vs MutCount strict Recall@100,
    #     d>=2 distance split, N=100, 20 seeds.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    X1, binder, strict, near = data["X_1hot"], data["binder"], data["strict"], data["near"]
    D = (13 - (X1 @ X1.T)).astype(int)
    n_comp, labels = connected_components(csr_matrix((D <= 1).astype(int)),
                                          directed=False)
    imp_comps = set(labels[np.where(strict)[0]].tolist())
    eval_ids, cnt, target = set(imp_comps), 0, int(0.40 * len(fit))
    cnt = sum((labels == c).sum() for c in eval_ids)
    rng = np.random.default_rng(0)
    others = [c for c in range(n_comp) if c not in eval_ids]
    rng.shuffle(others)
    for c in others:
        if cnt >= target:
            break
        eval_ids.add(c); cnt += (labels == c).sum()
    eval_idx = np.sort(np.where(np.isin(labels, list(eval_ids)))[0])
    pool_idx = np.sort(np.where(~np.isin(labels, list(eval_ids)))[0])

    print("\n(2) Censoring floor — Ridge vs MutCount strict Recall@100 "
          "(d>=2 split, N=100):")
    base = float(np.nanmin(fit))
    for delta in (0.5, 1.0, 2.0):
        y = np.where(np.isfinite(fit), fit, base - delta)
        scorers = {"Ridge(compact)": ridge_scorer(data["X_compact"], y),
                   "MutCount": lambda tr, te, s: -data["mutcount"][te]}
        curve = run_learning_curve_fixed(scorers, fit, binder, parent, strict,
                                         near, pool_idx, eval_idx, (100,),
                                         n_seeds=20)
        s = summarize(curve, "strict_Recall@100")
        r = s[s.method == "Ridge(compact)"]["mean"].iloc[0]
        m = s[s.method == "MutCount"]["mean"].iloc[0]
        print(f"  floor=min-{delta}: Ridge={r:.3f}  MutCount={m:.3f}  "
              f"{'Ridge wins' if r > m else 'no'}")
        rows.append({"check": "floor", "param": delta, "model": "Ridge-vs-Mut",
                     "value": r - m, "n_pos": 5})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
