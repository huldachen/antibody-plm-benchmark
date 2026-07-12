#!/usr/bin/env python3
"""
E2e leakage diagnostic — how near-duplicated are held-out eval variants?

The E2b fixed-test held the 5 improvers out (no positive leakage), but in a
combinatorial CDR-H3 library a test variant can still share most of its mutations
with a training variant (sequence-overlap leakage → near-memorisation). This
quantifies it: for the standard fixed-test split, the min CDR-H3 Hamming distance
from each eval variant to the nearest training-pool variant.

Hamming(i,j) = 13 - (#matching positions) via the one-hot Gram matrix.

Run (plm env): python experiments/e2e_leakage_diagnostic.py
Output: results/absci_e2e_leakage.csv, stdout histogram.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.absci_fewshot_common import load_absci  # noqa: E402
from evaluation.lowN_protocol import fixed_eval_split  # noqa: E402

OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e2e_leakage.csv")
EVAL_FRAC = 0.40
SPLIT_SEED = 0
L = 13


def main() -> None:
    data = load_absci()
    X = data["X_1hot"]                     # (n, 13*20) one-hot
    binder, strict = data["binder"], data["strict"]

    # Hamming distance matrix from the one-hot Gram (matches = X@X.T per position).
    matches = X @ X.T
    D = (L - matches).astype(int)          # (n, n) CDR-H3 Hamming distances
    np.fill_diagonal(D, L + 1)             # exclude self

    pool_idx, eval_idx = fixed_eval_split(binder, strict, EVAL_FRAC, SPLIT_SEED)

    # For each eval variant: nearest neighbour in the training POOL.
    min_to_pool = D[np.ix_(eval_idx, pool_idx)].min(axis=1)
    # For reference: global nearest-neighbour distance over the whole pool.
    global_nn = D.min(axis=1)

    def hist(arr, label):
        print(f"\n{label} (n={len(arr)}): min CDR-H3 Hamming distance")
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, counts):
            print(f"  d={v:>2}: {c:4d}  ({100*c/len(arr):4.1f}%)")
        print(f"  median={np.median(arr):.0f}, mean={arr.mean():.2f}, "
              f"frac(d<=2)={100*np.mean(arr<=2):.1f}%")

    hist(global_nn, "ALL variants -> nearest other variant")
    hist(min_to_pool, "EVAL variants -> nearest TRAINING-POOL variant")

    # Improvers specifically.
    imp_eval = np.where(strict[eval_idx])[0]
    imp_min = min_to_pool[imp_eval]
    print(f"\nThe {len(imp_eval)} held-out improvers -> nearest pool variant: "
          f"{sorted(imp_min.tolist())}")

    pd.DataFrame({"eval_idx": eval_idx, "min_hamming_to_pool": min_to_pool,
                  "is_strict_improver": strict[eval_idx].astype(int)}
                 ).to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")
    print("Reading: if most eval variants sit at d<=2 from a training variant, "
          "the fixed-test 'generalization' is partly near-memorisation → a "
          "distance-aware split is needed to trust the Ridge>MutCount headline.")


if __name__ == "__main__":
    main()
