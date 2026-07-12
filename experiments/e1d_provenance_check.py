#!/usr/bin/env python3
"""
E1 provenance robustness — do the zero-shot conclusions hold where PLM scores
are exactly correct?

Provenance (models/esm2.py): PLM masked-marginal scores were computed in
full-VH context but embedding ONLY the variant CDR-H3 — the variant's HCDR1/HCDR2
mutations were ignored. So on the 247/1266 Absci variants that carry HCDR1/2
variation, the PLM score is incomplete; on the other 1019 it is exactly correct.
The 5 strict and 42 near improvers are ALL in the correct subset (0 affected),
and Mason is CDR-H3-only (unaffected).

Cheap, rigorous check (no re-score): recompute the E1 enrichment + ESM-2
partial-rho on the HCDR3-only subset (scores exactly correct) and compare to the
full-set numbers. If they hold, the conclusions are not an artifact of the
incomplete scoring.

Run (plm env): python experiments/e1d_provenance_check.py
Output: results/absci_e1d_provenance.csv, stdout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import (  # noqa: E402
    improver_mask, near_improver_mask, beat_parent_enrichment,
)

SCORED = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_e1d_provenance.csv")
WT = {"HCDR1": "GFNIKDTY", "HCDR2": "IYPTNGYT"}
MODELS = [("MutCount", None), ("BLOSUM62", "BLOSUM62_score"),
          ("ESM-2", "ESM2_score"), ("AbLang", "AbLang_score"),
          ("AntiBERTy", "AntiBERTy_score")]


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        B = np.c_[np.ones_like(b), b]
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return pearsonr(resid(rx, rz), resid(ry, rz))


def enrich(df, mask, name, col):
    s = -df["NumMutations"].values if col is None else df[col].values
    e = beat_parent_enrichment(s, df["neg_log_Kd"].values,
                               PARENT, positive_mask=mask)
    return e["MW_p_enrich"], e["MW_p_anti"]


def main() -> None:
    global PARENT
    df = pd.read_csv(SCORED)
    PARENT = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])

    def ham(s, w):
        return sum(a != b for a, b in zip(str(s), w))
    h12 = (df["HCDR1"].apply(lambda s: ham(s, WT["HCDR1"]))
           + df["HCDR2"].apply(lambda s: ham(s, WT["HCDR2"]))).values
    correct = h12 == 0                      # PLM score exactly right here
    print(f"HCDR3-only subset (PLM scores exactly correct): "
          f"{int(correct.sum())}/{len(df)} variants\n")

    rows = []
    for label, sub in [("FULL set", np.ones(len(df), bool)),
                       ("HCDR3-only subset", correct)]:
        d = df[sub].copy()
        fit = d["neg_log_Kd"].values
        strict = improver_mask(fit, PARENT)
        near = near_improver_mask(fit, 0.10)
        print(f"=== {label}  (n={len(d)}, strict={int(strict.sum())}, "
              f"near={int(near.sum())}) ===")
        for name, col in MODELS:
            en_p, an_p = enrich(d, near, name, col)
            es_p, as_p = enrich(d, strict, name, col)
            print(f"  {name:<10} near_enrich={en_p:.2e}  strict_anti={as_p:.3f}")
            rows.append({"set": label, "model": name, "near_enrich_p": en_p,
                         "strict_anti_p": as_p})
        # ESM-2 partial-rho among binders, control = HCDR3 mutation count.
        b = d[d["neg_log_Kd"].notna()]
        pr = partial_spearman(b["ESM2_score"].values, b["neg_log_Kd"].values,
                              b["NumMutations"].values)
        print(f"  ESM-2 partial-rho (binders n={len(b)}, ctrl HCDR3 count): "
              f"{pr[0]:+.3f} (p={pr[1]:.3f})\n")
        rows.append({"set": label, "model": "ESM-2_partial_rho",
                     "near_enrich_p": pr[0], "strict_anti_p": pr[1]})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
