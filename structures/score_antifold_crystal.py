#!/usr/bin/env python3
"""
E4.0 crystal arm — score all Absci variants with AntiFold on the 1N8Z backbone.

RUNS IN THE `antifold` CONDA ENV ONLY (DEC-011: structure tools are isolated
from the analysis env). Hands off to the `plm` env via a CSV; the improver-metric
evaluation reads that CSV.

Method (DEC-010, fixed-backbone IF scoring): AntiFold conditions on the crystal
backbone and emits per-position amino-acid log-probabilities. Every Absci
variant shares the 1N8Z backbone (they differ only in CDR-H3 identity, all
length 13), so we run AntiFold ONCE on 1N8Z and score each variant analytically
as the sum of its CDR-H3 residue log-probs at the CDR-H3 positions (heavy chain
B, PDB residues 97-109 = WT SRWGGDGFYAMDY). custom_chain_mode=True includes the
HER2 antigen (chain C) as context.

Run (in antifold env):
    /path/to/envs/antifold/bin/python structures/score_antifold_crystal.py
Output:
    results/absci_antifold_crystal.csv  (HCDR3, antifold_crystal_score,
                                         antifold_crystal_delta_vs_wt)
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
PDB_DIR = os.path.join(PROJECT_ROOT, "structures", "pdb")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_antifold_crystal.csv")

PDB_NAME = "1n8z"
HCHAIN, LCHAIN = "B", "A"          # from fetch_and_inspect_1n8z.py
WT_HCDR3 = "SRWGGDGFYAMDY"
AA = list("ACDEFGHIKLMNPQRSTVWY")


def main() -> None:
    from antifold.antiscripts import (
        load_model, get_pdbs_logits, get_dfs_HL, get_df_seq,
        df_logits_to_logprobs,
    )

    print("Loading AntiFold model ...")
    model = load_model()

    pdbs_csv = pd.DataFrame([{"pdb": PDB_NAME, "Hchain": HCHAIN, "Lchain": LCHAIN}])
    print(f"Scoring backbone {PDB_NAME} (H={HCHAIN}, L={LCHAIN}, +antigen context) ...")
    df_logits_list = get_pdbs_logits(
        model, pdbs_csv, PDB_DIR, custom_chain_mode=True, save_flag=False,
    )
    df = df_logits_list[0]

    # Heavy chain, per-position log-probs over the 20 AAs.
    df_H, _ = get_dfs_HL(df)
    df_H = df_logits_to_logprobs(df_H).reset_index(drop=True)
    wt_seq = "".join(get_df_seq(df_H))
    print(f"  heavy-chain length {len(wt_seq)}")

    start = wt_seq.find(WT_HCDR3)
    if start < 0:
        raise SystemExit(f"WT HCDR3 {WT_HCDR3} not found in heavy chain sequence")
    L = len(WT_HCDR3)
    print(f"  CDR-H3 at heavy-chain index {start}..{start + L - 1}")

    # Per-position log-prob matrix for the CDR-H3 window (L x 20).
    logprob_window = df_H.loc[start:start + L - 1, AA].to_numpy(dtype=float)
    aa_idx = {a: i for i, a in enumerate(AA)}

    def score_cdr(seq: str) -> float:
        return float(sum(logprob_window[j, aa_idx[r]] for j, r in enumerate(seq)))

    wt_score = score_cdr(WT_HCDR3)
    print(f"  WT CDR-H3 AntiFold log-prob sum = {wt_score:.4f}")

    variants = pd.read_csv(SCORED_CSV)["HCDR3"].astype(str)
    if not (variants.str.len() == L).all():
        raise SystemExit("Not all HCDR3 variants are length 13 — window scoring invalid")
    scores = variants.map(score_cdr)

    out = pd.DataFrame({
        "HCDR3": variants,
        "antifold_crystal_score": scores,
        "antifold_crystal_delta_vs_wt": scores - wt_score,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(out)} variants. Saved -> "
          f"{os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
