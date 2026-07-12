#!/usr/bin/env python3
"""
E4 full-CDR re-score — AntiFold over ALL varying heavy CDRs, full-Fab context.

The crystal scorer (score_antifold_crystal.py) scored the HCDR3 window only.
Absci also varies in HCDR1/HCDR2, so that score was incomplete. Here we score
each variant as the sum of AntiFold per-position log-probs over HCDR1 + HCDR2 +
HCDR3, all conditioned on the same full-Fab + antigen context (custom_chain_mode).
ADDITIVE: writes a NEW CSV; the HCDR3-only crystal score is kept.

RUNS IN THE `antifold` CONDA ENV ONLY (DEC-011).

Output: results/absci_antifold_fullcdr.csv (HCDR3, antifold_fullcdr_score,
        antifold_fullcdr_delta_vs_wt)
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
PDB_DIR = os.path.join(PROJECT_ROOT, "structures", "pdb")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "absci_antifold_fullcdr.csv")

PDB_NAME, HCHAIN, LCHAIN = "1n8z", "B", "A"
WT = {"HCDR1": "GFNIKDTY", "HCDR2": "IYPTNGYT", "HCDR3": "SRWGGDGFYAMDY"}
AA = list("ACDEFGHIKLMNPQRSTVWY")


def main() -> None:
    from antifold.antiscripts import (
        load_model, get_pdbs_logits, get_dfs_HL, get_df_seq,
        df_logits_to_logprobs,
    )
    print("Loading AntiFold + scoring 1n8z (full-Fab context) ...")
    model = load_model()
    pdbs_csv = pd.DataFrame([{"pdb": PDB_NAME, "Hchain": HCHAIN, "Lchain": LCHAIN}])
    df_logits = get_pdbs_logits(model, pdbs_csv, PDB_DIR, custom_chain_mode=True,
                                save_flag=False)[0]
    df_H = df_logits_to_logprobs(get_dfs_HL(df_logits)[0]).reset_index(drop=True)
    wt_seq = "".join(get_df_seq(df_H))
    aa_idx = {a: i for i, a in enumerate(AA)}

    # Locate each WT CDR window in the heavy-chain sequence.
    windows = {}
    for cdr, wtseq in WT.items():
        start = wt_seq.find(wtseq)
        if start < 0:
            raise SystemExit(f"{cdr} ({wtseq}) not found in heavy chain")
        windows[cdr] = (start, len(wtseq),
                        df_H.loc[start:start + len(wtseq) - 1, AA].to_numpy(float))
        print(f"  {cdr}: heavy index {start}..{start + len(wtseq) - 1}")

    def score(row) -> float:
        s = 0.0
        for cdr, (_, L, W) in windows.items():
            seq = str(row[cdr])
            if len(seq) != L:
                return np.nan
            s += sum(W[j, aa_idx[a]] for j, a in enumerate(seq))
        return s

    wt_score = sum(sum(W[j, aa_idx[a]] for j, a in enumerate(WT[cdr]))
                   for cdr, (_, _, W) in windows.items())
    print(f"  WT full-CDR AntiFold log-prob = {wt_score:.4f}")

    df = pd.read_csv(SCORED_CSV)
    sc = df.apply(score, axis=1)
    out = pd.DataFrame({"HCDR3": df["HCDR3"], "antifold_fullcdr_score": sc,
                        "antifold_fullcdr_delta_vs_wt": sc - wt_score})
    out.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(out)} variants ({int(sc.isna().sum())} NaN). Saved -> "
          f"{os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
