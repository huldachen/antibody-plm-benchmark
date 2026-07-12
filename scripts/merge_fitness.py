#!/usr/bin/env python3
"""
Restore source affinity values into the shipped score tables.

The results CSVs in this repo contain our COMPUTED columns only (sequences,
mutation counts, model scores). The source authors' measured values
(Absci Kd / neg_log_Kd / Binder; KyDab neg_log_Kd; Mason AgClass / LogEnrichment /
read counts) are NOT redistributed here — you download them yourself (see
data/raw/README.md) and merge them back in with this script before running the
analysis / figures.

Usage:
    # Absci: join your downloaded Absci table (must contain HCDR3 + Kd_nM) into
    # results/absci_all_scored.csv, restoring Kd_nM / neg_log_Kd / Binder.
    python scripts/merge_fitness.py absci --raw /path/to/absci_her2_spr.csv

    # KyDab: join your downloaded KyDab table (heavy_cdr3 + KD) into
    # results/kydab_scores.csv, restoring neg_log_Kd.
    python scripts/merge_fitness.py kydab --raw /path/to/kydab_processed.csv

    # Mason: join your downloaded Mason table (must contain AASeq + AgClass) into
    # results/mason_all_scored.csv, restoring AgClass / LogEnrichment / counts.
    python scripts/merge_fitness.py mason --raw /path/to/mason_cdrh3.csv

After merging, the experiments in experiments/ and figures/make_figures.py run
unchanged.
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge_absci(raw_path: str) -> None:
    scored = pd.read_csv(os.path.join(ROOT, "results", "absci_all_scored.csv"))
    raw = pd.read_csv(raw_path)
    if "HCDR3" not in raw.columns or "Kd_nM" not in raw.columns:
        raise SystemExit("Absci raw file must contain columns 'HCDR3' and 'Kd_nM'.")
    raw = raw[["HCDR3", "Kd_nM"]].drop_duplicates("HCDR3")
    df = scored.merge(raw, on="HCDR3", how="left")
    df["neg_log_Kd"] = -np.log10(df["Kd_nM"] * 1e-9)   # Kd in nM -> neg_log_Kd (M)
    df["Binder"] = df["Kd_nM"].notna().astype(int)
    out = os.path.join(ROOT, "results", "absci_all_scored.csv")
    df.to_csv(out, index=False)
    print(f"Restored Kd_nM/neg_log_Kd/Binder for {int(df['Binder'].sum())} binders "
          f"({len(df)} rows) -> {os.path.relpath(out, ROOT)}")


def merge_kydab(raw_path: str) -> None:
    scored = pd.read_csv(os.path.join(ROOT, "results", "kydab_scores.csv"))
    raw = pd.read_csv(raw_path)
    key = "heavy_cdr3" if "heavy_cdr3" in raw.columns else "ID"
    kd_col = next((c for c in raw.columns if c.lower() in
                   ("neg_log_kd", "kd", "kd_m", "kd (m)")), None)
    if key not in scored.columns or kd_col is None:
        raise SystemExit("KyDab raw file must contain a join key (heavy_cdr3/ID) "
                         "and a Kd column.")
    raw = raw[[key, kd_col]].drop_duplicates(key).rename(columns={kd_col: "neg_log_Kd"})
    df = scored.merge(raw, on=key, how="left")
    out = os.path.join(ROOT, "results", "kydab_scores.csv")
    df.to_csv(out, index=False)
    print(f"Restored neg_log_Kd for {int(df['neg_log_Kd'].notna().sum())} rows -> "
          f"{os.path.relpath(out, ROOT)}")


MASON_SRC_COLS = ["AgClass", "CountPos", "CountNeg", "FracPos", "FracNeg", "LogEnrichment"]


def merge_mason(raw_path: str) -> None:
    scored = pd.read_csv(os.path.join(ROOT, "results", "mason_all_scored.csv"))
    raw = pd.read_csv(raw_path)
    if "AASeq" not in raw.columns or "AgClass" not in raw.columns:
        raise SystemExit("Mason raw file must contain columns 'AASeq' and 'AgClass'.")
    cols = ["AASeq"] + [c for c in MASON_SRC_COLS if c in raw.columns]
    raw = raw[cols].drop_duplicates("AASeq")
    df = scored.merge(raw, on="AASeq", how="left")
    out = os.path.join(ROOT, "results", "mason_all_scored.csv")
    df.to_csv(out, index=False)
    print(f"Restored {', '.join(cols[1:])} for {int(df['AgClass'].notna().sum())} "
          f"rows -> {os.path.relpath(out, ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["absci", "kydab", "mason"])
    ap.add_argument("--raw", required=True, help="Path to your downloaded source table")
    args = ap.parse_args()
    {"absci": merge_absci, "kydab": merge_kydab, "mason": merge_mason}[args.dataset](args.raw)


if __name__ == "__main__":
    main()
