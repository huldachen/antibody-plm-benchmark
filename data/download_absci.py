#!/usr/bin/env python3
"""
Download and process Absci HER2 trastuzumab dataset.

Source: Shanehsazzadeh et al., "Unlocking de novo antibody design with
generative artificial intelligence", bioRxiv 2023.
GitHub: https://github.com/AbSciBio/unlocking-de-novo-antibody-design

This dataset contains trastuzumab HCDR3 variants with continuous binding
affinity (Kd in nM) measured by SPR, making it ideal for Spearman ρ
benchmarking. Much better than the Mason binary classification data.

Files used:
  - spr-controls.csv: 758 binders (with Kd) + 1097 non-binders
  - zero-shot-binders.csv: 422 zero-shot AI-designed binders (with Kd)

WT trastuzumab HCDR3: SRWGGDGFYAMDY (Kd = 1.94 nM)

Note on CDR-H3 boundaries:
  - Absci format: 13 residues (SRWGGDGFYAMDY) — includes flanking SR and Y
  - Mason format: 10 residues (WGGDGFYAMD) — stripped of SR prefix and Y suffix
  These are the same loop, just different boundary conventions.
"""

import os
import urllib.request
import pandas as pd
import numpy as np

WT_HCDR3 = "SRWGGDGFYAMDY"  # 13 residues, Absci boundary convention

GITHUB_RAW = "https://raw.githubusercontent.com/AbSciBio/unlocking-de-novo-antibody-design/main"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def download_file(url, dest):
    """Download a file from URL to destination."""
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def download_absci_raw():
    """Download raw CSV files from Absci GitHub repo."""
    os.makedirs(RAW_DIR, exist_ok=True)

    files = {
        "absci_spr_controls.csv": f"{GITHUB_RAW}/spr-controls.csv",
        "absci_zero_shot_binders.csv": f"{GITHUB_RAW}/zero-shot-binders.csv",
    }

    for fname, url in files.items():
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest):
            print(f"  {fname} already exists, skipping.")
        else:
            download_file(url, dest)


def count_mutations(seq, wt=WT_HCDR3):
    """Count mutations from WT. Returns -1 for different-length sequences."""
    if len(seq) != len(wt):
        return -1
    return sum(1 for a, b in zip(seq, wt) if a != b)


def process_absci():
    """
    Process Absci data into standardized benchmark format.

    Uses spr-controls.csv as the primary dataset:
      - 758 binders with continuous Kd → Spearman ρ analysis
      - 1097 non-binders → AUC-ROC analysis
      - Fitness metric: -log10(Kd) for binders, NaN for non-binders

    Only includes same-length-as-WT sequences (13 residues) for clean
    position-wise BLOSUM scoring.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    spr_path = os.path.join(RAW_DIR, "absci_spr_controls.csv")
    if not os.path.exists(spr_path):
        raise FileNotFoundError(
            "Raw data not found. Run download_absci_raw() first."
        )

    spr = pd.read_csv(spr_path)
    print(f"  SPR controls: {len(spr)} total")
    print(f"  Binders: {(spr['Binder']==True).sum()}")
    print(f"  Non-binders: {(spr['Binder']==False).sum()}")

    rows = []
    for _, row in spr.iterrows():
        hcdr3 = row["HCDR3"]
        if not isinstance(hcdr3, str):
            continue

        kd_nm = row["KD (nM)"] if row["Binder"] else np.nan
        neg_log_kd = -np.log10(kd_nm * 1e-9) if not np.isnan(kd_nm) else np.nan
        n_mut = count_mutations(hcdr3)
        is_binder = 1 if row["Binder"] else 0

        rows.append({
            "HCDR3": hcdr3,
            "HCDR3_len": len(hcdr3),
            "Binder": is_binder,
            "Kd_nM": kd_nm,
            "neg_log_Kd": neg_log_kd,
            "NumMutations": n_mut,
            "SameLength": 1 if len(hcdr3) == len(WT_HCDR3) else 0,
        })

    df = pd.DataFrame(rows)

    # Sort by neg_log_Kd (best binders first)
    df = df.sort_values("neg_log_Kd", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, "absci_her2_spr.csv")
    df.to_csv(out_path, index=False)

    # Summary
    same_len = df[df["SameLength"] == 1]
    binders = df[df["Binder"] == 1]
    binders_same = same_len[same_len["Binder"] == 1]

    print(f"\n  Processed dataset saved: {out_path}")
    print(f"  Total variants: {len(df)}")
    print(f"  Same length as WT ({len(WT_HCDR3)}): {len(same_len)}")
    print(f"  Binders with Kd: {len(binders)}")
    print(f"  Binders, same length, with Kd: {len(binders_same)}")
    print(f"  Kd range: {binders['Kd_nM'].min():.2f} - {binders['Kd_nM'].max():.2f} nM")
    print(f"  WT Kd: {df.loc[df['HCDR3']==WT_HCDR3, 'Kd_nM'].values[0]:.2f} nM")

    print(f"\n  HCDR3 length distribution:")
    print(df["HCDR3_len"].value_counts().sort_index().to_string())

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Absci HER2 — Trastuzumab HCDR3 SPR Data")
    print("=" * 60)
    print(f"\nWT HCDR3: {WT_HCDR3} (Kd = 1.94 nM)")

    print("\n[1/2] Downloading raw data from GitHub...")
    download_absci_raw()

    print("\n[2/2] Processing into benchmark format...")
    df = process_absci()

    print("\n" + "=" * 60)
    print("Done! Absci data ready for benchmarking.")
    print("=" * 60)
