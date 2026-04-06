#!/usr/bin/env python3
"""
Download and process Mason et al. 2021 trastuzumab scFv DMS data.

Source: Mason et al., "Optimization of therapeutic antibodies by predicting
antigen specificity from antibody sequence via deep learning",
Nature Biomedical Engineering, 2021.

GitHub: https://github.com/dahjan/DMS_opt
Data: CDR-H3 loop variants of trastuzumab, classified as HER2 binders (AgPos)
      vs non-binders (AgNeg) by mammalian display + FACS.

The processed output contains:
  - AASeq: 10-residue CDR-H3 loop sequence (IMGT positions ~107-116)
  - AgClass: 1 = binder, 0 = non-binder
  - CountPos/CountNeg: read counts in positive/negative sort
  - FracPos/FracNeg: frequency in positive/negative sort
  - NumMutations: Hamming distance from WT trastuzumab CDR-H3
  - LogEnrichment: log2(FracPos / FracNeg) for overlapping sequences
"""

import os
import urllib.request
import pandas as pd
import numpy as np

# Trastuzumab WT CDR-H3 (10 mutated positions, IMGT ~107-116)
# Full VH: EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG
#           RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS
# CDR-H3 loop (after stripping CSR prefix and DY suffix per Mason processing):
WT_CDRH3 = "WGGDGFYAMD"

GITHUB_RAW = "https://raw.githubusercontent.com/dahjan/DMS_opt/master/data"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def download_file(url, dest):
    """Download a file from URL to destination."""
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def download_mason_raw():
    """Download raw AgPos and AgNeg CSV files from GitHub."""
    os.makedirs(RAW_DIR, exist_ok=True)

    files = {
        "mHER_H3_AgPos.csv": f"{GITHUB_RAW}/mHER_H3_AgPos.csv",
        "mHER_H3_AgNeg.csv": f"{GITHUB_RAW}/mHER_H3_AgNeg.csv",
    }

    for fname, url in files.items():
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest):
            print(f"  {fname} already exists, skipping.")
        else:
            download_file(url, dest)

    return files.keys()


def count_mutations(seq, wt=WT_CDRH3):
    """Count number of mutations (Hamming distance) from WT."""
    if len(seq) != len(wt):
        return -1
    return sum(1 for a, b in zip(seq, wt) if a != b)


def process_mason():
    """
    Process Mason data into a standardized benchmark format.

    Combines AgPos and AgNeg, computes enrichment scores, and assigns
    mutation counts relative to WT trastuzumab CDR-H3.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw data
    pos_path = os.path.join(RAW_DIR, "mHER_H3_AgPos.csv")
    neg_path = os.path.join(RAW_DIR, "mHER_H3_AgNeg.csv")

    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        raise FileNotFoundError(
            "Raw data not found. Run download_mason_raw() first."
        )

    pos = pd.read_csv(pos_path, index_col=0)
    neg = pd.read_csv(neg_path, index_col=0)

    print(f"  AgPos: {len(pos)} variants")
    print(f"  AgNeg: {len(neg)} variants")

    # Aggregate duplicate AASeq entries before building lookup maps
    pos_agg = pos.groupby("AASeq").agg({"Count": "sum", "Fraction": "sum"})
    neg_agg = neg.groupby("AASeq").agg({"Count": "sum", "Fraction": "sum"})

    # Build lookup maps
    pos_map = pos_agg.to_dict("index")
    neg_map = neg_agg.to_dict("index")

    # Combine all unique sequences
    all_seqs = set(pos["AASeq"].unique()) | set(neg["AASeq"].unique())
    print(f"  Unique sequences: {len(all_seqs)}")

    rows = []
    for seq in all_seqs:
        if not isinstance(seq, str) or len(seq) != 10:
            continue

        p = pos_map.get(seq, {"Count": 0, "Fraction": 0.0})
        n = neg_map.get(seq, {"Count": 0, "Fraction": 0.0})

        n_mut = count_mutations(seq)
        ag_class = 1 if seq in pos_map else 0

        # Compute log-enrichment for sequences in both pools
        # Add pseudocount to avoid log(0)
        frac_pos = p["Fraction"] if p["Fraction"] > 0 else 1e-7
        frac_neg = n["Fraction"] if n["Fraction"] > 0 else 1e-7
        log_enrichment = np.log2(frac_pos / frac_neg)

        rows.append({
            "AASeq": seq,
            "AgClass": ag_class,
            "CountPos": int(p["Count"]),
            "CountNeg": int(n["Count"]),
            "FracPos": p["Fraction"],
            "FracNeg": n["Fraction"],
            "NumMutations": n_mut,
            "LogEnrichment": log_enrichment,
        })

    df = pd.DataFrame(rows)

    # Sort by enrichment (descending)
    df = df.sort_values("LogEnrichment", ascending=False).reset_index(drop=True)

    # Save processed data
    out_path = os.path.join(PROCESSED_DIR, "mason_cdrh3_enrichment.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Processed dataset saved: {out_path}")
    print(f"  Total variants: {len(df)}")
    print(f"  Binders (AgClass=1): {(df['AgClass']==1).sum()}")
    print(f"  Non-binders (AgClass=0): {(df['AgClass']==0).sum()}")
    print(f"  Overlap (in both pools): "
          f"{((df['CountPos']>0) & (df['CountNeg']>0)).sum()}")

    # Summary stats
    print(f"\n  Mutation distance distribution:")
    print(df["NumMutations"].value_counts().sort_index().to_string())

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Mason et al. 2021 — Trastuzumab CDR-H3 DMS Data")
    print("=" * 60)
    print(f"\nWT CDR-H3 (10 positions): {WT_CDRH3}")

    print("\n[1/2] Downloading raw data from GitHub...")
    download_mason_raw()

    print("\n[2/2] Processing into benchmark format...")
    df = process_mason()

    print("\n" + "=" * 60)
    print("Done! Data ready for benchmarking.")
    print("=" * 60)
