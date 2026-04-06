#!/usr/bin/env python3
"""
BLOSUM62 parental-anchored scorer for antibody variant fitness prediction.

Scoring method:
  For each variant, compute the sum of BLOSUM62 substitution scores at every
  position that differs from the parental (WT) sequence. This is the
  "parental-anchored" approach described in Uçar & Sormanni (2025).

  score(variant) = Σ BLOSUM62(wt_i, mut_i)  for all i where wt_i ≠ mut_i

Key distinction (per briefing):
  - This is NOT global alignment — just position-wise substitution sum.
  - This is NOT consensus-based — it's anchored to the known parental sequence.
  - PWM/PSSM-based scores (without parental anchor) do NOT show the same
    correlation with PLM log-likelihoods.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# BLOSUM62 matrix (standard 20 amino acids)
# ---------------------------------------------------------------------------
# Amino acid order matching standard BLOSUM62
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# BLOSUM62 substitution matrix (symmetric, half-bit units)
# Rows/columns follow AA_ORDER
_BLOSUM62_FLAT = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
      4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,  # A
     -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3,  # R
     -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  # N
     -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  # D
      0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,  # C
     -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  # Q
     -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  # E
      0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3,  # G
     -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  # H
     -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3,  # I
     -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1,  # L
     -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  # K
     -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1,  # M
     -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1,  # F
     -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2,  # P
      1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  # S
      0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  # T
     -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3,  # W
     -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1,  # Y
      0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4,  # V
]

# Build lookup dictionary for fast access
_AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
BLOSUM62 = {}
for i, aa1 in enumerate(AA_ORDER):
    for j, aa2 in enumerate(AA_ORDER):
        BLOSUM62[(aa1, aa2)] = _BLOSUM62_FLAT[i * 20 + j]


def blosum62_score(aa_wt, aa_mut):
    """
    Get BLOSUM62 substitution score for a single amino acid change.

    Parameters
    ----------
    aa_wt : str
        Wild-type amino acid (single letter).
    aa_mut : str
        Mutant amino acid (single letter).

    Returns
    -------
    int
        BLOSUM62 score. Higher = more conservative substitution.
    """
    return BLOSUM62.get((aa_wt, aa_mut), -4)  # -4 default for unknown AAs


def parental_anchored_score(wt_seq, mut_seq):
    """
    Compute parental-anchored BLOSUM62 score for a variant.

    This sums the BLOSUM62 substitution scores at every position where the
    variant differs from the parental (WT) sequence.

    Parameters
    ----------
    wt_seq : str
        Parental (wild-type) amino acid sequence.
    mut_seq : str
        Variant amino acid sequence (same length as wt_seq).

    Returns
    -------
    float
        Sum of BLOSUM62 substitution scores at mutated positions.
        Returns NaN if sequences have different lengths.
    """
    if len(wt_seq) != len(mut_seq):
        return np.nan

    score = 0
    for wt_aa, mut_aa in zip(wt_seq, mut_seq):
        if wt_aa != mut_aa:
            score += blosum62_score(wt_aa, mut_aa)

    return score


def score_variants(df, wt_seq, seq_col="AASeq"):
    """
    Score all variants in a DataFrame using parental-anchored BLOSUM62.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variant sequences.
    wt_seq : str
        Parental (wild-type) sequence.
    seq_col : str
        Column name containing variant amino acid sequences.

    Returns
    -------
    pd.Series
        BLOSUM62 scores for each variant.
    """
    return df[seq_col].apply(lambda seq: parental_anchored_score(wt_seq, seq))


def score_per_position(wt_seq, mut_seq):
    """
    Return per-position BLOSUM62 scores (0 for conserved positions).

    Useful for analyzing which positions drive fitness differences.

    Parameters
    ----------
    wt_seq : str
        Parental sequence.
    mut_seq : str
        Variant sequence.

    Returns
    -------
    list[int]
        BLOSUM62 score at each position (0 if conserved).
    """
    if len(wt_seq) != len(mut_seq):
        return []

    scores = []
    for wt_aa, mut_aa in zip(wt_seq, mut_seq):
        if wt_aa != mut_aa:
            scores.append(blosum62_score(wt_aa, mut_aa))
        else:
            scores.append(0)
    return scores


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wt = "WGGDGFYAMD"

    # Test with some example variants
    test_variants = [
        ("WGGDGFYAMD", "WT (identical)"),
        ("WGGDGFYAME", "1 mutation: D->E (conservative)"),
        ("WGGDGFYAMP", "1 mutation: D->P (non-conservative)"),
        ("FGGDGFYAMD", "1 mutation: W->F"),
        ("FSNVNYYAFA", "8 mutations (heavily mutated)"),
    ]

    print("BLOSUM62 Parental-Anchored Scorer — Self-Test")
    print(f"WT: {wt}")
    print("-" * 60)

    for variant, desc in test_variants:
        score = parental_anchored_score(wt, variant)
        n_mut = sum(1 for a, b in zip(wt, variant) if a != b)
        print(f"  {variant}  score={score:+d}  ({n_mut} mut)  {desc}")

    # Verify known BLOSUM62 values
    assert blosum62_score("W", "W") == 11, "W->W should be 11"
    assert blosum62_score("A", "A") == 4, "A->A should be 4"
    assert blosum62_score("W", "F") == 1, "W->F should be 1"
    assert blosum62_score("D", "E") == 2, "D->E should be 2"
    print("\nAll assertions passed.")
