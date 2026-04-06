#!/usr/bin/env python3
"""
CDR vs Framework position assignment using IMGT numbering.

For production use with full-length antibody sequences, this module provides
an ANARCI wrapper (requires `pip install anarci`). For the Mason dataset
specifically, all 10 mutated positions are within CDR-H3 (IMGT ~107-116),
so stratification is trivial.

For datasets like Warszawski (framework mutations), ANARCI annotation
becomes essential.

IMGT CDR definitions (heavy chain):
  CDR-H1: positions 27-38
  CDR-H2: positions 56-65
  CDR-H3: positions 105-117
  Framework: everything else in positions 1-128
"""

# IMGT CDR boundaries (heavy chain)
IMGT_CDR_H = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

# IMGT CDR boundaries (light chain)
IMGT_CDR_L = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


def classify_imgt_position(pos, chain="H"):
    """
    Classify an IMGT-numbered position as CDR or Framework.

    Parameters
    ----------
    pos : int
        IMGT position number.
    chain : str
        'H' for heavy chain, 'L' for light chain.

    Returns
    -------
    str
        Region label: 'CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4'.
    """
    cdrs = IMGT_CDR_H if chain == "H" else IMGT_CDR_L

    for cdr_name, (start, end) in cdrs.items():
        if start <= pos <= end:
            return cdr_name

    # Framework regions
    if pos < 27:
        return "FR1"
    elif pos < 56:
        return "FR2"
    elif pos < 105:
        return "FR3"
    else:
        return "FR4"


def label_mason_positions():
    """
    Label positions for Mason CDR-H3 dataset.

    All 10 positions in the Mason dataset correspond to CDR-H3
    (IMGT positions ~107-116). This returns a simple constant label.

    Returns
    -------
    str
        "CDR3" for all Mason positions.
    """
    return "CDR3"


def annotate_with_anarci(sequence, scheme="imgt", chain_type="H"):
    """
    Annotate a full antibody variable domain sequence using ANARCI.

    Requires: pip install anarci

    Parameters
    ----------
    sequence : str
        Full VH or VL amino acid sequence.
    scheme : str
        Numbering scheme ('imgt', 'kabat', 'chothia', 'martin').
    chain_type : str
        'H' for heavy, 'L' or 'K' for light.

    Returns
    -------
    list[tuple]
        List of (imgt_position, amino_acid, region_label) tuples.
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        raise ImportError(
            "ANARCI is required for full-sequence annotation. "
            "Install with: pip install anarci"
        )

    # Run ANARCI
    results = run_anarci(
        [("query", sequence)],
        scheme=scheme,
        output=False,
    )

    if results[0] is None:
        raise ValueError(f"ANARCI failed to number sequence: {sequence[:30]}...")

    numbering = results[0][0][0]  # First sequence, first domain

    annotated = []
    for (pos, insertion), aa in numbering:
        if aa == "-":
            continue
        region = classify_imgt_position(pos, chain=chain_type)
        annotated.append((pos, aa, region))

    return annotated


def get_mutation_regions(wt_seq, mut_seq, wt_annotation):
    """
    Determine which CDR/FR regions each mutation falls in.

    Parameters
    ----------
    wt_seq : str
        Wild-type sequence.
    mut_seq : str
        Mutant sequence.
    wt_annotation : list[tuple]
        Output of annotate_with_anarci for the WT sequence.

    Returns
    -------
    list[dict]
        For each mutation: position, wt_aa, mut_aa, region.
    """
    mutations = []
    # Build position map from annotation
    pos_to_region = {i: ann[2] for i, ann in enumerate(wt_annotation)
                     if i < len(wt_seq)}

    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            region = pos_to_region.get(i, "unknown")
            mutations.append({
                "position": i,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "region": region,
            })

    return mutations


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("CDR/FR Stratification — Self-Test")
    print("-" * 40)

    # Test IMGT position classification
    test_positions = [
        (1, "FR1"), (26, "FR1"),
        (27, "CDR1"), (38, "CDR1"),
        (39, "FR2"), (55, "FR2"),
        (56, "CDR2"), (65, "CDR2"),
        (66, "FR3"), (104, "FR3"),
        (105, "CDR3"), (117, "CDR3"),
        (118, "FR4"), (128, "FR4"),
    ]

    for pos, expected in test_positions:
        result = classify_imgt_position(pos, "H")
        status = "OK" if result == expected else f"FAIL (got {result})"
        print(f"  Position {pos:3d} -> {result:5s} [{status}]")

    # Mason dataset: all positions are CDR-H3
    print(f"\nMason dataset position type: {label_mason_positions()}")
    print("(All 10 mutated positions are CDR-H3, IMGT ~107-116)")
