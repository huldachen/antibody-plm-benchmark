# Figures

Publication figures, generated from the results tables by `make_figures.py`
(run from the repo root, after restoring source affinities — see the main README
and `data/raw/README.md`). The shipped PNGs are pre-generated so they render here
without re-running:

    python figures/make_figures.py

| file | what it shows |
|---|---|
| `F1_leakage.png` | Sequence-overlap leakage, measured on the full CDR basis (CDR-H3-only distance overstates near-duplication: 39%→27% within Hamming ≤2, 12→0 exact duplicates). |
| `F2_zeroshot_two_prior.png` | Two edit-distance-like priors: parent-distance baselines (mutation count, BLOSUM62) enrich improvers; the PLMs do not (signed MW enrichment + near-improver AUROC). |
| `F3_structure_counting.png` | Fixed-backbone inverse folding reduces to counting: raw ρ vs partial ρ controlling for mutation count (AntiFold +0.135 → ≈0). |
| `F4_correlation_vs_discovery.png` | The advantage that generalizes (global ρ) is not the one that discovers (near-improver AUROC ≈ counting); Panel B is the decision-relevant metric. |
