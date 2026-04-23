# Antibody PLM Benchmark

**Can protein language models predict antibody fitness better than a substitution matrix?**

A systematic benchmark of antibody-specific PLMs (AbLang, AntiBERTy) vs. a general-purpose PLM (ESM-2) vs. BLOSUM62 for variant fitness prediction on combinatorial CDR-H3 libraries.

> **Full analysis narrative:** [`notebooks/lab_notebook.md`](notebooks/lab_notebook.md) — 15 entries covering dataset selection, scorer validation, statistical deep-dive, literature comparison, and interpretation of every result below.

<p align="center">
  <img src="figures/fig2_auc_comparison.png" width="700" alt="AUC-ROC comparison across models and datasets">
</p>

## Key Findings

**BLOSUM62 is the only model that beats counting mutations** — and only on one of two datasets. After controlling for the mutation-count confound, no protein language model significantly predicts continuous antibody fitness.

Specifically:

1. **Mutation count is a devastating trivial baseline.** On Mason (binary FACS data), simply counting how many mutations a variant has from the parental sequence achieves AUC = 0.750 — higher than any model. On Absci (quantitative SPR), mutation count still reaches AUC = 0.634.

2. **No PLM provides unique information beyond BLOSUM62.** Ensembling any PLM with BLOSUM62 always degrades performance. The antibody-specific models (AbLang, AntiBERTy) are nearly redundant with each other (Spearman ρ = 0.88), both trained on the Observed Antibody Space.

3. **ESM-2 anti-predicts on quantitative affinity data.** After partialling out mutation count, ESM-2 shows a significant *negative* correlation with fitness (partial ρ = −0.101) on Absci — it systematically assigns higher scores to weaker binders. This likely reflects its general evolutionary prior penalizing the hypervariability that CDR-H3 requires.

4. **Assay resolution determines whether sequence-aware models add value.** Binary FACS sorting → just count mutations. Quantitative SPR → BLOSUM62 substitution identity adds value. PLMs add nothing in either case.

5. **Antibody-specific PLMs rank *improvers* below average.** On Absci, only 5 of 420 binders strictly beat parental trastuzumab's Kd (1.94 nM). Mutation count significantly enriches these 5 improvers (Mann-Whitney p = 0.028); AbLang significantly anti-predicts them (p = 0.034), with ESM-2 and AntiBERTy trending in the same direction. Binary classification metrics (AUC, AUPRC, F1) mask this failure mode — a model can distinguish binders from non-binders while ranking the *useful* binders below the useless ones.

## Project Iterations

This is a **living benchmark** — each iteration adds rigor in response to community feedback. The scored-variant CSVs in `results/` are treated as the durable artefact; metrics are recomputed on top of them via `reevaluate.py`, so new metric ideas do not require re-running any PLM.

| Version | Scope added | Motivation |
|---|---|---|
| **v1** | 4 models × 2 datasets, AUC-ROC + Spearman ρ, MutCount confound analysis | Initial release |
| **v2** *(current)* | AUPRC, best-F1, Kendall τ, Precision@K (K=30/100/top-5%), beat-parent P@K, MutCount as first-class baseline | LinkedIn feedback: class imbalance, lab-workflow metrics, tie handling |
| **v3** *(planned)* | AbLang-2 (germline-bias-corrected); autoregressive PLM scoring (IgLM / p-IgGen); KyDab natural-repertoire dataset (51 immunogens, paired VH/VL) | Isolate germline bias from CDR3 scoring failure; masked-marginal ambiguity for multi-mutation variants; single-target generalization gap |

See the [lab notebook](notebooks/lab_notebook.md) for the reasoning and experiments behind each iteration.

## Datasets

| Property | Mason et al. 2021 | Absci HER2 SPR |
|---|---|---|
| Antibody | Trastuzumab scFv | Trastuzumab IgG |
| CDR-H3 length | 10 residues | 13 residues |
| Library type | Combinatorial | Combinatorial |
| Variants scored | 500 (subsample) | 1,266 (full, same-length) |
| Fitness readout | Binary FACS (binder/non-binder) | Quantitative SPR (Kd in nM) |
| Mutations/variant | Median 6–7 | Median 7–8 |
| Source | [Mason et al., Nature BME 2021](https://github.com/dahjan/DMS_opt) | [Engelhart et al., 2022](https://doi.org/10.1101/2022.01.31.478500) |

Both datasets are **combinatorial CDR-H3 libraries** — most variants carry 4–9 simultaneous mutations from the parental sequence. This distinguishes them from typical deep mutational scanning (single-point) studies in benchmarks like ProteinGym.

## Models

| Model | Type | Training Data | Scoring Method |
|---|---|---|---|
| **BLOSUM62** | Substitution matrix | Curated protein alignments | Sum of BLOSUM62(wt_i, mut_i) at mutated positions |
| **ESM-2 (650M)** | General protein PLM | UniRef50 (>60M sequences) | Masked marginal: Σ [log P(mut_i) − log P(wt_i)] |
| **AbLang-1** *(pkg `ablang` v0.3.1)* | Antibody-specific PLM | OAS (>500M antibody sequences; 42% naive + 39% unsorted B-cells) | Masked marginal (heavy chain) |
| **AntiBERTy** | Antibody-specific PLM | OAS (>500M antibody sequences) | Masked marginal (heavy chain) |

## Results

### Classification (AUC-ROC)

| Model | Mason (n=500) | Absci (n=1,266) |
|---|---|---|
| MutCount (trivial) | **0.750** | 0.634 |
| BLOSUM62 | 0.568 | **0.683** |
| ESM-2 | 0.523 | 0.561 |
| AbLang | 0.601 | 0.565 |
| AntiBERTy | 0.597 | 0.550 |

### Continuous Fitness Correlation (Spearman ρ)

| Model | Mason raw ρ | Mason partial ρ | Absci raw ρ | Absci partial ρ |
|---|---|---|---|---|
| BLOSUM62 | +0.049 | +0.047 | +0.027 | −0.066 |
| ESM-2 | +0.013 | −0.025 | −0.054 | **−0.101** |
| AbLang | +0.119 | +0.045 | +0.015 | −0.024 |
| AntiBERTy | +0.110 | +0.083 | −0.031 | −0.049 |

*Partial ρ controls for mutation count. Absci continuous fitness uses binders only (n=420) since non-binders lack measured Kd.*

### Lab-workflow metrics (v2)

These metrics reflect how discovery actually works: a lab scientist picks the top-K predictions and sends them for experimental follow-up. Added in response to LinkedIn feedback on the v1 release.

**Why each metric was added — and what new information it provided:**

| Metric | Why added | What it revealed |
|---|---|---|
| **AUPRC** (area under precision-recall curve) | v1 used AUC-ROC, which is symmetric across classes and can look respectable even when the model is bad at retrieving the minority (positive) class. AUPRC is dominated by top-of-ranking performance on positives; its random-retrieval baseline equals the prevalence. | **v1 conclusions hold up.** AUPRC tracks AUC-ROC qualitatively — MutCount still wins on Mason, BLOSUM62 still wins on Absci, PLMs hover near the prevalence floor. |
| **Best F1** (with prevalence reported alongside) | AUC/AUPRC are threshold-free. F1 reports a single-point, threshold-aware summary. Pairing it with prevalence lets the reader compare to the always-predict-positive baseline (F1 ≈ 0.50 at prev = 0.33). | All models sit near that trivial baseline — no one finds a clearly separating threshold. |
| **Kendall τ** (tie-aware rank correlation) | Spearman ρ can be inflated by a small number of high-fitness outliers; Kendall τ weights every pairwise comparison equally and handles ties explicitly (important given MutCount's integer-valued scores). | **v1 Spearman results are not outlier-driven.** Kendall and Spearman agree in sign and scale in every case. |
| **Precision@K** (K = 30, 100, top-5%) + tie diagnostic | The metric that matches real lab decisions: given a finite synthesis budget, what fraction of your top-K picks are binders? AUC-ROC treats rank 50,000 vs. 60,000 the same as rank 30 vs. 31 — meaningless for a lab. Tiebreak-aware: averages over 20 random shuffles and reports the number of tied scores at the boundary. | **Integer-scored models (MutCount, BLOSUM) have many ties** — 32 and 14 variants respectively share the K=30 boundary. Without the diagnostic, their P@K numbers look better than they really are. |
| **Beat-parent Precision@K + improver-rank diagnostic** (Absci only) | Binary binder/non-binder is not discovery — it's a filter. A lab with a working lead (trastuzumab, Kd = 1.94 nM) doesn't need another 100-nM binder; they need an *improver*. With only 5 improvers in 1266 variants, P@30 is noisy — so we complement it with a Mann-Whitney rank test asking whether the 5 improvers are systematically ranked higher or lower than the other 1261 variants. | **The strongest finding of v2.** MutCount significantly *enriches* improvers (p = 0.028); AbLang significantly *anti-predicts* them (p = 0.034). ESM-2 trends anti-predictive (p = 0.073). Binary-classification metrics (AUC ~0.55–0.68, AUPRC ~0.37–0.49) hid this entirely. |

Detailed metric definitions, formulae, and full rationale are in [Entry 16 of the lab notebook](notebooks/lab_notebook.md).

**Precision@30 — of the top-30 picks, how many are binders?**

| Model | Mason P@30 | Absci P@30 | Mason AUPRC | Absci AUPRC |
|---|---|---|---|---|
| MutCount | **0.682**† | **0.853**† | 0.536 | 0.470 |
| BLOSUM62 | 0.335 | 0.667 | 0.350 | 0.493 |
| ESM-2    | 0.467 | 0.387 | 0.348 | 0.369 |
| AbLang   | 0.467 | 0.533 | 0.390 | 0.409 |
| AntiBERTy| 0.367 | 0.567 | 0.386 | 0.388 |
| *prevalence* | *0.308* | *0.332* | *—* | *—* |

*† MutCount produces many tied ranks (32 boundary-ties on Mason, 14 on Absci). Reported values are averaged over 20 random tiebreak shuffles. Full metrics (Kendall τ, best-F1, P@100, P@top-5%) are in `results/*_v2_metrics.csv`.*

**Beat-parent analysis (Absci) — can a model surface variants that *improve on* the parental trastuzumab (Kd = 1.94 nM)?**

Only **5 of 1266 variants** (5 of 420 binders) strictly improve on the parent — a 0.4% base rate overall, 1.2% among binders. With n = 5, P@30 has a granularity of 1/30 and most models bottom out at 0.000, so we complement it with the ranks of all 5 improvers across the 1266-variant list (null-hypothesis median rank = 633) and a Mann-Whitney U test.

| Model | P@30 beat-WT | Improver ranks (out of 1266) | Median rank | MW p (enrich) | MW p (anti-predict) |
|---|---|---|---|---|---|
| MutCount  | 0.067 (2/30) | 3, 4, 447, 448, 713        |  447 | **0.028** | 0.972 |
| BLOSUM62  | 0.000 (0/30) | 111, 174, 237, 784, 863    |  237 | 0.114     | 0.886 |
| ESM-2     | 0.000 (0/30) | 597, 747, 945, 963, 1113   |  945 | 0.927     | 0.073 |
| AbLang    | 0.000 (0/30) | 606, 913, 959, 975, 1204   |  959 | 0.967     | **0.034** |
| AntiBERTy | 0.000 (0/30) | 113, 460, 1106, 1158, 1178 | 1106 | 0.846     | 0.155 |

**MutCount significantly enriches for improvers (p = 0.028).** Distance-from-parent is a real signal — near-WT variants disproportionately include the improvers.

**AbLang significantly anti-predicts improvers (p = 0.034); ESM-2 trends in the same direction (p = 0.073); AntiBERTy shows the most extreme median rank (1106) but is blunted by a single outlier.** All three PLMs rank the 5 improvers *below* the dataset median — i.e., they rank the most useful variants as if they were the least useful. This consolidates v1's negative partial-ρ finding for ESM-2 on Absci: the same phenomenon is visible across multiple PLMs when viewed through a discovery-relevant lens.

Binary classification metrics (AUC 0.55–0.68, AUPRC 0.37–0.49, best F1 0.51–0.55) masked this entirely: these models *can* distinguish binders from non-binders at some level, but when asked which binders are genuinely better than the starting antibody, they actively rank them wrong.

Reproduce with: `python reevaluate.py --dataset absci --beat-parent`. Per-variant improver ranks are saved to `results/absci_improver_ranks.csv`.

## Figures

| | |
|---|---|
| ![Fig 1](figures/fig1_dataset_characterization.png) | ![Fig 2](figures/fig2_auc_comparison.png) |
| **Fig 1.** Dataset characterization | **Fig 2.** AUC-ROC comparison |
| ![Fig 3](figures/fig3_confound_scatter.png) | ![Fig 4](figures/fig4_partial_correlation.png) |
| **Fig 3.** Score–mutation count confound | **Fig 4.** Raw vs. partial correlation |

<p align="center">
  <img src="figures/fig5_model_agreement.png" width="700" alt="Inter-model agreement">
</p>
<p align="center"><b>Fig 5.</b> Inter-model agreement heatmaps</p>

Detailed figure interpretations are in [`figures/figure_descriptions.txt`](figures/figure_descriptions.txt).

## Quickstart

```bash
# Clone and set up environment
git clone https://github.com/huldachen/antibody-plm-benchmark.git
cd antibody-plm-benchmark
conda env create -f environment.yml
conda activate ab-plm-bench

# Download datasets
python data/download_mason.py
python data/download_absci.py

# Run BLOSUM62 baseline (no GPU needed)
python run_benchmark.py --dataset mason --models blosum
python run_benchmark.py --dataset absci --models blosum

# Run all models (requires GPU for PLMs)
python run_benchmark.py --dataset mason --models blosum esm2 ablang antiberty
python run_benchmark.py --dataset absci --models blosum esm2 ablang antiberty

# Recompute v2 metrics (AUPRC, F1, Kendall τ, Precision@K) from already-scored CSVs
# — no GPU needed, runs in seconds.
python reevaluate.py --dataset mason
python reevaluate.py --dataset absci --beat-parent
```

## Project Structure

```
antibody-plm-benchmark/
├── README.md
├── environment.yml
├── run_benchmark.py             # Main CLI — score + evaluate
├── reevaluate.py                # Recompute v2 metrics from scored CSVs (no GPU)
├── data/
│   ├── download_mason.py
│   ├── download_absci.py
│   ├── raw/                     # Original downloaded CSVs
│   └── processed/               # Cleaned datasets
├── models/
│   ├── blosum.py                # BLOSUM62 scorer
│   ├── esm2.py                  # ESM-2 masked marginal scorer
│   ├── ablang_scorer.py         # AbLang scorer
│   └── antiberty_scorer.py      # AntiBERTy scorer
├── evaluation/
│   ├── metrics.py               # AUC, Spearman, bootstrap CI
│   └── stratify.py              # CDR/framework stratification
├── results/                     # Scored variant CSVs
├── figures/                     # Publication-quality figures
└── notebooks/
    └── lab_notebook.md          # Full analysis narrative
```

## Context: Why This Matters

Standard PLM benchmarks like [ProteinGym](https://proteingym.org/) (Notin et al., 2023) overwhelmingly evaluate on single-point mutations, where additive models perform well by construction. Real antibody engineering campaigns produce **combinatorial libraries** with many simultaneous mutations, especially in CDR-H3 — the primary determinant of antigen specificity.

Our results suggest that on combinatorial CDR-H3 data, current PLMs (both general and antibody-specific) fail to capture the epistatic interactions that determine binding. The field's reported PLM accuracies may not transfer to the multi-mutation regimes that matter in practice.

**Why the antibody-specific PLMs can underperform — OAS germline bias.** A natural question is why AbLang-1 and AntiBERTy — explicitly antibody-trained — rank improvers *worse* than the general-purpose ESM-2 on Absci. The mechanism is documented: OAS is 42% naive + 39% unsorted B-cells, so the training distribution is dominated by germline-close sequences. [Olsen et al. (*Bioinformatics* 2024)](https://academic.oup.com/bioinformatics/article/40/11/btae618/7845256) measured that AntiBERTy and AbLang-1 predict germline 86.7% / 84.9% of the time at non-germline positions, with **<2% cumulative probability** for valid non-germline residues. Engineered therapeutic improvers (11–20+ non-germline mutations) are therefore far outside the training distribution; the model's "likelihood" functions as an inverted proxy for engineered binding fitness. **AbLang-2** (same authors) added focal loss to partially fix this — but the paper explicitly flags that it remains **"less informative for variants involving CDR3 changes"**, which is precisely our benchmark setting. [Kenlay et al. (*PLoS Comp Bio* 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262217/) showed that a small nucleotide-level CNN outperforms AbLang2 and ESM-1v on affinity maturation, because somatic hypermutation is nucleotide-driven and protein PLMs cannot access that signal.

Related work:

- Qiu & Marks (2025) — Scaling PLMs improves single-mutant but not multi-mutant fitness prediction
- Notin et al. (2023) — ProteinGym benchmark (predominantly single-point DMS)
- AbMAP (PNAS 2025) — Antibody embedding benchmarks, similar limitations in combinatorial regimes
- Kenlay et al. (PLoS Comp Bio 2025) — Nucleotide-level context models for antibody engineering

## Limitations & Future Work

**CDR vs. framework stratification.** The original hypothesis was that PLMs would underperform in CDRs (shaped by VDJ recombination, not phylogenetic evolution) but retain value in conserved framework regions. Both datasets here contain only CDR-H3 mutations, so this comparison was not testable. The stratification module (`evaluation/stratify.py`) is implemented and ready for datasets with framework mutations (e.g., Warszawski et al.). This remains the most interesting open question.

**Mason subsample size.** Mason results use a 500-variant subsample of the full 36k library. Full-dataset BLOSUM analysis confirmed the signal (partial ρ = +0.069, p < 0.001 on 36k), but running all four PLMs on the full set would strengthen the conclusions.

**Single antibody target.** Both datasets target HER2 via trastuzumab variants. Generalization to other targets and antibody scaffolds is unknown. The planned v3 integration of [KyDab](https://kydab.naturalantibody.com/) — 51 immunogens, natural-repertoire antibodies — directly addresses this.

**Ranking ≠ generation.** One piece of community feedback reframed the v1 conclusion: **70–80% score correlation between BLOSUM and PLMs is normal** across many molecule classes, and the *additional* value of PLMs is not ranking accuracy but generative design — proposing plausible single-substitutions, paired chain generation, humanization, etc. That use case is not tested here. The honest statement is "for ranking combinatorial CDR-H3 variants, PLMs ≈ BLOSUM," not "PLMs are useless."

**Masked-marginal scoring is ambiguous for multi-mutation variants.** The masked marginal treats each mutated position independently, which is theoretically mis-specified when many positions are mutated together (as in these combinatorial libraries). An autoregressive PLM (factorized scoring via chain rule) is more principled for this regime and is on the v3 roadmap.

## License

MIT
