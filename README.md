# Antibody PLM Benchmark

**No method here robustly beats a mutation counter at retrieving affinity improvers — and standard evaluation hides this.**

I benchmark PLMs, inverse-folding models, and low-N supervised regressors against trivial baselines for antibody affinity maturation. The contribution is an evaluation protocol — distance-aware splits, budget-matched precision@K, improver-tier separation — that exposes failure modes a global correlation on random splits conceals.

<p align="center">
  <img src="figures/F4_correlation_vs_discovery.png" width="760"
       alt="The advantage that generalizes (correlation) is not the one that discovers">
</p>

---

## Why this matters

Antibody affinity maturation is resource-bound: from a large combinatorial CDR library, a team can only synthesize and assay a small number of variants per round. The decision that consumes wet-lab budget is not "predict every Kd" but "put the rare improvers at the top of a short list." A model can look good on global correlation and still be useless at that decision.

Standard benchmarks inflate apparent PLM performance on combinatorial CDR data in two ways: fitness correlates with edit distance from the parent, so any score tracking mutation count inherits signal; and random splits place near-duplicates in the test set, so the model memorizes rather than generalizes. I designed this benchmark to separate these confounds.

## Research questions

| | Question | Verdict |
|---|---|---|
| **Q1** | Do zero-shot PLMs or inverse folding beat a mutation counter at enriching improvers? | No. Only parent-distance baselines significantly enrich. |
| **Q2** | Does fixed-backbone inverse folding add signal beyond mutation count? | No. AntiFold's correlation vanishes under partial-ρ control. |
| **Q3** | Does low-N supervision convert a correlation edge into a discovery edge? | Not robustly. Ridge keeps correlation but ≈ counting under stringent precision@K. |
| **Q4** | Do conclusions hold across metric, label budget, and task? | Partially — PLM effect *flips* between binder vs. improver retrieval. |

## Key findings

**Two priors, only one enriches improvers.** Mutation count (MW p = 3.2×10⁻⁷) and BLOSUM62 (p = 1.9×10⁻³) significantly enrich improvers; antibody PLMs do not. PLMs encode a germline-naturalness prior, not the parent-distance prior that matters here. The effect **flips with the task**: on Mason, the same PLMs *enrich* binders (p ≈ 10⁻⁴) because binders there are low-mutation.

**Structure reduces to counting.** AntiFold on the trastuzumab–HER2 crystal out-enriches BLOSUM, yet its correlation vanishes once mutation count is controlled (raw ρ = +0.135 → partial ρ = +0.003, n.s.; ρ = −0.71 with mutation count). Whether reported inverse-folding advantages are partly a mutation-count artifact is a testable question.

**Correlation ≠ discovery.** With leakage removed via distance-aware splits, ridge regression keeps a correlation advantage (ρ 0.68 → 0.48 vs. a flat counter) but matches counting under stringent, budget-matched precision@K.

## Models

| Model | Type | Why included |
|---|---|---|
| **Mutation count** | parent-distance | The baseline every method must beat |
| **BLOSUM62** | substitution-scored distance | Adds biochemistry to distance |
| **ESM-2 (650M)** | general PLM, masked-marginal | Most-used zero-shot ranker |
| **AbLang, AntiBERTy** | antibody-specific PLMs | Do antibody LMs help? |
| **AntiFold** | inverse folding (PDB 1N8Z) | Fixed-backbone structure prior |
| **Ridge** (one-hot / compact) | low-N supervised | Does spending a few labels beat counting? |

## Evaluation protocol

A pre-campaign checklist for vetting any affinity-ranking method:

1. Report a **mutation-count baseline** (and BLOSUM62)
2. Use **distance-/cluster-aware splits**; quantify near-duplicate leakage
3. Report **budget-matched precision@K**, not only global correlation
4. **Separate the tiers** — binder vs. beat-parent (improver) retrieval
5. **Control for mutation count** via partial correlation
6. **Correct for multiple comparisons**; label borderline effects nominal

## What I corrected

Several early numbers moved under scrutiny. Corrected values below; full log in `validation_checklist.md`.

- **Full-CDR basis.** First analysis scored CDR-H3 only; Absci variants also vary in HCDR1/HCDR2. Re-ran on the complete 29-position basis. Conclusions held.
- **Leakage overstated.** CDR-H3-only distance exaggerated near-duplication: within-Hamming-≤2 overlap 39% → 27%, exact duplicates 14 → 0.
- **A signal demoted.** ESM-2's partial correlation was nominally anti-fitness on the CDR-H3 basis (ρ = −0.11, p = 0.02) but did not survive full-CDR re-scoring. Dropped.
- **Tie correction.** Strict-tier enrichment p rested on a score tie; midrank Mann–Whitney moved it 0.028 → 0.075. Now applied to every rank statistic.
- **Mason flipped the thesis.** Expected the anti-ranking to replicate; instead PLMs *enrich* binders there. The sign of the PLM effect depends on the task — this became load-bearing, not a failed replication.
- **Multiple comparisons.** Only the strong effects survive Bonferroni; borderline effects labelled nominal.

## Practical implications

- **Check the trivial baseline first.** Before trusting any zero-shot model to rank affinity-maturation variants, verify it beats "fewer mutations from the parent." Most here do not.
- **Route models by objective.** Antibody PLMs help for humanness, developability, and immunogenicity. They hurt for affinity improver-ranking.
- **Don't trust random-split validation.** Supervised models validated on random splits of combinatorial libraries return optimistic upper bounds. Use distance-aware splits + budget-matched precision@K.

## Data

Source datasets (Absci HER2 SPR Kd, Mason 2021 FACS, KyDab) are **not redistributed**. Tables in `results/` contain computed columns only (sequences, mutation counts, model scores); measured values are stripped. Download from the original sources (`data/raw/README.md`) and restore:

```bash
python scripts/merge_fitness.py absci --raw /path/to/absci_her2_spr.csv
python scripts/merge_fitness.py kydab --raw /path/to/kydab.csv
python scripts/merge_fitness.py mason --raw /path/to/mason_cdrh3.csv
```

Model weights, PLM embeddings, and the 1N8Z crystal are downloaded by the code, not committed.

## Quickstart

```bash
conda env create -f environment.yml
conda env create -f environment_antifold.yml   # optional: AntiFold (isolated deps)

python scripts/merge_fitness.py absci --raw <your absci table>

python experiments/e1_antirank.py             # Q1: zero-shot enrichment
python experiments/e2f_distance_split.py      # Q3: leakage-free supervised
python experiments/e4b_fullcdr_eval.py        # Q2: structure partial-ρ
python figures/make_figures.py                # F1–F4
```

## Repository layout

```
models/         scorers: BLOSUM62, ESM-2, AbLang, AntiBERTy (masked-marginal)
evaluation/     metrics — improver_metrics.py, lowN_protocol.py, metrics.py, stratify.py
experiments/    e1* zero-shot · e2* supervised + leakage · e4* structure
structures/     AntiFold fixed-backbone scoring on 1N8Z
data/           download scripts + data/raw/README.md (access instructions)
scripts/        merge_fitness.py (restore source measured values)
results/        computed scores + derived-metric CSVs (no source measured values)
figures/        F1–F4 + make_figures.py
```

`CODEMAP.md` documents every script, its output, and its role (default path / robustness check / regenerate-from-scratch).

## Reproducibility

Every number comes from a committed script. Metrics recompute from scored CSVs — no model re-run needed. Structure tools are in a separate conda env (`environment_antifold.yml`) due to dependency conflicts. `validation_checklist.md` names the artifact behind each claim.

## Status

Q1–Q4 are complete and reproducible from committed scores. In progress:

- **Fine-tuning learning curve** — does task-specific supervision close the gap, and does it survive distance-aware splits? (extends Q3)
- **Benchmark expansion** — additional combinatorial antibody datasets to test protocol generality
- **Manuscript** — targeting a workshop methods contribution

## Limitations

Single target (trastuzumab / HER2). Strict improver tier is n = 5, validated against a robust relaxed tier with bootstrap CIs. Structure arm is fixed-backbone only. Zero-shot PLM scores model CDR-H3 variation only; HCDR1/2 variation (20% of variants, 0% of improvers) is unmodeled; conclusions re-verified on the exactly-scored subset. Full log in `validation_checklist.md`.

## License

Code and computed outputs: MIT. Third-party datasets and model weights are not covered and not redistributed.
