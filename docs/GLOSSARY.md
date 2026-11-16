# Glossary

Terms as this benchmark uses them. Where a definition carries a number, the
number is the one in `results/`.

## The biology and the data

| term | meaning here |
|---|---|
| **CDR-H3** | Third complementarity-determining region of the antibody heavy chain — the loop that contributes most antigen contact. Mason varies 10 positions in it; Absci's is 13 residues. |
| **HCDR1 / HCDR2** | The other two heavy-chain CDR loops. Absci variants also vary in these, which is why a CDR-H3-only analysis is incomplete. |
| **full-CDR basis** | All 29 varying positions across HCDR1+HCDR2+HCDR3. The **corrected** basis; CDR-H3-only (13 positions) was the original and overstates near-duplication. |
| **parent / WT** | The unmutated starting antibody (trastuzumab for Absci). Every variant is described by its differences from it. |
| **combinatorial library** | A set of variants that are all mutants of *one* parent, same length. Mason and Absci are; **KyDab is not** — it is natural antibodies from four immunisations, with no parent and variable CDR-H3 length. |
| **affinity maturation** | Iteratively improving an antibody's binding strength. The task this benchmark models. |
| **germline** | The unmutated antibody gene an immune response starts from. Antibody PLMs are trained on natural repertoires and so encode a *germline-proximity* prior. |

## Measured quantities

| term | meaning here |
|---|---|
| **Kd** | Dissociation constant. **Lower = binds more tightly.** |
| **neg_log_Kd** | −log₁₀(Kd). **Higher = tighter binding = fitter.** Used so that "higher is better" holds everywhere. |
| **Binder / AgClass** | Binary label: does this variant bind at all. Absci `Binder`, Mason `AgClass`. |
| **LogEnrichment** | Mason's continuous readout — how much a variant was enriched in a FACS sort. |
| **fitness** | Generic term for whatever "good" means in a dataset: neg_log_Kd for Absci, LogEnrichment for Mason. |

## Positive sets — what counts as a "good" variant

This distinction is the benchmark's sharpest result: **changing only the
definition of a positive reverses the verdict.**

| term | meaning here |
|---|---|
| **improver** (strict tier) | A variant that **beats the parent**: Kd lower than the parent's. On Absci there are **5** of 1,266. The operationally interesting set — the ones actually worth making. |
| **near-improver** (near tier) | Top-decile binders by neg_log_Kd — **42** of 1,266. A proxy for the strict tier with enough members to support a claim. |
| **binder** | Any variant that binds at all — **420** of 1,266. The easiest positive set, and the one standard benchmarks use. |
| **tier** | Which positive set a metric was computed against. A number without its tier is meaningless: mutation count scores 0.63 on binders and 0.72 on near-improvers. |

## Models and scorers

| term | meaning here |
|---|---|
| **PLM** | Protein language model — trained on sequences, predicts amino acids in context. |
| **ESM-2 (650M)** | General-purpose PLM, all proteins. |
| **AbLang / AntiBERTy / AbLang-2** | Antibody-specific PLMs, trained on antibody repertoires. |
| **zero-shot** | Scoring with a pretrained model, **no labels used**. |
| **masked marginal** | Zero-shot scoring: mask each mutated position, sum the log-odds of the variant residue vs the parent's. |
| **PLL** | Pseudo-log-likelihood — a whole-sequence score summing per-position log-probabilities. Used for KyDab, where sequences differ in length. |
| **AntiFold / inverse folding** | Given a fixed backbone structure, score which sequences fit it. "Fixed-backbone" = the structure is held constant (PDB **1N8Z**, trastuzumab–HER2). |
| **BLOSUM62** | A substitution matrix from 1992. A deliberately trivial baseline. |
| **mutation count (MutCount)** | How many mutations a variant carries from the parent, negated so fewer = predicted fitter. **The baseline everything must beat**, and mostly doesn't. |
| **one-hot** | Each sequence position encoded as 20 binary features. 200 features for Mason, 260 for Absci. The baseline that ties the 650M fine-tune. |
| **LoRA** | Low-Rank Adaptation — fine-tuning that trains small adapter matrices (7.7 M parameters, 1.17%) instead of all 650 M. |
| **adapter** | The trained LoRA weights. A few MB, saved instead of the full model. |
| **ridge / RidgeCV** | Linear regression with L2 penalty; `CV` picks the penalty by cross-validation. |
| **MLP(128)** | A one-hidden-layer neural net on one-hot features — the non-linear baseline. |

## Statistics

| term | meaning here |
|---|---|
| **AUROC** | Probability a random positive scores above a random negative. 0.5 = chance; **below 0.5 = anti-prediction**, actively ranking good variants low. |
| **Spearman ρ** | Rank correlation. Measures global ordering, not whether the top of the list is right. |
| **precision@K / P@K** | Of your top K picks, what fraction are positives. **Budget-matched** = K set to what a team could actually synthesise. |
| **Recall@K** | Of all positives, what fraction appear in your top K. |
| **partial correlation** | Correlation between x and y after removing what both share with a control z. Here: does a score predict affinity *beyond* mutation count? |
| **control basis** | Which mutation count was controlled for — CDR-H3 or full-CDR. **Part of the value**: a partial ρ quoted without its basis cannot be reconciled. |
| **Mann–Whitney (MW)** | Rank test for whether positives score higher than negatives. Reported as an enrichment p-value. |
| **midrank / tie correction** | Averaging ranks across tied scores. Matters: one p-value moved 0.028 → 0.075 when applied. |
| **DeLong test** | Tests whether **two AUROCs differ** when both are computed on the same samples. Per-model confidence intervals cannot do this — overlapping CIs do not mean "no difference". |
| **paired** | Both models scored the *same* variants, so their errors are correlated and the test must account for it. |
| **stratified bootstrap CI** | Resampling positives and negatives separately to keep n_pos fixed — the dominant uncertainty when positives are rare (n = 5). |
| **n.s.** | Not significant at p < 0.05. |
| **BH-FDR / Bonferroni** | Corrections for testing many hypotheses at once. |

## Splits and leakage

| term | meaning here |
|---|---|
| **leakage** | Test data the model effectively saw in training. Here: near-duplicates, not exact copies. |
| **near-duplicate** | A test variant within a couple of mutations of a training variant. On the CDR-H3 basis **39.1%** of eval variants are within Hamming ≤ 2 of training; on full-CDR, **26.3%**. |
| **Hamming distance** | Number of positions at which two equal-length sequences differ. |
| **random split** | Shuffle and divide. On a combinatorial library this **inflates** results — the optimistic regime. |
| **distance-aware / leave-cluster-out split** | Cluster sequences within Hamming ≤ `d_link`, then assign **whole clusters** to train or test, so every test sequence is ≥ d_link+1 from every training one. |
| **d_link** | The clustering threshold. `d_link = 2` ⇒ test sequences are ≥ 3 mutations from training. |
| **stratified split** | Split preserving the positive/negative ratio. |
| **cross-fitting / out-of-fold (OOF)** | K folds; every variant scored by a model that never saw it. Used when one held-out split has too few improvers (Absci: 2 strict, 7 near) to test anything. |
| **prevalence / class balance** | Fraction of positives. The distance split shifts it (train 34.6%, test 16.4%), so absolute values are not comparable across split types. |

## Repository mechanics

| term | meaning here |
|---|---|
| **artifact** | A committed file containing a number. **Every published number must trace to one** — no number lives only in a printout. |
| **DEC-NNN** | A dated design-decision record. The ones cited in published code are glossed in `CODEMAP.md`. |
| **e1\* / e2\* / e4\* / e5\*** | Experiment IDs: e1 zero-shot, e2 few-shot + leakage, e4 structure, e5 fine-tuned improver retrieval. They tie a script to the result table it writes. |
| **provenance check** | Restricting an analysis to the variants a model actually scores correctly. This, not the change of control basis, is what demoted the ESM-2 partial correlation — see *What I corrected* in the README for the figures. A glossary defines terms; numbers live in one place. |
| **censored floor** | Non-binders have no measurable Kd, so they enter regression one log unit below the weakest binder rather than being dropped. |
