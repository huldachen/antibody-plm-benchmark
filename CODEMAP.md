# Code map

What every script and module is for, what it produces, and when you'd run it.
This benchmark accreted through a v2 → v4 arc, so not every file sits on the
default reproduction path: some are **robustness checks** (each answers one
skeptical question — see `validation_checklist.md`), some **regenerate inputs
from scratch**, and a few are **superseded but kept as the record**. Every file
below is either imported by the code, produces a committed `results/*.csv`, or is
documented here as intentionally off-path.

**Legend** — ▶ default path · ✓ robustness check · ⟲ regenerate-from-scratch
(needs model weights / GPU) · ⊘ superseded, kept as record · 📦 library (imported).

## Pipeline at a glance

```
download_*.py ─▶ (raw)         run_benchmark.py ⟲ ─▶ results/*_all_scored.csv  (model scores; committed)
                                                          │
scripts/merge_fitness.py ▶ restores measured values into the committed scored CSVs
                                                          │
              e1*/e2*/e4* ─────────────────────────────▶ results/*.csv (metrics)
              structures/score_antifold_* ⟲ ───────────▶ results/absci_antifold_*.csv
              figures/make_figures.py ─────────────────▶ figures/F1-F4.png
```

---

## Zero-shot enrichment — Q1 (`experiments/e1*`)

| Script | Role | Output |
|---|---|---|
| `run_benchmark.py` ⟲ | Score every model (BLOSUM/ESM-2/AbLang/AntiBERTy) over a dataset. Produces the scored tables everything else reads. Needs torch + weights; **not needed** if you use the committed scores. | `results/absci_all_scored.csv`, `results/mason_all_scored.csv` |
| `experiments/e1_antirank.py` ▶ | Improver enrichment / anti-ranking, strict + near tiers (the headline). Mann–Whitney with midrank ties, P@K, Recall@K, fold-improvement. | `absci_improver_metrics.csv`, `absci_near_improver_metrics.csv` |
| `experiments/e1b_mason_replication.py` ✓ | Second-assay check on Mason (binder retrieval) — the task-dependence result. Shows PLM enrichment **flips** when retrieving binders vs improvers. | `mason_binder_retrieval.csv` |
| `experiments/e1c_mechanism.py` ✓ | Is PLM enrichment just a low-mutation-count proxy? Correlates PLM AUROC with MutCount AUROC across 4 positive-set conditions. | `absci_mason_mechanism.csv` |
| `experiments/e1d_provenance_check.py` ✓ | Re-run E1 on the exactly-scored (HCDR3-only) subset — the ESM-2 partial-ρ demotion. Confirms CDR-H3–only results hold when HCDR1/2 variants are excluded. | `absci_e1d_provenance.csv` |
| `experiments/e1_tie_sensitivity.py` ✓ | Tie-consistent (midrank) vs stable-sort enrichment p-values. Diagnostic that moved strict-tier p from 0.028 → 0.075. | `absci_e1_tie_sensitivity.csv` |
| `experiments/e1_figure.py` ⊘ | Early E1 anti-ranking figure; superseded by F2 in `make_figures.py`. Kept as the analysis record. | `figures/figE1_improver_antirank.png` |

## Low-N supervision & leakage — Q3 (`experiments/e2*`)

| Script | Role | Output |
|---|---|---|
| `experiments/e2_learning_curve.py` ▶ | Few-shot learning curve (Ridge on one-hot/compact vs MutCount) with sliding-window splits. | `absci_e2_lowN.csv`, `absci_e2_lowN_summary.csv` |
| `experiments/e2b_fixed_test.py` ✓ | Fixed held-out test protocol (all 5 strict improvers always in eval). Clean generalization test. | `absci_e2b_fixed.csv`, `absci_e2b_fixed_summary.csv` |
| `experiments/e2c_group_c.py` ✓ | Additional scorer family (GP, parent-anchored Ridge, C1-Rank pairwise ranker). | `absci_e2c_groupc.csv`, `absci_e2c_groupc_summary.csv` |
| `experiments/e2d_embeddings.py` ✓ | Embedding-based supervised scorers (ESM-2 / AbLang features). Requires cached `.npy` from `extract_embeddings.py`. | `absci_e2d_emb.csv`, `absci_e2d_emb_summary.csv` |
| `experiments/e2e_leakage_diagnostic.py` ✓ | Quantify eval→train near-duplicate leakage in CDR-H3 Hamming distance (Fig F1). | `absci_e2e_leakage.csv` |
| `experiments/e2f_distance_split.py` ▶ | Leave-cluster-out distance-aware split — the keystone validity check. Guarantees no near-duplicate leakage. | `absci_e2f_distance_d{1,2}.csv`, `absci_e2f_distance_d{1,2}_summary.csv` |
| `experiments/e2g_distance_stratified.py` ✓ | Metric vs min-distance-to-train, seed-averaged — the correlation-vs-discovery reconciliation (Fig F4). Reports both Spearman ρ and near-AUROC per distance bin. | `absci_e2g_dist_stratified.csv` |
| `experiments/e2h_sensitivity.py` ✓ | Near-tier cutoff (5/10/20%) + censoring-floor sensitivity. | `absci_sensitivity.csv` |
| `experiments/e2i_fullcdr_recheck.py` ✓ | Full-CDR (vs CDR-H3-only) re-run of leakage/curve — the foundational correction that moved leakage from 39%→27%. | `absci_fullcdr_recheck.csv` |

## Structure arm — Q2 (`structures/`, `experiments/e4*`)

| Script | Role | Output |
|---|---|---|
| `structures/fetch_and_inspect_1n8z.py` ⟲ | Fetch + inspect the trastuzumab–HER2 crystal (PDB 1N8Z). Reports chain assignments and CDR-H3 loop residues. | `structures/pdb/` (gitignored) |
| `structures/score_antifold_crystal.py` ⟲ | AntiFold fixed-backbone scoring, HCDR3 window only (antifold env). | `absci_antifold_crystal.csv` |
| `structures/score_antifold_fullcdr.py` ⟲ | AntiFold over all heavy CDRs — HCDR1+2+3 (antifold env). | `absci_antifold_fullcdr.csv` |
| `experiments/e4_eval_crystal.py` ✓ | Enrichment metrics for the crystal scores. | `absci_e4_crystal_metrics.csv` |
| `experiments/e4_partial_correlation.py` ⊘ | Partial-ρ controlling mutation count, CDR-H3 basis. Superseded by e4b (full-CDR). | `absci_e4_partial_corr.csv` |
| `experiments/e4b_fullcdr_eval.py` ▶ | Full-CDR partial-ρ + tier-3 CI + leakage-inflation factor (the reported values). | `absci_e4b_fullcdr.csv` |

## Generalization (KyDab) & re-evaluation

| Script | Role | Output |
|---|---|---|
| `evaluation/kydab_eval.py` ▶ | KyDab per-antigen / pooled / null eval — out-of-distribution generalization check with bootstrap CI, BH-FDR, and Fisher-z pooling. `--out-suffix _strictpH72` writes the strict-pH sensitivity variants. | `kydab_eval_{pooled,per_antigen,null}[_strictpH72].csv`, `kydab_scores.csv` |
| `reevaluate.py` ✓ | Recompute the v2 metric suite on the committed scored CSVs (no model re-run). | `absci_v2_metrics.csv`, `mason_v2_metrics.csv`, `absci_improver_ranks.csv` |
| `experiments/extract_embeddings.py` ⟲ | Cache mean-pooled PLM embeddings (ESM-2 1280-d, AbLang 768-d) consumed by `e2d_embeddings.py`. Requires GPU. | `*.npy` (gitignored) |

## Data & figures

| Script | Role |
|---|---|
| `data/download_{absci,mason,kydab}.py` ⟲ | Process each source dataset into the local (gitignored) tables. |
| `scripts/merge_fitness.py` ▶ | Restore stripped measured values (Kd / labels) into the committed scored CSVs. |
| `figures/make_figures.py` ▶ | Render F1–F4 from the committed CSVs into `figures/`. |

## Libraries (imported, no standalone output) 📦

| Module | Used by | Purpose |
|---|---|---|
| `experiments/absci_fewshot_common.py` | e1/e2/e4 (11×) | Shared Absci loader, feature builders (one-hot/compact/delta), censored targets, scorer factories (Ridge, GP, C1-Rank, embedding). |
| `evaluation/improver_metrics.py` | 13× | Improver / near-improver masks, tie-corrected enrichment (midrank MW), P@K / Recall@K, fold-improvement. |
| `evaluation/lowN_protocol.py` | 9× | Stratified splits, fixed-test splits, learning-curve runners, summaries with CIs. |
| `evaluation/metrics.py` | 16× | v2 metric suite (Spearman, Kendall, AUROC, AUPRC, P@K, best F1). |
| `evaluation/stratify.py` ⊘ | — (none) | IMGT/ANARCI CDR-vs-framework position assignment. **Not on the current path** — the Mason/Absci datasets vary only within CDRs, so stratification is trivial; retained for extending the benchmark to framework-mutation datasets. |
| `models/blosum.py` | `run_benchmark.py` | BLOSUM62 parental-anchored substitution scorer (sum of BLOSUM62 scores at mutated positions vs WT). |
| `models/esm2.py` | `run_benchmark.py` | ESM-2 (650M) masked-marginal scorer — masks each mutated position in full-VH context, sums log-odds vs WT. |
| `models/ablang_scorer.py` | `run_benchmark.py` | AbLang heavy-chain masked-marginal scorer, same mask-and-sum approach. |
| `models/antiberty_scorer.py` | `run_benchmark.py` | AntiBERTy (OAS-trained) masked-marginal scorer, same mask-and-sum approach. |
| `evaluation/__init__.py` | — | Package marker (empty). |
| `models/__init__.py` | — | Package marker (empty). |

## Committed figures

| File | What it shows | Script |
|---|---|---|
| `F1_leakage.png` | Sequence-overlap leakage on the full-CDR basis (CDR-H3-only distance overstates: 39%→27% within Hamming ≤2, 14→0 exact duplicates). | `make_figures.py` |
| `F2_zeroshot_two_prior.png` | Two priors: parent-distance baselines enrich improvers; PLMs do not. Signed MW enrichment + near-AUROC. | `make_figures.py` |
| `F3_structure_counting.png` | Fixed-backbone IF reduces to counting: raw ρ → partial ρ controlling for mutation count (AntiFold +0.135 → ≈0). | `make_figures.py` |
| `F4_correlation_vs_discovery.png` | The advantage that generalizes (global ρ) is not the one that discovers (near-AUROC ≈ counting). | `make_figures.py` |

## Configuration & documentation

| File | Purpose |
|---|---|
| `environment.yml` | Conda env for the main analysis stack (Python 3.10, scipy, torch, ablang, antiberty, fair-esm). |
| `environment_antifold.yml` | Isolated conda env for the E4 structure arm (AntiFold/ESM-IF, numpy 1.26 + torch 2.3). Separate because pinned deps conflict with the analysis stack. |
| `.gitignore` | Excludes `data/raw/`, model weights, `.npy` embeddings, `__pycache__/`, OS files. |
| `LICENSE` | MIT (code and computed outputs). |
| `README.md` | Repo-level README with motivation, research questions, findings, evaluation protocol, quickstart. |
| `CODEMAP.md` | This file. |
| `validation_checklist.md` | Full robustness / self-audit log (A1–E3). Names every skeptical check, the question it fixed, the outcome, and the committed artifact. |
| `data/raw/README.md` | Access instructions for the three source datasets (Mason, Absci, KyDab) and what ships in `results/`. |
| `figures/README.md` | Figure-level captions mapping each PNG to the result it visualises. |

## Committed results (quick reference)

39 CSVs in `results/`. Each script's output is named in the tables above; the
full listing for grep:

**Scored tables (model outputs):**
`absci_all_scored.csv`, `mason_all_scored.csv`, `kydab_scores.csv`,
`absci_antifold_crystal.csv`, `absci_antifold_fullcdr.csv`

**Q1 zero-shot metrics:**
`absci_improver_metrics.csv`, `absci_near_improver_metrics.csv`,
`mason_binder_retrieval.csv`, `absci_mason_mechanism.csv`,
`absci_e1d_provenance.csv`, `absci_e1_tie_sensitivity.csv`

**Q3 supervised / leakage metrics:**
`absci_e2_lowN.csv`, `absci_e2_lowN_summary.csv`,
`absci_e2b_fixed.csv`, `absci_e2b_fixed_summary.csv`,
`absci_e2c_groupc.csv`, `absci_e2c_groupc_summary.csv`,
`absci_e2d_emb.csv`, `absci_e2d_emb_summary.csv`,
`absci_e2e_leakage.csv`,
`absci_e2f_distance_d1.csv`, `absci_e2f_distance_d1_summary.csv`,
`absci_e2f_distance_d2.csv`, `absci_e2f_distance_d2_summary.csv`,
`absci_e2g_dist_stratified.csv`, `absci_fullcdr_recheck.csv`,
`absci_sensitivity.csv`

**Q2 structure metrics:**
`absci_e4_crystal_metrics.csv`, `absci_e4_partial_corr.csv`,
`absci_e4b_fullcdr.csv`

**Generalization / legacy:**
`kydab_eval_pooled.csv`, `kydab_eval_per_antigen.csv`, `kydab_eval_null.csv`,
`kydab_eval_pooled_strictpH72.csv`, `kydab_eval_per_antigen_strictpH72.csv`,
`kydab_eval_null_strictpH72.csv`,
`absci_v2_metrics.csv`, `mason_v2_metrics.csv`, `absci_improver_ranks.csv`

*The single module with no current caller (`stratify.py`) is flagged here rather
than deleted: it is working infrastructure for a documented future case, not dead
code from a refactor.*
