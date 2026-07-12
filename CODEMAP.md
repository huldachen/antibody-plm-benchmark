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

## Zero-shot enrichment — Q1 (`experiments/e1*`)

| Script | Role | Output |
|---|---|---|
| `run_benchmark.py` ⟲ | Score every model (BLOSUM/ESM-2/AbLang/AntiBERTy) over a dataset. Produces the scored tables everything else reads. Needs torch + weights; **not needed** if you use the committed scores. | `results/absci_all_scored.csv`, `results/mason_all_scored.csv` |
| `experiments/e1_antirank.py` ▶ | Improver enrichment / anti-ranking, strict + near tiers (the headline). | `absci_improver_metrics.csv`, `absci_near_improver_metrics.csv` |
| `experiments/e1b_mason_replication.py` ✓ | Second-assay check on Mason (binder retrieval) — the task-dependence result. | `mason_binder_retrieval.csv` |
| `experiments/e1c_mechanism.py` ✓ | Is PLM enrichment just a low-mutation-count proxy? (four positive-set conditions) | `absci_mason_mechanism.csv` |
| `experiments/e1d_provenance_check.py` ✓ | Re-run E1 on the exactly-scored (HCDR3-only) subset — the ESM-2 partial-ρ demotion. | `absci_e1d_provenance.csv` |
| `experiments/e1_tie_sensitivity.py` ✓ | Tie-consistent (midrank) vs stable-sort enrichment p-values. | `absci_e1_tie_sensitivity.csv` |
| `experiments/e1_figure.py` ⊘ | Early E1 anti-ranking figure; superseded by F2 in `make_figures.py`. Kept as the analysis record. | `figures/figE1_improver_antirank.png` |

## Low-N supervision & leakage — Q3 (`experiments/e2*`)

| Script | Role | Output |
|---|---|---|
| `experiments/e2_learning_curve.py` ▶ | Few-shot learning curve; mutation count as the explicit baseline. | `absci_e2_lowN(_summary).csv` |
| `experiments/e2b_fixed_test.py` ✓ | Fixed held-out test protocol (improvers held out). | `absci_e2b_fixed(_summary).csv` |
| `experiments/e2c_group_c.py` ✓ | Additional scorer family (GP / rank objectives). | `absci_e2c_groupc(_summary).csv` |
| `experiments/e2d_embeddings.py` ✓ | Embedding-based supervised scorers (ESM-2 / AbLang features). | `absci_e2d_emb(_summary).csv` |
| `experiments/e2e_leakage_diagnostic.py` ✓ | Quantify eval→train near-duplicate leakage (Fig F1). | `absci_e2e_leakage.csv` |
| `experiments/e2f_distance_split.py` ▶ | Leave-cluster-out distance-aware split (the #1 validity check). | `absci_e2f_distance_d{1,2}(_summary).csv` |
| `experiments/e2g_distance_stratified.py` ✓ | Metric vs min-distance-to-train, seed-averaged (Fig F4). | `absci_e2g_dist_stratified.csv` |
| `experiments/e2h_sensitivity.py` ✓ | Near-tier cutoff (5/10/20%) + censoring-floor sensitivity. | `absci_sensitivity.csv` |
| `experiments/e2i_fullcdr_recheck.py` ✓ | Full-CDR (vs CDR-H3-only) re-run of leakage/curve — the foundational correction. | `absci_fullcdr_recheck.csv` |

## Structure arm — Q2 (`structures/`, `experiments/e4*`)

| Script | Role | Output |
|---|---|---|
| `structures/fetch_and_inspect_1n8z.py` ⟲ | Fetch + inspect the trastuzumab–HER2 crystal. | `structures/pdb/` (gitignored) |
| `structures/score_antifold_crystal.py` ⟲ | AntiFold fixed-backbone scoring, HCDR3 (antifold env). | `absci_antifold_crystal.csv` |
| `structures/score_antifold_fullcdr.py` ⟲ | AntiFold over all heavy CDRs (antifold env). | `absci_antifold_fullcdr.csv` |
| `experiments/e4_eval_crystal.py` ✓ | Enrichment metrics for the crystal scores. | `absci_e4_crystal_metrics.csv` |
| `experiments/e4_partial_correlation.py` ⊘ | Partial-ρ controlling mutation count, CDR-H3 basis. Superseded by e4b. | `absci_e4_partial_corr.csv` |
| `experiments/e4b_fullcdr_eval.py` ▶ | Full-CDR partial-ρ + tier-3 CI + leakage-inflation factor (the reported values). | `absci_e4b_fullcdr.csv` |

## Generalization (KyDab) & re-evaluation

| Script | Role | Output |
|---|---|---|
| `evaluation/kydab_eval.py` ▶ | KyDab per-antigen / pooled / null eval. `--out-suffix _strictpH72` writes the strict-pH sensitivity variants. | `kydab_eval_{pooled,per_antigen,null}[_strictpH72].csv`, `kydab_scores.csv` |
| `reevaluate.py` ✓ | Recompute the v2 metric suite on the committed scored CSVs (no model re-run). | `absci_v2_metrics.csv`, `mason_v2_metrics.csv`, `absci_improver_ranks.csv` |
| `experiments/extract_embeddings.py` ⟲ | Cache PLM embeddings consumed by `e2d_embeddings.py`. | `*.npy` (gitignored) |

## Data & figures

| Script | Role |
|---|---|
| `data/download_{absci,mason,kydab}.py` ⟲ | Process each source dataset into the local (gitignored) tables. |
| `scripts/merge_fitness.py` ▶ | Restore stripped measured values (Kd / labels) into the committed scored CSVs. |
| `figures/make_figures.py` ▶ | Render F1–F4 from the committed CSVs into `figures/`. |

## Libraries (imported, no standalone output) 📦

| Module | Used by | Purpose |
|---|---|---|
| `experiments/absci_fewshot_common.py` | e1/e2/e4 (11×) | Shared Absci loader, feature builders, scorer factories. |
| `evaluation/improver_metrics.py` | 13× | Improver / near-improver masks, tie-corrected enrichment, P@K / Recall@K. |
| `evaluation/lowN_protocol.py` | 9× | Stratified splits, learning-curve runners, summaries. |
| `evaluation/metrics.py` | 16× | v2 metric suite (Spearman, AUROC, AUPRC, P@K). |
| `models/{blosum,esm2,ablang_scorer,antiberty_scorer}.py` | `run_benchmark.py` | Per-model scorer classes. |
| `evaluation/stratify.py` ⊘ | — (none) | IMGT/ANARCI CDR-vs-framework position assignment. **Not on the current path** — the Mason/Absci datasets vary only within CDRs, so stratification is trivial; retained for extending the benchmark to framework-mutation datasets. |

*The single module with no current caller (`stratify.py`) is flagged here rather
than deleted: it is working infrastructure for a documented future case, not dead
code from a refactor.*
