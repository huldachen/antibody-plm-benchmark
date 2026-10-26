# results/

Derived artifacts, grouped by dataset. Every file is produced by a committed
script — see `CODEMAP.md` for the script behind each, and the validation
checklist for the question each was computed to answer.

```
results/absci/   trastuzumab HER2 SPR, n = 1266   (29 files)
results/mason/   CDR-H3 FACS, n = 500 scored      (4 files)
results/kydab/   multi-target panel               (7 files)
```

Scripts never hardcode these paths: `evaluation/paths.py` resolves
`result_path("absci_e2e_leakage.csv")` to its grouped location, so the layout is
defined in one place and can change without touching the ~28 call sites.

## Naming

`<dataset>_<stage>_<what>.csv` — **stage** is the experiment that produced it
(`e1*` zero-shot, `e2*` low-N supervised and leakage, `e4*` structure; absent for
scored tables and re-evaluations). `*_summary.csv` is aggregated across
seeds/budgets; the unsuffixed file is the per-run long form.

## The files you probably want

| file | what it answers |
|---|---|
| `absci/absci_all_scored.csv`, `mason/mason_all_scored.csv` | the scored variant tables everything else reads |
| `absci/absci_mason_mechanism.csv` | **the headline** — AUROC per positive-set definition, bootstrap CIs, paired DeLong p (F5) |
| `absci/absci_near_improver_metrics.csv` | zero-shot enrichment on the robust n=42 tier (F2) |
| `absci/absci_improver_metrics.csv` | the rare n=5 strict tier — reported with intervals, too wide to carry a claim |
| `absci/absci_e4b_fullcdr.csv` | structure partial-ρ on the corrected full-CDR basis (F3) |
| `absci/absci_e2e_leakage.csv` | near-duplicate leakage between train and eval (F1) |
| `absci/absci_e2f_distance_d{1,2}*.csv` | leakage-free distance-aware split |
| `absci/absci_e2g_dist_stratified.csv` | correlation vs discovery by distance bin (F4) |
| `mason/mason_finetune_baselines.csv` | LoRA vs one-hot baselines, split diagnostics on every row |

`*_v2_metrics.csv` and `absci_improver_ranks.csv` carry the global-AUC framing of
v1–v3, superseded by improver retrieval in v4. Kept as the record, not as current
results.
