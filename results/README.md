# results/

Derived artifacts. Every file here is produced by a committed script — see
[`../docs/CODEMAP.md`](../docs/CODEMAP.md) for the script behind each one, and
[`../docs/validation_checklist.md`](../docs/validation_checklist.md) for the
question each was computed to answer.

These are **computed columns only**. The source datasets' measured values (Absci
/ KyDab Kd, Mason binder labels and enrichment) are stripped and not
redistributed; restore them with `scripts/merge_fitness.py` (see the main README).

## Naming

```
<dataset>_<stage>_<what>.csv
```

- **dataset** — `absci` (trastuzumab HER2 SPR, n=1266) · `mason` (CDR-H3 FACS,
  n=500 scored / 36391 full) · `kydab` (multi-target panel)
- **stage** — the experiment that produced it: `e1*` zero-shot · `e2*` low-N
  supervised and leakage · `e4*` structure · absent for scored tables and
  re-evaluations
- `*_summary.csv` — aggregated across seeds/budgets; the unsuffixed file is the
  per-run long form

## The files you probably want

| file | what it answers |
|---|---|
| `absci_all_scored.csv`, `mason_all_scored.csv` | the scored variant tables everything else reads |
| `absci_mason_mechanism.csv` | **the headline** — AUROC per positive-set definition, with bootstrap CIs and paired DeLong p (F5) |
| `absci_near_improver_metrics.csv` | zero-shot enrichment on the robust n=42 tier (F2) |
| `absci_improver_metrics.csv` | same on the rare n=5 strict tier — reported with intervals, too wide to carry a claim |
| `absci_e4b_fullcdr.csv` | structure partial-ρ on the corrected full-CDR basis (F3) |
| `absci_e2e_leakage.csv` | near-duplicate leakage between train and eval (F1) |
| `absci_e2f_distance_d{1,2}*.csv` | leakage-free distance-aware split |
| `absci_e2g_dist_stratified.csv` | correlation vs discovery by distance bin (F4) |
| `mason_finetune_baselines.csv` | LoRA vs one-hot baselines, with the split diagnostics on every row |

`*_v2_metrics.csv` and `absci_improver_ranks.csv` carry the global-AUC framing of
v1–v3, superseded by improver retrieval in v4. They are kept as the record, not
as current results.
