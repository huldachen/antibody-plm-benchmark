# LoRA fine-tuning (Task A1)

Custom training loop for ESM-2 + LoRA on Mason / Absci CDR-H3 libraries. See the
v3 task queue for context.

> **Adapter weights are not retained.** `train_lora.py` writes `best_adapter/`
> and `final_adapter/` locally, but they are **not committed** (weights are
> gitignored, and the weights from the published run no longer exist on disk).
> What ships is the run's *provenance* — `config.json` (hyperparameters, seed,
> best epoch, test metrics) and `training_log.csv` (per-epoch curve) — plus the
> baseline comparison in `results/mason_finetune_baselines.csv`. So the reported
> numbers are **auditable but not re-scorable**: reproducing the fine-tuned row
> requires re-running the training (~1 h on one GPU/MPS at the committed config),
> whereas every non-LoRA baseline in that table re-computes in seconds from
> `leak_check.py`.
>
> Scope, stated plainly: **one** configuration (`mason_cls_r8_lr1e-4_e5`), Mason
> classification only. Not a hyperparameter sweep, not a label-budget learning
> curve, and not yet evaluated under the distance-aware protocol — the split here
> is random-stratified, so the fine-tuned number is an upper bound.

## Smoke test

Run before any real training to confirm the pipeline (model load, data, forward,
backward, loss decrease):

```bash
python training/train_lora.py --dataset mason --task classification --smoke-test
```

Expected: 5 steps on 32 samples, prints per-step loss, asserts `last < first`.

## macOS / Apple Silicon notes

If you hit a libomp duplicate-init error from torch + another lib both linking
OpenMP, prepend `KMP_DUPLICATE_LIB_OK=TRUE` to the command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python training/train_lora.py --dataset mason \
    --task classification --smoke-test --device mps
```

Device auto-detection picks `cuda > mps > cpu`. Override with `--device cpu`
if you hit MPS-unsupported ops (none in the smoke path on transformers 5.5,
peft 0.19; flag if you see new ones).

## Real training

```bash
# Mason (binary classification on AgClass)
python training/train_lora.py \
    --dataset mason --task classification \
    --lora-rank 8 --lr 1e-4 --epochs 10 \
    --output-dir training/checkpoints/mason_cls_r8

# Absci classification on Binder
python training/train_lora.py \
    --dataset absci --task classification \
    --lora-rank 8 --lr 1e-4 --epochs 10 \
    --output-dir training/checkpoints/absci_cls_r8

# Absci regression on neg_log_Kd (binders only, SameLength=1; ~420 rows)
python training/train_lora.py \
    --dataset absci --task regression \
    --lora-rank 8 --lr 1e-4 --epochs 20 \
    --output-dir training/checkpoints/absci_reg_r8
```

Output per run:
- `best_adapter/` — LoRA adapter at the best validation metric (ROC-AUC for
  classification, RMSE for regression).
- `final_adapter/` — adapter at the final epoch.
- `training_log.csv` — per-epoch train loss, val metrics, elapsed time.
- `config.json` — full args, best epoch, best val metric, test metrics, label
  summaries.

## Inputs

By design, the input to the tokenizer is the raw CDR-H3 sequence (10 res
for Mason, 12–13 res for Absci). This is technically out-of-distribution for
ESM-2 (pretrained on full proteins) but matches the literal A1 spec. See
`lab_notebook/decisions/DEC-003` for the rationale and the planned A1.5
ablation that embeds CDR-H3 in the full VH context.

## Splits

80 / 10 / 10 train / val / test, stratified:
- Classification — by integer label (binder/non-binder).
- Regression — by quartile bins of the continuous target so all splits cover
  the same dynamic range.

Seed default 42. Reproduce with `--seed 42`.

## Known compute notes

- ESM-2-650M LoRA fine-tune on a single T4/A100 GPU with batch size 8 is fine
  per the project spec.
- CPU smoke test runs but is slow — expect ~minutes for 5 steps. Don't try
  full training on CPU.
