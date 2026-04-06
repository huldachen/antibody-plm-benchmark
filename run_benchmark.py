#!/usr/bin/env python3
"""
Main benchmark runner: score datasets with all models and evaluate.

Supports two datasets:
  - Mason CDR-H3 (binary classification, log-enrichment)
  - Absci HER2 SPR (continuous Kd, binder/non-binder)

Usage:
    # First download data (requires internet access):
    python data/download_mason.py
    python data/download_absci.py

    # Run BLOSUM62 only on Mason (no GPU needed):
    python run_benchmark.py --dataset mason --models blosum

    # Run BLOSUM62 on Absci:
    python run_benchmark.py --dataset absci --models blosum

    # Run all models on both datasets:
    python run_benchmark.py --dataset mason --models blosum esm2 ablang antiberty
    python run_benchmark.py --dataset absci --models blosum esm2 ablang antiberty

    # Quick test with subset:
    python run_benchmark.py --dataset mason --models blosum --max-variants 500
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.blosum import score_variants as blosum_score_variants
from evaluation.metrics import spearman_correlation, auc_roc

# -------------------------------------------------------------------
# Dataset configurations
# -------------------------------------------------------------------

DATASETS = {
    "mason": {
        "name": "Mason CDR-H3",
        "data_path": os.path.join(PROJECT_ROOT, "data", "processed", "mason_cdrh3_enrichment.csv"),
        "download_cmd": "python data/download_mason.py",
        "wt_seq": "WGGDGFYAMD",          # 10 residues, Mason boundary
        "seq_col": "AASeq",                # column with CDR-H3 sequences
        "fitness_col": "LogEnrichment",    # continuous fitness (for Spearman)
        "label_col": "AgClass",            # binary label (for AUC-ROC)
        "mutation_col": "NumMutations",
        "cdr_h3_len": 10,
    },
    "absci": {
        "name": "Absci HER2 SPR",
        "data_path": os.path.join(PROJECT_ROOT, "data", "processed", "absci_her2_spr.csv"),
        "download_cmd": "python data/download_absci.py",
        "wt_seq": "SRWGGDGFYAMDY",        # 13 residues, Absci boundary
        "seq_col": "HCDR3",                # column with CDR-H3 sequences
        "fitness_col": "neg_log_Kd",       # continuous fitness (for Spearman)
        "label_col": "Binder",             # binary label (for AUC-ROC)
        "mutation_col": "NumMutations",
        "cdr_h3_len": 13,
    },
}

AVAILABLE_MODELS = ["blosum", "esm2", "ablang", "antiberty"]

MODEL_NAMES = {
    "blosum": "BLOSUM62",
    "esm2": "ESM-2 (650M)",
    "ablang": "AbLang",
    "antiberty": "AntiBERTy",
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_dataset(dataset_key):
    """Load a processed dataset by key."""
    cfg = DATASETS[dataset_key]
    if not os.path.exists(cfg["data_path"]):
        print(f"ERROR: Processed data not found at {cfg['data_path']}")
        print(f"Run '{cfg['download_cmd']}' first.")
        sys.exit(1)

    df = pd.read_csv(cfg["data_path"])
    print(f"Loaded {len(df)} variants from {cfg['name']}")

    n_binders = (df[cfg["label_col"]] == 1).sum()
    n_nonbinders = (df[cfg["label_col"]] == 0).sum()
    print(f"  Binders: {n_binders}")
    print(f"  Non-binders: {n_nonbinders}")

    # For Absci: filter to same-length sequences for BLOSUM scoring
    if dataset_key == "absci":
        wt_len = len(cfg["wt_seq"])
        same_len = df[cfg["seq_col"]].str.len() == wt_len
        n_same = same_len.sum()
        n_diff = (~same_len).sum()
        print(f"  Same length as WT ({wt_len}): {n_same}")
        print(f"  Different length (excluded): {n_diff}")
        df = df[same_len].reset_index(drop=True)
        print(f"  Working with {len(df)} same-length variants")

    return df


# -------------------------------------------------------------------
# Scoring functions for each model
# -------------------------------------------------------------------

def score_blosum(df, cfg):
    """Score with parental-anchored BLOSUM62."""
    print(f"\n{'='*60}")
    print("Scoring with BLOSUM62...")
    t0 = time.time()
    df["BLOSUM62_score"] = blosum_score_variants(
        df, cfg["wt_seq"], seq_col=cfg["seq_col"]
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Score range: [{df['BLOSUM62_score'].min()}, {df['BLOSUM62_score'].max()}]")
    print(f"  Mean: {df['BLOSUM62_score'].mean():.2f}")
    return df, "BLOSUM62_score"


def score_esm2(df, cfg):
    """Score with ESM-2 masked marginal scoring."""
    print(f"\n{'='*60}")
    print("Scoring with ESM-2 (esm2_t33_650M_UR50D)...")
    print("  Loading model (this may take a minute)...")
    from models.esm2 import ESM2Scorer

    scorer = ESM2Scorer()
    print(f"  Model loaded on device: {scorer.device}")

    seq_col = cfg["seq_col"]
    wt_seq = cfg["wt_seq"]
    t0 = time.time()
    n = len(df)
    scores = []
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            score = scorer.score_variant(row[seq_col], wt_cdr_h3=wt_seq)
            scores.append(score)
        except Exception as e:
            scores.append(np.nan)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} seq/s, ETA {eta:.0f}s")

    df["ESM2_score"] = scores
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} seq/s)")
    valid = df["ESM2_score"].notna().sum()
    print(f"  Valid scores: {valid}/{len(df)}")
    print(f"  Score range: [{df['ESM2_score'].min():.2f}, {df['ESM2_score'].max():.2f}]")
    return df, "ESM2_score"


def score_ablang(df, cfg):
    """Score with AbLang masked marginal scoring."""
    print(f"\n{'='*60}")
    print("Scoring with AbLang (heavy chain)...")
    print("  Loading model...")
    from models.ablang_scorer import AbLangScorer

    scorer = AbLangScorer(chain="heavy")
    print(f"  Model loaded on device: {scorer.device}")

    seq_col = cfg["seq_col"]
    wt_seq = cfg["wt_seq"]
    t0 = time.time()
    n = len(df)
    scores = []
    first_error_shown = False
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            score = scorer.score_variant(row[seq_col], wt_cdr_h3=wt_seq)
            scores.append(score)
        except Exception as e:
            scores.append(np.nan)
            if not first_error_shown:
                import traceback
                print(f"  ERROR on seq {i}: {e}")
                traceback.print_exc()
                first_error_shown = True
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} seq/s, ETA {eta:.0f}s")

    df["AbLang_score"] = scores
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} seq/s)")
    valid = df["AbLang_score"].notna().sum()
    print(f"  Valid scores: {valid}/{len(df)}")
    return df, "AbLang_score"


def score_antiberty(df, cfg):
    """Score with AntiBERTy masked marginal scoring."""
    print(f"\n{'='*60}")
    print("Scoring with AntiBERTy...")
    print("  Loading model...")
    from models.antiberty_scorer import AntiBERTyScorer

    scorer = AntiBERTyScorer()
    print(f"  Model loaded on device: {scorer.device}")

    seq_col = cfg["seq_col"]
    wt_seq = cfg["wt_seq"]
    t0 = time.time()
    n = len(df)
    scores = []
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            score = scorer.score_variant(row[seq_col], wt_cdr_h3=wt_seq)
            scores.append(score)
        except Exception as e:
            scores.append(np.nan)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} seq/s, ETA {eta:.0f}s")

    df["AntiBERTy_score"] = scores
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} seq/s)")
    valid = df["AntiBERTy_score"].notna().sum()
    print(f"  Valid scores: {valid}/{len(df)}")
    return df, "AntiBERTy_score"


SCORER_MAP = {
    "blosum": score_blosum,
    "esm2": score_esm2,
    "ablang": score_ablang,
    "antiberty": score_antiberty,
}


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate_all_models(df, score_cols, cfg):
    """
    Evaluate all scored models and produce a comparison table.

    Args:
        df: DataFrame with all model scores.
        score_cols: Dict mapping model_key -> score_column_name.
        cfg: Dataset configuration dict.

    Returns:
        Summary DataFrame with per-model metrics.
    """
    fitness_col = cfg["fitness_col"]
    label_col = cfg["label_col"]
    mutation_col = cfg["mutation_col"]

    print("\n" + "=" * 60)
    print(f"COMPARATIVE EVALUATION — All Models on {cfg['name']}")
    print("=" * 60)

    # Filter to rows with finite fitness values for Spearman
    has_fitness = df[fitness_col].notna() & np.isfinite(df[fitness_col])
    df_f = df[has_fitness].copy()

    results = []
    for model_key, col in score_cols.items():
        name = MODEL_NAMES.get(model_key, model_key)
        valid = df[col].notna()
        valid_f = df_f[col].notna()

        # Spearman on continuous fitness
        if valid_f.sum() > 10:
            sp = spearman_correlation(
                df_f.loc[valid_f, col], df_f.loc[valid_f, fitness_col]
            )
        else:
            sp = {"rho": np.nan, "pvalue": np.nan, "n": 0}

        # AUC on binary label
        auc = auc_roc(df.loc[valid, col], df.loc[valid, label_col])

        results.append({
            "Model": name,
            "Spearman_rho": sp["rho"],
            "Spearman_p": sp["pvalue"],
            "AUC_ROC": auc,
            "N_spearman": sp["n"],
            "N_total": valid.sum(),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Spearman_rho", ascending=False)

    # Print comparison table
    print(f"\n{'Model':<20s}  {'Spearman rho':>12s}  {'AUC-ROC':>8s}  {'N(Spear)':>8s}  {'N(total)':>8s}")
    print(f"{'-'*20}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20s}  {row['Spearman_rho']:>12.4f}  "
              f"{row['AUC_ROC']:>8.4f}  {row['N_spearman']:>8.0f}  {row['N_total']:>8.0f}")

    # Stratified by mutation count
    if mutation_col in df_f.columns:
        print(f"\n{'='*60}")
        print("STRATIFIED BY MUTATION COUNT")
        print(f"{'='*60}")

        mut_counts = sorted(df_f[mutation_col].unique())
        # Header
        header = f"{'#Mut':>5s}"
        for model_key in score_cols:
            name = MODEL_NAMES.get(model_key, model_key)[:8]
            header += f"  {name:>10s}"
        header += f"  {'N':>6s}"
        print(header)
        print("-" * len(header))

        for n_mut in mut_counts:
            if n_mut < 0:
                continue  # skip different-length flag
            subset = df_f[df_f[mutation_col] == n_mut]
            if len(subset) < 20:
                continue
            line = f"{n_mut:>5d}"
            for model_key, col in score_cols.items():
                sp = spearman_correlation(subset[col], subset[fitness_col])
                line += f"  {sp['rho']:>10.4f}"
            line += f"  {len(subset):>6d}"
            print(line)

    # Score distribution by binding class for each model
    print(f"\n{'='*60}")
    print("SCORE DISTRIBUTION BY BINDING CLASS")
    print(f"{'='*60}")
    for model_key, col in score_cols.items():
        name = MODEL_NAMES.get(model_key, model_key)
        binders = df.loc[(df[label_col] == 1) & df[col].notna(), col]
        non_binders = df.loc[(df[label_col] == 0) & df[col].notna(), col]
        if len(binders) > 0 and len(non_binders) > 0:
            sep = binders.mean() - non_binders.mean()
            print(f"\n  {name}:")
            print(f"    Binder:     mean={binders.mean():+.2f}  std={binders.std():.2f}")
            print(f"    Non-binder: mean={non_binders.mean():+.2f}  std={non_binders.std():.2f}")
            print(f"    Separation: {sep:+.2f}")

    return results_df


def save_results(df, results_df, score_cols, dataset_key, cfg):
    """Save all results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save full scored dataset
    scored_path = os.path.join(RESULTS_DIR, f"{dataset_key}_all_scored.csv")
    df.to_csv(scored_path, index=False)
    print(f"\nScored dataset saved: {scored_path}")

    # Save/update benchmark summary
    summary_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    dataset_label = cfg["name"]
    rows = []
    for model_key, col in score_cols.items():
        name = MODEL_NAMES.get(model_key, model_key)
        row = results_df[results_df["Model"] == name]
        if len(row) > 0:
            rows.append({
                "model": name,
                "dataset": dataset_label,
                "wt_sequence": cfg["wt_seq"],
                "n_variants": len(df),
                "spearman_rho": row["Spearman_rho"].values[0],
                "spearman_pvalue": row["Spearman_p"].values[0],
                "auc_roc": row["AUC_ROC"].values[0],
            })

    new_df = pd.DataFrame(rows)

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        # Remove models we're updating for this dataset
        model_names = [MODEL_NAMES.get(k, k) for k in score_cols]
        existing = existing[~(
            (existing["model"].isin(model_names)) &
            (existing["dataset"] == dataset_label)
        )]
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df.to_csv(summary_path, index=False)
    print(f"Benchmark results saved: {summary_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Antibody PLM Benchmark"
    )
    parser.add_argument(
        "--dataset", default="mason",
        choices=list(DATASETS.keys()),
        help="Dataset to benchmark on (default: mason)"
    )
    parser.add_argument(
        "--models", nargs="+", default=["blosum"],
        choices=AVAILABLE_MODELS,
        help="Models to run (default: blosum only)"
    )
    parser.add_argument(
        "--max-variants", type=int, default=None,
        help="Limit to first N variants (useful for quick test runs)"
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    print("=" * 60)
    print("Antibody PLM Benchmark")
    print(f"Dataset: {cfg['name']}")
    print(f"WT: {cfg['wt_seq']} ({len(cfg['wt_seq'])} residues)")
    print(f"Models: {', '.join(MODEL_NAMES.get(m, m) for m in args.models)}")
    if args.max_variants:
        print(f"Max variants: {args.max_variants}")
    print("=" * 60)

    # Load data
    df = load_dataset(args.dataset)

    if args.max_variants and args.max_variants < len(df):
        # Stratified sample: keep binder/non-binder ratio
        label_col = cfg["label_col"]
        n = args.max_variants
        binder_frac = (df[label_col] == 1).mean()
        n_bind = int(n * binder_frac)
        n_nonbind = n - n_bind
        binders = df[df[label_col] == 1].sample(
            n=min(n_bind, (df[label_col]==1).sum()), random_state=42
        )
        nonbinders = df[df[label_col] == 0].sample(
            n=min(n_nonbind, (df[label_col]==0).sum()), random_state=42
        )
        df = pd.concat([binders, nonbinders]).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} variants (stratified)")

    # Score with each model
    score_cols = {}
    for model_key in args.models:
        scorer_fn = SCORER_MAP[model_key]
        df, col_name = scorer_fn(df, cfg)
        score_cols[model_key] = col_name

    # Evaluate
    results_df = evaluate_all_models(df, score_cols, cfg)

    # Save
    save_results(df, results_df, score_cols, args.dataset, cfg)

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    if len(args.models) == 1 and args.models[0] == "blosum":
        print(f"To add PLM models: python run_benchmark.py --dataset {args.dataset} --models blosum esm2 ablang antiberty")
    print("=" * 60)


if __name__ == "__main__":
    main()
