#!/usr/bin/env python3
"""
Reevaluate already-scored datasets with the v2 metric suite.

Loads results/<dataset>_all_scored.csv (produced by run_benchmark.py) and
recomputes metrics without re-running any PLMs. This is the right tool when
we add a new metric and want to refresh the reported numbers.

v2 metrics (added in response to community feedback on v1):
  - AUPRC + best F1        : class-imbalance-aware classification metrics.
  - Kendall tau            : tie-aware rank correlation (complements Spearman).
  - Precision @ K          : lab-workflow metric — of the top-K picks,
                             how many are binders? Reported for K=30, 100,
                             and top-5%.
  - MutCount baseline      : treated as a first-class "model" to expose
                             how trivial distance-from-parent really is.

Usage
-----
    python reevaluate.py --dataset mason
    python reevaluate.py --dataset absci
    python reevaluate.py --dataset absci --beat-parent
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import (
    spearman_correlation, kendall_tau,
    auc_roc, auprc, best_f1,
    precision_at_k, precision_at_k_beat_reference,
)


DATASETS = {
    "mason": {
        "name": "Mason CDR-H3",
        "scored_path": os.path.join(PROJECT_ROOT, "results", "mason_all_scored.csv"),
        "seq_col": "AASeq",
        "fitness_col": "LogEnrichment",
        "label_col": "AgClass",
        "mutation_col": "NumMutations",
        "wt_fitness": None,  # no WT row in the 500-variant subsample
    },
    "absci": {
        "name": "Absci HER2 SPR",
        "scored_path": os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv"),
        "seq_col": "HCDR3",
        "fitness_col": "neg_log_Kd",
        "label_col": "Binder",
        "mutation_col": "NumMutations",
        # Parental trastuzumab is the NumMutations=0 row in the scored CSV
        # (Kd = 1.94 nM, neg_log_Kd ≈ 8.7122). We look it up at runtime rather
        # than hard-coding a rounded value — rounding caused WT itself to fall
        # above the threshold, falsely inflating MutCount's beat-parent P@30.
        "wt_fitness": "lookup:NumMutations==0",
    },
}

# Columns in the scored CSV that correspond to model scores, in display order.
MODEL_COLUMNS = [
    ("MutCount",  "MutCount_score"),    # synthesized from NumMutations
    ("BLOSUM62",  "BLOSUM62_score"),
    ("ESM-2",     "ESM2_score"),
    ("AbLang",    "AbLang_score"),
    ("AntiBERTy", "AntiBERTy_score"),
]


def add_mutcount_column(df, mutation_col):
    """MutCount scorer: fewer mutations = higher score (closer to parent)."""
    df = df.copy()
    df["MutCount_score"] = -df[mutation_col].astype(float)
    return df


def resolve_wt_fitness(df, cfg):
    """
    Resolve the parental-WT fitness value.

    Accepts either a numeric literal (legacy) or the string
    `"lookup:<column>==<value>"` which looks up the row matching that
    predicate in the loaded DataFrame and returns its `fitness_col` value.
    Looking up at runtime avoids the floating-point rounding bug we hit with
    a hard-coded constant (WT's true fitness was slightly above the rounded
    threshold, so WT itself counted as an improver against itself).
    """
    raw = cfg.get("wt_fitness")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.startswith("lookup:"):
        expr = raw[len("lookup:"):]
        col, val = expr.split("==")
        matches = df[df[col].astype(str) == val.strip()]
        if len(matches) == 0:
            raise ValueError(f"WT row not found: {expr}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous WT lookup: {len(matches)} rows match {expr}")
        return float(matches[cfg["fitness_col"]].iloc[0])
    raise ValueError(f"Unrecognized wt_fitness spec: {raw!r}")


def improver_rank_diagnostic(df, cfg, wt_fitness):
    """
    Per-model improver-rank summary — complements the noisy P@K metric when
    the number of improvers is small (n=5 on Absci).

    For each model, returns the ranks (1 = best) of the variants whose fitness
    strictly exceeds the WT, plus a two-sided Mann–Whitney U test comparing
    improver ranks to non-improver ranks. A significant result with improvers
    ranked *higher-numbered* (lower-scored) than non-improvers means the model
    is anti-predictive for improvement.
    """
    from scipy.stats import mannwhitneyu

    fit = df[cfg["fitness_col"]]
    is_improver = (fit > wt_fitness).values
    n_improvers = int(is_improver.sum())
    n_total = len(df)

    rows = []
    for display_name, col in MODEL_COLUMNS:
        if col not in df.columns:
            continue
        scores = df[col].values
        # Descending rank: rank 1 = best score.
        order = np.argsort(-scores, kind="mergesort")
        ranks = np.empty(n_total, dtype=int)
        ranks[order] = np.arange(1, n_total + 1)

        imp_ranks = np.sort(ranks[is_improver])
        non_ranks = ranks[~is_improver]

        if n_improvers >= 2 and len(non_ranks) >= 2:
            # Alternative 'greater' = improvers have HIGHER ranks (= worse scores);
            # a small p-value here means the model anti-predicts improvement.
            u, p_anti = mannwhitneyu(imp_ranks, non_ranks, alternative="greater")
            _, p_enrich = mannwhitneyu(imp_ranks, non_ranks, alternative="less")
        else:
            p_anti = np.nan
            p_enrich = np.nan

        rows.append({
            "Model": display_name,
            "n_improvers": n_improvers,
            "improver_ranks": imp_ranks.tolist(),
            "median_rank": int(np.median(imp_ranks)) if n_improvers else -1,
            "in_top_30": int((imp_ranks <= 30).sum()),
            "in_top_100": int((imp_ranks <= 100).sum()),
            "in_bottom_half": int((imp_ranks > n_total // 2).sum()),
            "MW_p_anti": p_anti,
            "MW_p_enrich": p_enrich,
        })
    return pd.DataFrame(rows), n_improvers, n_total


def evaluate_all(df, cfg, beat_parent=False):
    fitness_col = cfg["fitness_col"]
    label_col = cfg["label_col"]

    # Resolve WT fitness once (via runtime lookup) so every model uses the
    # exact same reference threshold.
    wt_fit = resolve_wt_fitness(df, cfg) if beat_parent else None

    rows = []
    for display_name, col in MODEL_COLUMNS:
        if col not in df.columns:
            continue

        pred = df[col]
        valid = pred.notna()

        sp = spearman_correlation(pred[valid], df.loc[valid, fitness_col])
        kt = kendall_tau(pred[valid], df.loc[valid, fitness_col])
        auc = auc_roc(pred[valid], df.loc[valid, label_col])
        ap = auprc(pred[valid], df.loc[valid, label_col])
        f1i = best_f1(pred[valid], df.loc[valid, label_col])
        pk30 = precision_at_k(pred[valid], df.loc[valid, label_col], k=30)
        pk100 = precision_at_k(pred[valid], df.loc[valid, label_col], k=100)
        pk5 = precision_at_k(pred[valid], df.loc[valid, label_col], k=0.05)

        row = {
            "Model": display_name,
            "Spearman_rho": sp["rho"],
            "Kendall_tau": kt["tau"],
            "AUC_ROC": auc,
            "AUPRC": ap,
            "Best_F1": f1i["f1"],
            "P@30": pk30["mean"],
            "P@100": pk100["mean"],
            "P@top5%": pk5["mean"],
            "P@30_ties": pk30["n_ties"],
            "N": int(valid.sum()),
        }

        if beat_parent and wt_fit is not None:
            pkbp30 = precision_at_k_beat_reference(
                pred[valid], df.loc[valid, fitness_col],
                reference_fitness=wt_fit, k=30,
            )
            pkbp100 = precision_at_k_beat_reference(
                pred[valid], df.loc[valid, fitness_col],
                reference_fitness=wt_fit, k=100,
            )
            row["P@30_beatWT"] = pkbp30["mean"]
            row["P@100_beatWT"] = pkbp100["mean"]

        rows.append(row)

    return pd.DataFrame(rows)


def print_table(results_df, cfg, beat_parent=False):
    print("=" * 78)
    print(f"v2 METRICS — {cfg['name']}")
    print("=" * 78)

    core_cols = ["Model", "Spearman_rho", "Kendall_tau",
                 "AUC_ROC", "AUPRC", "Best_F1",
                 "P@30", "P@100", "P@top5%", "P@30_ties", "N"]

    # Pretty print core table
    print(results_df[core_cols].to_string(
        index=False,
        formatters={
            "Spearman_rho": "{:+.3f}".format,
            "Kendall_tau":  "{:+.3f}".format,
            "AUC_ROC":      "{:.3f}".format,
            "AUPRC":        "{:.3f}".format,
            "Best_F1":      "{:.3f}".format,
            "P@30":         "{:.3f}".format,
            "P@100":        "{:.3f}".format,
            "P@top5%":      "{:.3f}".format,
        },
    ))

    if beat_parent and "P@30_beatWT" in results_df.columns:
        print()
        print("Beat-parent Precision@K (positive = fitness > WT):")
        print(results_df[["Model", "P@30_beatWT", "P@100_beatWT"]].to_string(
            index=False,
            formatters={
                "P@30_beatWT":  "{:.3f}".format,
                "P@100_beatWT": "{:.3f}".format,
            },
        ))
    print()


def print_improver_diagnostic(rank_df, n_improvers, n_total):
    """Companion to the beat-parent P@K table for small-n-improver regimes."""
    print("Improver-rank diagnostic  "
          f"(n_improvers = {n_improvers} of {n_total}; "
          f"expected median under null = {n_total // 2})")
    print("  MW_p_anti:   one-sided Mann-Whitney p-value for improvers ranked"
          " WORSE than non-improvers (model anti-predicts improvement).")
    print("  MW_p_enrich: one-sided p-value for improvers ranked BETTER than"
          " non-improvers (model enriches for improvers).")
    cols = ["Model", "improver_ranks", "median_rank",
            "in_top_30", "in_top_100", "in_bottom_half",
            "MW_p_anti", "MW_p_enrich"]
    print(rank_df[cols].to_string(
        index=False,
        formatters={
            "MW_p_anti":   lambda x: "nan" if pd.isna(x) else f"{x:.3f}",
            "MW_p_enrich": lambda x: "nan" if pd.isna(x) else f"{x:.3f}",
        },
    ))
    print()


def main():
    ap = argparse.ArgumentParser(description="Reevaluate scored datasets with v2 metrics.")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    ap.add_argument("--beat-parent", action="store_true",
                    help="Also compute P@K where positive = fitness > parental WT. "
                         "Requires a known WT fitness (Absci only).")
    ap.add_argument("--save", default=None,
                    help="Optional path to save the metrics table as CSV.")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    if not os.path.exists(cfg["scored_path"]):
        print(f"ERROR: scored CSV not found at {cfg['scored_path']}", file=sys.stderr)
        print(f"Run run_benchmark.py --dataset {args.dataset} first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(cfg["scored_path"])
    print(f"Loaded {len(df)} variants from {cfg['name']}")
    pos = (df[cfg["label_col"]] == 1).sum()
    print(f"  Positives: {pos} / {len(df)} (prevalence = {pos/len(df):.3f})")

    df = add_mutcount_column(df, cfg["mutation_col"])
    results = evaluate_all(df, cfg, beat_parent=args.beat_parent)
    print_table(results, cfg, beat_parent=args.beat_parent)

    if args.beat_parent and cfg.get("wt_fitness") is not None:
        wt_fit = resolve_wt_fitness(df, cfg)
        print(f"Resolved WT fitness ({cfg['fitness_col']}) = {wt_fit:.10f}")
        rank_df, n_imp, n_tot = improver_rank_diagnostic(df, cfg, wt_fit)
        print_improver_diagnostic(rank_df, n_imp, n_tot)
        rank_path = os.path.join(PROJECT_ROOT, "results",
                                 f"{args.dataset}_improver_ranks.csv")
        rank_df.to_csv(rank_path, index=False)
        print(f"Improver-rank diagnostic saved → {rank_path}")

    if args.save:
        results.to_csv(args.save, index=False)
        print(f"Saved metrics table → {args.save}")
    else:
        default = os.path.join(PROJECT_ROOT, "results",
                               f"{args.dataset}_v2_metrics.csv")
        results.to_csv(default, index=False)
        print(f"Saved metrics table → {default}")


if __name__ == "__main__":
    main()
