#!/usr/bin/env python3
"""
Evaluation metrics for antibody variant fitness prediction benchmark.

Primary metric: Spearman ρ between model score and measured fitness.
Secondary metrics: AUC-ROC (for binary classification), calibration.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def spearman_correlation(predicted, observed):
    """
    Compute Spearman rank correlation between predicted and observed fitness.

    Parameters
    ----------
    predicted : array-like
        Model scores (e.g., BLOSUM sum, PLM log-likelihood).
    observed : array-like
        Measured fitness values (e.g., log-enrichment, binding score).

    Returns
    -------
    dict
        rho: Spearman correlation coefficient.
        pvalue: two-sided p-value.
        n: number of valid observations.
    """
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)

    # Remove NaN/inf
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]

    if len(pred) < 3:
        return {"rho": np.nan, "pvalue": np.nan, "n": len(pred)}

    rho, pval = stats.spearmanr(pred, obs)
    return {"rho": rho, "pvalue": pval, "n": len(pred)}


def auc_roc(predicted_score, binary_label):
    """
    Compute AUC-ROC for binary classification (binder vs non-binder).

    Parameters
    ----------
    predicted_score : array-like
        Model scores (higher = predicted binder).
    binary_label : array-like
        True labels (1 = binder, 0 = non-binder).

    Returns
    -------
    float
        AUC-ROC score.
    """
    pred = np.asarray(predicted_score, dtype=float)
    labels = np.asarray(binary_label, dtype=int)

    mask = np.isfinite(pred)
    pred = pred[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan

    return roc_auc_score(labels, pred)


def evaluate_model(df, score_col, fitness_col="LogEnrichment",
                   label_col="AgClass"):
    """
    Run full evaluation suite for a single model on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with model scores and fitness values.
    score_col : str
        Column name for model predicted scores.
    fitness_col : str
        Column name for continuous fitness metric.
    label_col : str
        Column name for binary classification label.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    results = {}

    # Spearman on continuous fitness (all data)
    sp = spearman_correlation(df[score_col], df[fitness_col])
    results["spearman_rho"] = sp["rho"]
    results["spearman_pvalue"] = sp["pvalue"]
    results["n_total"] = sp["n"]

    # AUC-ROC on binary labels
    results["auc_roc"] = auc_roc(df[score_col], df[label_col])

    return results


def evaluate_stratified(df, score_col, position_col="PositionType",
                        fitness_col="LogEnrichment", label_col="AgClass"):
    """
    Evaluate model stratified by position type (CDR vs Framework).

    This is the key differentiating analysis for this benchmark:
    do PLMs fail more in CDRs due to non-evolutionary VDJ recombination?

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with model scores, fitness, and position type labels.
    score_col : str
        Column name for model predicted scores.
    position_col : str
        Column name for CDR/FR position labels.
    fitness_col : str
        Column name for continuous fitness metric.
    label_col : str
        Column name for binary classification label.

    Returns
    -------
    pd.DataFrame
        Metrics broken down by position type.
    """
    results = []

    # Overall
    overall = evaluate_model(df, score_col, fitness_col, label_col)
    overall["region"] = "ALL"
    results.append(overall)

    # By position type
    if position_col in df.columns:
        for region, group in df.groupby(position_col):
            if len(group) < 10:
                continue
            metrics = evaluate_model(group, score_col, fitness_col, label_col)
            metrics["region"] = region
            results.append(metrics)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Create synthetic test data
    n = 1000
    true_fitness = np.random.randn(n)
    # Model score = fitness + noise
    model_score = true_fitness + np.random.randn(n) * 0.5
    binary_label = (true_fitness > 0).astype(int)

    print("Evaluation Metrics — Self-Test")
    print("-" * 40)

    sp = spearman_correlation(model_score, true_fitness)
    print(f"Spearman ρ: {sp['rho']:.3f} (p={sp['pvalue']:.2e}, n={sp['n']})")

    auc = auc_roc(model_score, binary_label)
    print(f"AUC-ROC:    {auc:.3f}")

    # Test with DataFrame
    df = pd.DataFrame({
        "score": model_score,
        "fitness": true_fitness,
        "label": binary_label,
        "region": np.random.choice(["CDR", "FR"], n),
    })

    stratified = evaluate_stratified(
        df, "score", position_col="region",
        fitness_col="fitness", label_col="label"
    )
    print(f"\nStratified results:")
    print(stratified.to_string(index=False))
