#!/usr/bin/env python3
"""
Evaluation metrics for antibody variant fitness prediction benchmark.

Primary metric: Spearman ρ between model score and measured fitness.
Secondary metrics: AUC-ROC (for binary classification), calibration.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


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


def auprc(predicted_score, binary_label):
    """
    Area under the Precision-Recall curve (= average precision).

    Preferred over AUC-ROC when classes are imbalanced, since it reflects
    retrieval quality on the minority (positive) class.
    """
    pred = np.asarray(predicted_score, dtype=float)
    labels = np.asarray(binary_label, dtype=int)

    mask = np.isfinite(pred)
    pred = pred[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan

    return average_precision_score(labels, pred)


def best_f1(predicted_score, binary_label):
    """
    Best F1 over all thresholds, with the corresponding positive rate.

    Returns a dict {f1, threshold, precision, recall, pos_rate}.
    Positive rate (prevalence) is reported so F1 can be interpreted against
    the trivial "always predict positive" baseline.
    """
    pred = np.asarray(predicted_score, dtype=float)
    labels = np.asarray(binary_label, dtype=int)

    mask = np.isfinite(pred)
    pred = pred[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return {"f1": np.nan, "threshold": np.nan,
                "precision": np.nan, "recall": np.nan,
                "pos_rate": float(labels.mean()) if len(labels) else np.nan}

    # Sweep unique thresholds (descending) and score each.
    order = np.argsort(-pred)
    pred_sorted = pred[order]
    labels_sorted = labels[order]
    n_pos = labels.sum()

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall + 1e-12),
                  0.0)
    best = int(np.argmax(f1))
    return {
        "f1": float(f1[best]),
        "threshold": float(pred_sorted[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "pos_rate": float(labels.mean()),
    }


def kendall_tau(predicted, observed):
    """
    Kendall's tau-b rank correlation (handles ties).

    Complements Spearman: tau-b is more conservative for noisy ranks and is
    often preferred when tied scores are common (e.g., mutation-count baseline
    where all 1-mut variants receive identical rank).
    """
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)

    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]

    if len(pred) < 3:
        return {"tau": np.nan, "pvalue": np.nan, "n": len(pred)}

    tau, pval = stats.kendalltau(pred, obs, variant="b")
    return {"tau": tau, "pvalue": pval, "n": len(pred)}


def precision_at_k(predicted_score, binary_label, k,
                   n_tiebreaks=20, random_state=42):
    """
    Precision@K with explicit tie handling.

    The lab-workflow metric: if a scientist picks the top-K predicted
    variants, what fraction are true positives?

    Tied scores are broken randomly; we average over `n_tiebreaks` shuffles
    and also report the best/worst case so readers can see the spread.
    This matters for baselines that produce many ties (e.g., mutation count,
    where all 1-mutation variants get identical rank).

    Parameters
    ----------
    predicted_score : array-like
    binary_label : array-like (0/1)
    k : int or float
        If int, top-K variants. If float in (0,1], top fraction of variants.
    n_tiebreaks : int
        Number of random tiebreak shuffles to average over.

    Returns
    -------
    dict with {mean, min, max, k, n_ties}.
    """
    pred = np.asarray(predicted_score, dtype=float)
    labels = np.asarray(binary_label, dtype=int)

    mask = np.isfinite(pred)
    pred = pred[mask]
    labels = labels[mask]
    n = len(pred)

    if isinstance(k, float) and 0 < k <= 1:
        k_int = max(1, int(round(k * n)))
    else:
        k_int = int(k)
    k_int = min(k_int, n)

    if n == 0 or k_int == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan,
                "k": k_int, "n_ties": 0}

    # Count how many predictions are tied at the K-th position boundary.
    # (Purely diagnostic — tells the reader if tiebreaks dominate the result.)
    threshold_val = np.sort(pred)[::-1][k_int - 1]
    n_ties_at_boundary = int(np.sum(pred == threshold_val))

    rng = np.random.default_rng(random_state)
    precisions = []
    for _ in range(n_tiebreaks):
        noise = rng.uniform(-1e-9, 1e-9, size=n)
        order = np.argsort(-(pred + noise))
        top = labels[order[:k_int]]
        precisions.append(top.mean())

    precisions = np.array(precisions)
    return {
        "mean": float(precisions.mean()),
        "min": float(precisions.min()),
        "max": float(precisions.max()),
        "k": k_int,
        "n_ties": n_ties_at_boundary,
    }


def precision_at_k_beat_reference(predicted_score, fitness, reference_fitness,
                                  k, n_tiebreaks=20, random_state=42):
    """
    Precision@K where a "positive" is a variant whose measured fitness
    exceeds a reference (e.g., the parental antibody's fitness).

    Addresses the real lab-discovery criterion: "of my top-K picks,
    how many are actually BETTER than the parent I already have?"
    """
    fit = np.asarray(fitness, dtype=float)
    labels = (fit > reference_fitness).astype(int)
    return precision_at_k(predicted_score, labels, k,
                          n_tiebreaks=n_tiebreaks,
                          random_state=random_state)


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

    # Classification metrics on binary labels
    results["auc_roc"] = auc_roc(df[score_col], df[label_col])
    results["auprc"] = auprc(df[score_col], df[label_col])
    f1_info = best_f1(df[score_col], df[label_col])
    results["best_f1"] = f1_info["f1"]
    results["pos_rate"] = f1_info["pos_rate"]

    # Kendall tau (complement to Spearman; tie-aware)
    kt = kendall_tau(df[score_col], df[fitness_col])
    results["kendall_tau"] = kt["tau"]

    # Lab-workflow metric: Precision at top-K picks.
    pk30 = precision_at_k(df[score_col], df[label_col], k=30)
    pk100 = precision_at_k(df[score_col], df[label_col], k=100)
    pk5pct = precision_at_k(df[score_col], df[label_col], k=0.05)
    results["p_at_30"] = pk30["mean"]
    results["p_at_100"] = pk100["mean"]
    results["p_at_top5pct"] = pk5pct["mean"]
    results["p_at_30_ties"] = pk30["n_ties"]

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

    kt = kendall_tau(model_score, true_fitness)
    print(f"Kendall τ:  {kt['tau']:.3f} (p={kt['pvalue']:.2e})")

    auc = auc_roc(model_score, binary_label)
    print(f"AUC-ROC:    {auc:.3f}")

    ap = auprc(model_score, binary_label)
    print(f"AUPRC:      {ap:.3f}  (prevalence={binary_label.mean():.3f})")

    f1i = best_f1(model_score, binary_label)
    print(f"Best F1:    {f1i['f1']:.3f} at threshold={f1i['threshold']:.2f} "
          f"(prec={f1i['precision']:.3f}, rec={f1i['recall']:.3f})")

    for k in (30, 100, 0.05):
        pk = precision_at_k(model_score, binary_label, k=k)
        label = f"P@{k}" if isinstance(k, int) else f"P@top{int(k*100)}%"
        print(f"{label:<10s} mean={pk['mean']:.3f}  "
              f"(min={pk['min']:.3f}, max={pk['max']:.3f}, "
              f"k={pk['k']}, ties={pk['n_ties']})")

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
