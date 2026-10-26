#!/usr/bin/env python3
"""
DeLong's test for two correlated ROC AUCs.

Why this is needed here: everywhere this benchmark compares two scorers, both
are evaluated on the SAME variants, so their AUCs are correlated. Per-model
bootstrap intervals (as reported in `absci_mason_mechanism.csv`) do not test the
difference -- two intervals can overlap while the paired difference is highly
significant, and vice versa. DeLong estimates the covariance between the two
AUCs and tests the difference directly, which is the right statistic for
"does model A rank these positives better than model B".

Implementation is the fast midrank algorithm of Sun & Xu (2014), which is
O(n log n) rather than the O(n^2) original. Results agree with pROC::roc.test.

References
----------
DeLong, DeLong & Clarke-Pearson (1988) Biometrics 44:837.
Sun & Xu (2014) IEEE Signal Processing Letters 21:1389.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of `x`, averaging ranks within ties (1-based)."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(scores_pos_first: np.ndarray, n_pos: int):
    """Structural components for k scorers.

    Parameters
    ----------
    scores_pos_first : np.ndarray
        (k, n) array; columns ordered positives first, then negatives.
    n_pos : int
        Number of positive samples.

    Returns
    -------
    tuple
        (aucs, covariance) with shapes (k,) and (k, k).
    """
    m = n_pos
    n = scores_pos_first.shape[1] - m
    k = scores_pos_first.shape[0]
    pos, neg = scores_pos_first[:, :m], scores_pos_first[:, m:]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores_pos_first[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, score_a, score_b) -> dict:
    """Paired DeLong test of AUC(a) vs AUC(b) on the same samples.

    Parameters
    ----------
    y_true : array-like
        Binary labels (1 = positive).
    score_a, score_b : array-like
        Scores from the two models; higher = more positive.

    Returns
    -------
    dict
        auc_a, auc_b, delta (a - b), se_delta, z, p (two-sided), n_pos, n_neg.
        `p` is NaN when the variance estimate is degenerate (e.g. too few
        positives for the covariance to be estimable), rather than a
        misleadingly small number.
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)

    order = np.argsort(-y, kind="mergesort")   # positives first, stable
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return {"auc_a": np.nan, "auc_b": np.nan, "delta": np.nan,
                "se_delta": np.nan, "z": np.nan, "p": np.nan,
                "n_pos": n_pos, "n_neg": n_neg}

    stacked = np.vstack((a, b))[:, order]
    aucs, cov = _fast_delong(stacked, n_pos)
    contrast = np.array([[1.0, -1.0]])
    var = float(contrast @ cov @ contrast.T)
    delta = float(aucs[0] - aucs[1])
    if not np.isfinite(var) or var <= 0:
        # Degenerate covariance. If the two scorers also rank identically the
        # limit is unambiguous (no difference, p = 1); otherwise the variance
        # is simply not estimable and we say so with NaN rather than invent one.
        if abs(delta) < 1e-12:
            return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": 0.0,
                    "se_delta": 0.0, "z": 0.0, "p": 1.0,
                    "n_pos": n_pos, "n_neg": n_neg}
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
                "se_delta": np.nan, "z": np.nan, "p": np.nan,
                "n_pos": n_pos, "n_neg": n_neg}
    se = float(np.sqrt(var))
    z = delta / se
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
            "se_delta": se, "z": float(z), "p": p,
            "n_pos": n_pos, "n_neg": n_neg}
