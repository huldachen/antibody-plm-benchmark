#!/usr/bin/env python3
"""
Shared Absci few-shot setup for the E2 family of experiments.

Single source of truth so the sliding-window (e2_learning_curve.py) and
fixed-test (e2b_fixed_test.py) drivers -- and the later GP / C1 / embedding
variants -- build identical features, censored targets, positive masks, and
baseline scorers. Keeps every E2 number traceable to one place (portfolio:
every recorded number reproducible from committed code).

Non-binder handling is DEC-008 (censored floor). Positive tiers are the strict
improvers (Kd < parent) and the near-improver tier (top-decile binders).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.improver_metrics import improver_mask, near_improver_mask  # noqa: E402

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
ESM2_NPY = os.path.join(PROJECT_ROOT, "results", "absci_esm2_emb.npy")
ABLANG_NPY = os.path.join(PROJECT_ROOT, "results", "absci_ablang_emb.npy")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def one_hot(seqs) -> np.ndarray:
    """One-hot encode equal-length sequences -> (n, L*20) float array."""
    seqs = list(seqs)
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("one_hot requires equal-length sequences")
    X = np.zeros((len(seqs), L * 20), dtype=float)
    for i, s in enumerate(seqs):
        for p, a in enumerate(s):
            j = AA_IDX.get(a)
            if j is not None:
                X[i, p * 20 + j] = 1.0
    return X


def load_absci() -> dict:
    """Load Absci, build features/targets/masks. Returns a dict of arrays."""
    df = pd.read_csv(SCORED_CSV)
    fit = df["neg_log_Kd"].values.astype(float)
    binder = df["Binder"].values.astype(int)
    mutcount = df["NumMutations"].values.astype(float)
    parent = float(df.loc[df["NumMutations"] == 0, "neg_log_Kd"].iloc[0])

    strict = improver_mask(fit, parent)
    near = near_improver_mask(fit, top_frac=0.10)

    # Censored target (DEC-008): non-binders -> 1 log unit below weakest binder.
    floor = float(np.nanmin(fit)) - 1.0
    y = np.where(np.isfinite(fit), fit, floor)

    X_1hot = one_hot(df["HCDR3"].values)
    X_compact = np.hstack([X_1hot, mutcount.reshape(-1, 1)])

    # Parent-anchored delta features (Group C / C1): x_variant - x_parent, so a
    # position equal to parent is all-zero and only mutations carry signal.
    parent_seq = df.loc[df["NumMutations"] == 0, "HCDR3"].iloc[0]
    X_parent = one_hot([parent_seq])          # (1, L*20)
    X_delta = X_1hot - X_parent

    # PLM embeddings (Group B i/ii), if extracted (experiments/extract_embeddings.py).
    X_esm2 = np.load(ESM2_NPY) if os.path.exists(ESM2_NPY) else None
    X_ablang = np.load(ABLANG_NPY) if os.path.exists(ABLANG_NPY) else None

    return {
        "df": df, "fit": fit, "binder": binder, "mutcount": mutcount,
        "parent": parent, "strict": strict, "near": near, "floor": floor,
        "y": y, "X_1hot": X_1hot, "X_compact": X_compact, "X_delta": X_delta,
        "X_esm2": X_esm2, "X_ablang": X_ablang,
    }


def ridge_scorer(X: np.ndarray, y: np.ndarray):
    """A scorer that fits RidgeCV on train rows of X and predicts test rows."""
    def fn(train_idx, test_idx, seed):
        est = RidgeCV(alphas=RIDGE_ALPHAS)
        est.fit(X[train_idx], y[train_idx])
        return est.predict(X[test_idx])
    return fn


def ridge_scaled_scorer(X: np.ndarray, y: np.ndarray):
    """RidgeCV with per-fit StandardScaler — for dense PLM embeddings, whose
    features are not 0/1 and benefit from standardisation. Scaler is fit on the
    training rows only (no leakage)."""
    def fn(train_idx, test_idx, seed):
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        pipe.fit(X[train_idx], y[train_idx])
        return pipe.predict(X[test_idx])
    return fn


def embedding_scorers(data: dict) -> dict:
    """Group B (i/ii): Ridge on ESM-2 and AbLang embeddings (if extracted)."""
    y = data["y"]
    out = {}
    if data.get("X_esm2") is not None:
        out["Ridge(ESM-2 emb)"] = ridge_scaled_scorer(data["X_esm2"], y)
    if data.get("X_ablang") is not None:
        out["Ridge(AbLang emb)"] = ridge_scaled_scorer(data["X_ablang"], y)
    return out


def base_scorers(data: dict) -> dict:
    """The E2 first-cut scorer set: baselines + Ridge on one-hot / compact."""
    mutcount = data["mutcount"]
    y = data["y"]

    def mutcount_scorer(train_idx, test_idx, seed):
        return -mutcount[test_idx]                 # fewer mutations = fitter

    def random_scorer(train_idx, test_idx, seed):
        return np.random.default_rng(seed).random(len(test_idx))

    return {
        "MutCount (0 labels)": mutcount_scorer,
        "Random": random_scorer,
        "Ridge(one-hot)": ridge_scorer(data["X_1hot"], y),
        "Ridge(compact)": ridge_scorer(data["X_compact"], y),
    }


def gp_scorer(X: np.ndarray, y: np.ndarray):
    """GP regression scorer (Group B). DotProduct kernel = Bayesian linear
    model, well-matched to one-hot features; WhiteKernel absorbs the censored-
    target noise. Kernel hyperparameters are fixed (optimizer=None) to keep
    each low-N fit fast and deterministic."""
    def fn(train_idx, test_idx, seed):
        kernel = DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      alpha=0.0, optimizer=None)
        gp.fit(X[train_idx], y[train_idx])
        return gp.predict(X[test_idx])
    return fn


def c1_rank_scorer(X: np.ndarray, y: np.ndarray,
                   n_pairs: int = 20000, top_frac: float = 0.25,
                   top_emphasis: float = 0.5):
    """
    C1 (Group C, our method): a pairwise-logistic ranker (RankNet-style) on
    parent-anchored delta features, with a top-K-aware sampling emphasis.

    Instead of MSE it learns a direction w such that fitter variants score
    higher: for sampled pairs it fits logistic regression on (x_hi - x_lo).
    `top_emphasis` biases one member of each pair toward the top `top_frac` of
    the training fitness, so the loss focuses on getting the TOP of the ranking
    right (what improver retrieval cares about). Escalation path (spec Group C):
    add PU / listwise loss (future work).
    Predicts w . x_delta on the test set.
    """
    def fn(train_idx, test_idx, seed):
        rng = np.random.default_rng(2000 + seed)
        Xtr = X[train_idx]
        ytr = y[train_idx]
        m = len(train_idx)
        if m < 4:
            return X[test_idx] @ np.zeros(X.shape[1])
        thr = np.quantile(ytr, 1.0 - top_frac)
        top = np.where(ytr >= thr)[0]
        if top.size == 0:
            top = np.arange(m)
        use_top = rng.random(n_pairs) < top_emphasis
        a = np.where(use_top, rng.choice(top, size=n_pairs),
                     rng.integers(0, m, size=n_pairs))
        b = rng.integers(0, m, size=n_pairs)
        keep = ytr[a] != ytr[b]
        a, b = a[keep], b[keep]
        if a.size < 10:
            # Degenerate (all-equal targets): fall back to mean-fitness direction.
            w = Xtr.T @ (ytr - ytr.mean())
            return X[test_idx] @ w
        hi = np.where(ytr[a] > ytr[b], a, b)
        lo = np.where(ytr[a] > ytr[b], b, a)
        D = Xtr[hi] - Xtr[lo]
        flip = rng.random(D.shape[0]) < 0.5           # balance the two classes
        feats = np.where(flip[:, None], -D, D)
        labels = np.where(flip, 0, 1)
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
        clf.fit(feats, labels)
        return X[test_idx] @ clf.coef_[0]
    return fn


def group_c_scorers(data: dict) -> dict:
    """Group B GP + Group C parent-anchored ridge and C1 ranker."""
    y = data["y"]
    return {
        "GP(compact)": gp_scorer(data["X_compact"], y),
        "Ridge(parent-Δ)": ridge_scorer(data["X_delta"], y),
        "C1-Rank(parent-Δ)": c1_rank_scorer(data["X_delta"], y),
    }
