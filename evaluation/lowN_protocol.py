#!/usr/bin/env python3
"""
Low-N learning-curve protocol for the v4 few-shot improver-retrieval benchmark.

The novel axis (spec §3): for N labeled variants drawn at the natural rarity
(stratified by binder label, improvers NOT oversampled), fit a method, rank the
held-out pool, and score improver retrieval — repeated over many seeds to give
learning curves (metric vs N) with confidence intervals.

This module is generic: it takes "scorers" (callables that turn a train/test
split into per-test scores) so zero-label baselines (mutation count, random),
supervised few-shot models (ridge, GP on any features), and future structure
scorers all plug into the same harness. Metrics are always computed on the
held-out test set only — no train/test leakage.

Non-binder handling: DEC-008 (censored floor); the caller passes the already
censored target into its scorers, so this module only deals with splits + eval.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from evaluation.improver_metrics import improver_retrieval_at_k

# A scorer maps (train_idx, test_idx, seed) -> scores over the TEST set
# (higher = predicted fitter). Zero-label baselines ignore train_idx.
Scorer = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def stratified_split(n: int, n_train: int, strata: Sequence[int], seed: int):
    """
    One stratified train/test split: `n_train` labeled variants, rest held out.

    Stratifying on `strata` (the binder label) preserves the natural class
    balance in the training draw — we do NOT oversample improvers (spec §3).

    Returns
    -------
    (train_idx, test_idx) : np.ndarray, np.ndarray
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train,
                                 random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(n), np.asarray(strata)))
    return train_idx, test_idx


def evaluate_test_scores(scores_test: np.ndarray,
                         fit_test: np.ndarray,
                         parent_fitness: float,
                         strict_mask_test: np.ndarray,
                         near_mask_test: np.ndarray,
                         ks=(30, 100)) -> Dict[str, float]:
    """
    Retrieval metrics on one held-out test set, for both positive tiers.

    Returns near-improver P@30 (primary learning-curve axis — the strict tier
    is too rare for a stable P@K) and strict Recall@100 (secondary), plus their
    companions.
    """
    out: Dict[str, float] = {}
    # Near-improver precision@K (has signal; n_near larger).
    for k in ks:
        r = improver_retrieval_at_k(scores_test, fit_test, parent_fitness, k,
                                    positive_mask=near_mask_test)
        out[f"near_P@{k}"] = r["precision"]["mean"]
        out[f"near_Recall@{k}"] = r["recall"]["mean"]
    # Strict improver recall@K (rare; recall@budget is the stable summary).
    for k in ks:
        r = improver_retrieval_at_k(scores_test, fit_test, parent_fitness, k,
                                    positive_mask=strict_mask_test)
        out[f"strict_Recall@{k}"] = r["recall"]["mean"]
    return out


def run_learning_curve(scorers: Dict[str, Scorer],
                       fitness: Sequence[float],
                       strata: Sequence[int],
                       parent_fitness: float,
                       strict_mask: Sequence[bool],
                       near_mask: Sequence[bool],
                       Ns: Iterable[int],
                       n_seeds: int = 20,
                       seed0: int = 0) -> pd.DataFrame:
    """
    Drive the full learning curve.

    Parameters
    ----------
    scorers : dict name -> Scorer
        Each maps (train_idx, test_idx, seed) -> per-test scores.
    fitness : array-like
        neg_log_Kd per variant (NaN for non-binders) — used only for metrics.
    strata : array-like
        Binder label (0/1) for stratified splitting.
    parent_fitness : float
    strict_mask, near_mask : array-like of bool
        Positive sets over the WHOLE pool; subset to the test split each round.
    Ns : iterable of int
        Training-set sizes.
    n_seeds : int
    seed0 : int
        Base seed; split/seed = seed0 + i.

    Returns
    -------
    pd.DataFrame (long form): columns [method, N, seed, <metric>...].
    """
    fit = np.asarray(fitness, dtype=float)
    strict_mask = np.asarray(strict_mask, dtype=bool)
    near_mask = np.asarray(near_mask, dtype=bool)
    n = fit.size

    rows = []
    for N in Ns:
        for i in range(n_seeds):
            seed = seed0 + i
            tr, te = stratified_split(n, N, strata, seed)
            fit_te = fit[te]
            strict_te = strict_mask[te]
            near_te = near_mask[te]
            for name, fn in scorers.items():
                scores_te = np.asarray(fn(tr, te, seed), dtype=float)
                m = evaluate_test_scores(scores_te, fit_te, parent_fitness,
                                         strict_te, near_te)
                rows.append({"method": name, "N": int(N), "seed": int(seed),
                             **m})
    return pd.DataFrame(rows)


def fixed_eval_split(strata: Sequence[int],
                     force_eval_mask: Sequence[bool],
                     eval_frac: float,
                     seed: int):
    """
    One FIXED evaluation split (E2b robustness protocol; DEC-009).

    All variants in `force_eval_mask` (the strict improvers) are placed in the
    evaluation set, so the strict-tier denominator is fixed and no improver ever
    appears in training. The rest of the evaluation set is a binder-stratified
    sample of the remaining variants, sized so |eval| ~ eval_frac * n. The
    training POOL is the complement (contains zero forced positives).

    Returns
    -------
    (pool_idx, eval_idx) : np.ndarray, np.ndarray  (sorted)
    """
    strata = np.asarray(strata)
    force = np.asarray(force_eval_mask, dtype=bool)
    n = strata.size
    force_idx = np.where(force)[0]
    rest = np.where(~force)[0]
    n_eval_extra = int(round(eval_frac * n)) - force_idx.size
    n_eval_extra = int(np.clip(n_eval_extra, 1, rest.size - 1))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_eval_extra,
                                 random_state=seed)
    extra_rel, pool_rel = next(sss.split(np.zeros(rest.size), strata[rest]))
    eval_idx = np.sort(np.concatenate([force_idx, rest[extra_rel]]))
    pool_idx = np.sort(rest[pool_rel])
    return pool_idx, eval_idx


def run_learning_curve_fixed(scorers: Dict[str, Scorer],
                             fitness: Sequence[float],
                             strata: Sequence[int],
                             parent_fitness: float,
                             strict_mask: Sequence[bool],
                             near_mask: Sequence[bool],
                             pool_idx: np.ndarray,
                             eval_idx: np.ndarray,
                             Ns: Iterable[int],
                             n_seeds: int = 20,
                             seed0: int = 0) -> pd.DataFrame:
    """
    Learning curve on a FIXED held-out evaluation set (DEC-009).

    Training draws of size N come from `pool_idx` only (stratified by binder);
    every method is scored on the identical `eval_idx`. Because the evaluation
    set — and its positives — never changes, the strict-tier denominator is
    constant and the sliding-window shrinking-denominator confound is removed.
    """
    fit = np.asarray(fitness, dtype=float)
    strata = np.asarray(strata)
    strict_mask = np.asarray(strict_mask, dtype=bool)
    near_mask = np.asarray(near_mask, dtype=bool)

    fit_ev = fit[eval_idx]
    strict_ev = strict_mask[eval_idx]
    near_ev = near_mask[eval_idx]
    strata_pool = strata[pool_idx]

    rows = []
    for N in Ns:
        if N > pool_idx.size:
            continue
        for i in range(n_seeds):
            seed = seed0 + i
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(N),
                                         random_state=seed)
            tr_rel, _ = next(sss.split(np.zeros(pool_idx.size), strata_pool))
            tr = pool_idx[tr_rel]
            for name, fn in scorers.items():
                scores_ev = np.asarray(fn(tr, eval_idx, seed), dtype=float)
                m = evaluate_test_scores(scores_ev, fit_ev, parent_fitness,
                                         strict_ev, near_ev)
                rows.append({"method": name, "N": int(N), "seed": int(seed),
                             **m})
    return pd.DataFrame(rows)


def summarize(curve: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Mean and 95% normal CI of `metric` per (method, N) across seeds.

    Returns columns [method, N, mean, ci, lo, hi, n_seeds].
    """
    g = curve.groupby(["method", "N"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci"] = 1.96 * out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["lo"] = out["mean"] - out["ci"]
    out["hi"] = out["mean"] + out["ci"]
    return out.rename(columns={"count": "n_seeds"})[
        ["method", "N", "mean", "ci", "lo", "hi", "n_seeds"]]
