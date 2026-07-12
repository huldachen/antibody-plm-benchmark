#!/usr/bin/env python3
"""
Improver-retrieval metrics for the v4 few-shot benchmark.

The v4 thesis: a real
lead-optimization round has a handful of labeled variants and no crystal
structure, and the goal is to rank an unlabeled pool so the top-K *enriches
the rare variants that beat the parent* (improvers). These metrics score that
task, as opposed to the global Spearman-rho / AUC framing of v1--v3.

Locked metric suite (pre-registered 2026-07-01, before any code):
  - Improver Precision@K       : fraction of top-K that beat the parent
  - Recall@budget              : fraction of all improvers captured in top-K
  - Fold-improvement@K         : best fold-Kd improvement found within top-K
  - Beat-parent enrichment     : Mann-Whitney U of improver vs non-improver ranks

Conventions (reproduced from reevaluate.py so numbers are comparable):
  - Higher score = predicted fitter. Ranks are descending, rank 1 = best.
  - improver = (fitness present) AND (fitness > parent_fitness), strictly.
    Fitness here is neg_log_Kd; a missing value (non-binder) is NEVER an
    improver. A naive numeric filter that coerces missing Kd to 0 reports 851
    improvers instead of 5 -- see entries/2026-07-01.md.
  - Tied scores (e.g. integer mutation count) are broken by averaging over
    `n_tiebreaks` random shuffles, matching evaluation.metrics.precision_at_k.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, rankdata


def improver_mask(fitness: Sequence[float], parent_fitness: float) -> np.ndarray:
    """
    Boolean mask of improvers: fitness strictly greater than the parent.

    Missing fitness (NaN, i.e. a non-binder with no measured Kd) is treated as
    a non-improver, never an improver.

    Parameters
    ----------
    fitness : array-like
        Measured fitness per variant (neg_log_Kd; higher = tighter binder).
    parent_fitness : float
        Parent (wild-type) fitness on the same scale.

    Returns
    -------
    np.ndarray of bool
        True where the variant beats the parent.
    """
    fit = np.asarray(fitness, dtype=float)
    with np.errstate(invalid="ignore"):
        mask = fit > parent_fitness
    mask[~np.isfinite(fit)] = False
    return mask


def near_improver_mask(fitness: Sequence[float], top_frac: float = 0.10) -> np.ndarray:
    """
    Relaxed "near-improver" positive set: the top ``top_frac`` of variants by
    measured fitness (neg_log_Kd), among those that have a measured value.

    Pre-registered secondary target (entries/2026-07-01.md): with only 5 strict
    improvers, Precision@K is jittery. The top-decile-of-binders tier (~42
    variants on Absci) is a less-rare, clearly-secondary positive set that
    de-noises the signal while staying discovery-relevant (tightest binders).
    It is a superset of the strict improvers (they are the very tightest).

    Non-binders (missing fitness) are never near-improvers.

    Parameters
    ----------
    fitness : array-like
        Measured fitness (neg_log_Kd); NaN for non-binders.
    top_frac : float
        Fraction (among finite values) that counts as near-improver. 0.10 =
        top decile.

    Returns
    -------
    np.ndarray of bool
    """
    fit = np.asarray(fitness, dtype=float)
    finite = fit[np.isfinite(fit)]
    mask = np.zeros(fit.shape, dtype=bool)
    if finite.size == 0:
        return mask
    thr = float(np.quantile(finite, 1.0 - top_frac))
    with np.errstate(invalid="ignore"):
        mask = fit >= thr
    mask[~np.isfinite(fit)] = False
    return mask


def _topk_metrics_single(order: np.ndarray, k: int, is_imp: np.ndarray,
                         neg_log_kd: np.ndarray, parent_fitness: float) -> Dict:
    """Compute P@K, recall@K, fold@K for one ranking permutation."""
    top = order[:k]
    n_imp_total = int(is_imp.sum())
    hits = int(is_imp[top].sum())
    precision = hits / k
    recall = hits / n_imp_total if n_imp_total else np.nan

    # Fold-improvement: best (max neg_log_Kd) measured binder in the top-K.
    top_fit = neg_log_kd[top]
    finite = top_fit[np.isfinite(top_fit)]
    if finite.size:
        best = float(finite.max())
        fold = 10.0 ** (best - parent_fitness)  # Kd_parent / Kd_best
    else:
        fold = np.nan  # no measured binder in the top-K at all
    return {"precision": precision, "recall": recall, "fold": fold}


def improver_retrieval_at_k(scores: Sequence[float],
                            fitness: Sequence[float],
                            parent_fitness: float,
                            k: int,
                            n_tiebreaks: int = 20,
                            random_state: int = 42,
                            positive_mask: Optional[Sequence[bool]] = None) -> Dict:
    """
    Improver Precision@K, Recall@K, and Fold-improvement@K with tie handling.

    Parameters
    ----------
    scores : array-like
        Model scores over the ranking pool (higher = predicted fitter).
    fitness : array-like
        Measured fitness (neg_log_Kd) over the same pool; NaN for non-binders.
    parent_fitness : float
        Parent fitness threshold; improver = fitness > parent_fitness.
    k : int
        Top-K budget.
    n_tiebreaks : int
        Number of random tiebreak shuffles to average over.
    random_state : int
        Seed for the tiebreak RNG (locked for reproducibility).

    Returns
    -------
    dict
        precision / recall / fold each as {mean, min, max}; plus k, n_ties,
        n_improvers, n_total.
    """
    s = np.asarray(scores, dtype=float)
    neg_log_kd = np.asarray(fitness, dtype=float)
    n = s.size
    is_imp = (improver_mask(neg_log_kd, parent_fitness)
              if positive_mask is None else np.asarray(positive_mask, dtype=bool))
    n_imp = int(is_imp.sum())
    k = int(min(k, n))

    # Diagnostic: how many scores tie at the K-th boundary (tiebreaks matter
    # more the larger this is).
    kth_val = np.sort(s)[::-1][k - 1]
    n_ties = int(np.sum(s == kth_val))

    rng = np.random.default_rng(random_state)
    prec, rec, fold = [], [], []
    for _ in range(n_tiebreaks):
        noise = rng.uniform(-1e-9, 1e-9, size=n)
        order = np.argsort(-(s + noise), kind="mergesort")
        m = _topk_metrics_single(order, k, is_imp, neg_log_kd, parent_fitness)
        prec.append(m["precision"])
        rec.append(m["recall"])
        fold.append(m["fold"])

    def _agg(vals):
        a = np.asarray(vals, dtype=float)
        if not np.any(np.isfinite(a)):
            # e.g. fold-improvement when no top-K contains a measured binder.
            return {"mean": np.nan, "min": np.nan, "max": np.nan}
        return {"mean": float(np.nanmean(a)),
                "min": float(np.nanmin(a)),
                "max": float(np.nanmax(a))}

    return {
        "precision": _agg(prec),
        "recall": _agg(rec),
        "fold": _agg(fold),
        "k": k,
        "n_ties": n_ties,
        "n_improvers": n_imp,
        "n_total": n,
    }


def beat_parent_enrichment(scores: Sequence[float],
                           fitness: Sequence[float],
                           parent_fitness: float,
                           ks: Iterable[int] = (10, 30, 100),
                           n_tiebreaks: int = 20,
                           random_state: int = 42,
                           positive_mask: Optional[Sequence[bool]] = None) -> Dict:
    """
    Mann-Whitney enrichment of improver ranks vs non-improver ranks, computed
    on a **tie-consistent** basis (see DEC-007).

    An earlier version ranked by stable mergesort (rank 1 = best, ties broken
    by input order) and ran the MW test on those positional ranks. For a
    heavily-tied score like mutation count that is dishonest: the two Absci
    improvers at NumMutations=2 sit in a 20-way score tie, and stable sort
    happened to place them at ranks 3-4 — the single most favourable tie
    assignment. That inflated MutCount's enrichment to p=0.028 while P@K/Recall
    were (correctly) tie-shuffle averaged. You cannot correct one and not the
    other.

    Fix: the MW test runs on the **raw scores** (not positional ranks), so
    scipy applies its midrank tie correction — the deterministic, principled
    equivalent of averaging over all tiebreaks. Displayed ranks use
    ``scipy.stats.rankdata(method='average')`` midranks, on the same basis. A
    tiebreak-shuffle mean p is also returned as an independent cross-check
    (the two agree to ~0.001).

    Returns
    -------
    dict
        n_improvers; improver_midranks (sorted, tie-averaged); improver_tie_ranges
        ([lo, hi] positional-rank span of each improver's score tie, aligned
        with improver_midranks); median_rank (median midrank); MW_p_anti
        (improvers score WORSE), MW_p_enrich (improvers score BETTER);
        MW_p_enrich_shuffle_mean and MW_p_enrich_frac_sig (cross-check over
        n_tiebreaks shuffles).
    """
    s = np.asarray(scores, dtype=float)
    is_imp = (improver_mask(fitness, parent_fitness)
              if positive_mask is None else np.asarray(positive_mask, dtype=bool))
    n = s.size
    n_imp = int(is_imp.sum())

    # Midranks: rank 1 = best (highest score); tied scores share the average
    # positional rank. Same tie treatment scipy uses in the MW test below.
    midranks = rankdata(-s, method="average")
    imp_mid_unsorted = midranks[is_imp]
    order = np.argsort(imp_mid_unsorted)
    imp_mid = imp_mid_unsorted[order]

    # Per-improver tie-block span [lo, hi] in positional ranks: how uncertain
    # each improver's exact rank is because of score ties (lo==hi ⇒ no tie).
    imp_scores = s[is_imp][order]
    tie_ranges = []
    for v in imp_scores:
        n_better = int(np.sum(s > v))
        n_equal = int(np.sum(s == v))
        tie_ranges.append([n_better + 1, n_better + n_equal])

    if n_imp >= 2 and (n - n_imp) >= 2:
        # MW on raw scores ⇒ proper midrank tie correction. 'greater' = improver
        # scores tend to be higher (enriched); 'less' = lower (anti-ranked).
        _, p_enrich = mannwhitneyu(s[is_imp], s[~is_imp], alternative="greater")
        _, p_anti = mannwhitneyu(s[is_imp], s[~is_imp], alternative="less")

        # Independent cross-check: explicit tiebreak shuffles, MW on positional
        # ranks each time. Mean should match p_enrich; frac_sig shows how often
        # a single arbitrary tie assignment would call it significant.
        rng = np.random.default_rng(random_state)
        ps = []
        for _ in range(n_tiebreaks):
            noise = rng.uniform(-1e-9, 1e-9, size=n)
            o = np.argsort(-(s + noise), kind="mergesort")
            r = np.empty(n, dtype=int)
            r[o] = np.arange(1, n + 1)
            _, pp = mannwhitneyu(r[is_imp], r[~is_imp], alternative="less")
            ps.append(pp)
        ps = np.asarray(ps)
        shuffle_mean = float(ps.mean())
        frac_sig = float(np.mean(ps < 0.05))
    else:
        p_enrich = p_anti = shuffle_mean = frac_sig = np.nan

    return {
        "n_improvers": n_imp,
        "improver_midranks": [round(float(x), 1) for x in imp_mid],
        "improver_tie_ranges": tie_ranges,
        "median_rank": float(np.median(imp_mid)) if n_imp else -1.0,
        "MW_p_anti": float(p_anti),
        "MW_p_enrich": float(p_enrich),
        "MW_p_enrich_shuffle_mean": shuffle_mean,
        "MW_p_enrich_frac_sig": frac_sig,
    }


def evaluate_improver_retrieval(scores: Sequence[float],
                                fitness: Sequence[float],
                                parent_fitness: float,
                                ks: Iterable[int] = (10, 30, 100),
                                n_tiebreaks: int = 20,
                                random_state: int = 42,
                                positive_mask: Optional[Sequence[bool]] = None) -> Dict:
    """
    Full pre-registered improver-retrieval suite for a single model.

    Returns a flat dict suitable for a results row:
    P@{k}, Recall@{k}, Fold@{k} (means), plus the beat-parent enrichment
    diagnostic fields.

    If ``positive_mask`` is given it defines the positive set (e.g. the
    near-improver tier); otherwise positives are the strict improvers
    (fitness > parent_fitness). Fold@{k} is always measured vs the parent.
    """
    row: Dict[str, float] = {}
    for k in ks:
        r = improver_retrieval_at_k(scores, fitness, parent_fitness, k,
                                    n_tiebreaks=n_tiebreaks,
                                    random_state=random_state,
                                    positive_mask=positive_mask)
        row[f"P@{k}"] = r["precision"]["mean"]
        row[f"Recall@{k}"] = r["recall"]["mean"]
        row[f"Fold@{k}"] = r["fold"]["mean"]
        row[f"P@{k}_ties"] = r["n_ties"]

    enr = beat_parent_enrichment(scores, fitness, parent_fitness, ks=ks,
                                 positive_mask=positive_mask)
    row["median_rank"] = enr["median_rank"]
    row["MW_p_enrich"] = enr["MW_p_enrich"]
    row["MW_p_enrich_shuf"] = enr["MW_p_enrich_shuffle_mean"]
    row["MW_p_anti"] = enr["MW_p_anti"]
    row["improver_midranks"] = enr["improver_midranks"]
    row["improver_tie_ranges"] = enr["improver_tie_ranges"]
    row["n_improvers"] = enr["n_improvers"]
    row["n_total"] = len(np.asarray(scores))
    return row


def random_baseline_metrics(fitness: Sequence[float],
                            parent_fitness: float,
                            ks: Iterable[int] = (10, 30, 100),
                            n_seeds: int = 200,
                            random_state: int = 0,
                            positive_mask: Optional[Sequence[bool]] = None) -> Dict:
    """
    Random-ranking null: mean P@K / Recall@K over `n_seeds` random score
    vectors. Establishes the chance level for every K (the line every real
    method must clear). ``positive_mask`` selects the positive set (defaults to
    the strict improvers).
    """
    neg_log_kd = np.asarray(fitness, dtype=float)
    n = neg_log_kd.size
    rng = np.random.default_rng(random_state)
    acc = {f"P@{k}": [] for k in ks}
    acc.update({f"Recall@{k}": [] for k in ks})
    for _ in range(n_seeds):
        s = rng.random(n)
        for k in ks:
            r = improver_retrieval_at_k(s, neg_log_kd, parent_fitness, k,
                                        n_tiebreaks=1,
                                        random_state=int(rng.integers(1 << 30)),
                                        positive_mask=positive_mask)
            acc[f"P@{k}"].append(r["precision"]["mean"])
            acc[f"Recall@{k}"].append(r["recall"]["mean"])
    return {key: float(np.mean(v)) for key, v in acc.items()}
