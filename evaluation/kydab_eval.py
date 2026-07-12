"""
KyDab out-of-distribution evaluation: per-antigen Spearman ρ between
zero-shot scorer outputs and neg_log_Kd, with proper statistics
(bootstrap CI, BH-FDR correction, random null, Fisher-z pooled ρ).

The post-DEC-004 v3 hypothesis under test: zero-shot PLM signal that's
indistinguishable from sklearn baselines in-library (Mason, Absci) might
still show up out-of-distribution, where there is no labelled training
data and no single-parent WT to lean on. KyDab provides 4 antigens
totally distinct from trastuzumab/HER2, with paired VH/VL natural
antibodies and quantitative SPR Kd.

Usage:
    python evaluation/kydab_eval.py --scorers random,cdrh3_length,vh_length
    python evaluation/kydab_eval.py --scorers esm2_pll
    python evaluation/kydab_eval.py --scorers all

Outputs:
    results/kydab_scores.csv          one row per antibody, one column per scorer
    results/kydab_eval_per_antigen.csv  per (scorer, antigen) stats
    results/kydab_eval_pooled.csv     per scorer, Fisher-z pooled
    results/kydab_eval_null.csv       per-antigen random-shuffle null distribution
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
KYDAB_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "kydab_paired_with_affinity.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCORES_CACHE = os.path.join(RESULTS_DIR, "kydab_scores.csv")

ANTIGEN_COL = "project"
KD_COL = "neg_log_Kd"

# Pre-specified thresholds for "scientifically meaningful" (|ρ|) and
# "statistically significant after FDR" (corrected p). Locked here so the
# eval is preregistered, not adjusted post hoc.
SIG_RHO_THRESHOLD = 0.20
SIG_P_THRESHOLD = 0.05

DEFAULT_BOOTSTRAP = 1000
DEFAULT_NULL_SHUFFLES = 1000
DEFAULT_SEED = 42


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_kydab(strict_typhi: bool = False) -> pd.DataFrame:
    """Load the KyDab combined CSV, filter unusable Kd, dedup by antibody.

    Dedup: same heavy_seq within a project tests the same antibody. Source
    data multiplies rows when (a) the same antibody is assayed against
    several antigen variants (esp. malaria-csp-01/02 where each antibody
    is measured against 6 different CSP epitope constructs with Kd
    spanning ~5 orders of magnitude), or (b) clonally related siblings
    share CDR-H3 (typhi-01).

    Policy: keep the row with the MIN Kd per (project, heavy_seq) — i.e.
    the antibody's best-case binding affinity across whatever variants it
    was tested on. This frames the OOD test as 'does the PLM rank
    antibodies by their best-case binding tightness?'. Source ANALYTE /
    IMMUNOGEN columns are not in the processed CSV — antigen-variant-
    stratified analysis is a follow-up.

    `strict_typhi=True` restricts typhi-01 rows to those whose Kd came
    from `KD_M_PH7_2` (the most-physiological pH). Sensitivity analysis
    against the multi-pH default (PH7_2 > PH7_6 > PH5_5 priority).
    """
    df = pd.read_csv(KYDAB_CSV)
    n0 = len(df)
    df = df.dropna(subset=["heavy_seq", "heavy_cdr3", KD_COL, ANTIGEN_COL])
    df = df[np.isfinite(df[KD_COL])]
    df = df[df[KD_COL] < 30]   # neg_log_Kd > 30 means Kd < 1e-30 M, impossible
    if strict_typhi:
        # Restrict typhi-01 to PH7_2-only rows; leave other projects alone.
        n_typhi_before = (df[ANTIGEN_COL] == "typhi-01").sum()
        keep_mask = (df[ANTIGEN_COL] != "typhi-01") | \
                    (df["kd_col_source"] == "KD_M_PH7_2")
        df = df[keep_mask]
        n_typhi_after = (df[ANTIGEN_COL] == "typhi-01").sum()
        print(f"[load] strict-typhi: typhi-01 {n_typhi_before} -> "
              f"{n_typhi_after} rows (PH7_2 only)")
    n1 = len(df)
    # Drop antibodies with non-canonical AAs in heavy_seq. AbLang-1, AbLang-2
    # and AntiBERTy all reject 'X' / '*' (only the canonical 20 are in their
    # vocabularies); ESM-2 accepts them but we filter consistently so every
    # scorer sees the same set of antibodies.
    canon = set("ACDEFGHIKLMNPQRSTVWY")
    is_clean = df["heavy_seq"].apply(
        lambda s: all(c.upper() in canon for c in str(s))
    )
    df = df[is_clean]
    n_canon = len(df)
    # AbLang heavy has PositionEmbeddings(160, ...) — effective protein limit
    # 158 residues (160 minus the start/stop special tokens). Anything longer
    # crashes embedding lookup. Drop 3 typhi antibodies above this limit so
    # all 4 PLMs see the same set.
    df = df[df["heavy_seq"].str.len() <= 158]
    n_lenok = len(df)
    df = (df.sort_values(KD_COL, ascending=False)
            .drop_duplicates(subset=[ANTIGEN_COL, "heavy_seq"], keep="first")
            .reset_index(drop=True))
    n2 = len(df)
    print(f"[load] {n0} raw -> {n1} after Kd filter -> {n_canon} after "
          f"non-canonical-AA drop -> {n_lenok} after length<=158 -> {n2} "
          f"after dedup-by-(project, heavy_seq) [keep best Kd per antibody]")
    print(f"[load] per-antigen counts: "
          f"{df[ANTIGEN_COL].value_counts().to_dict()}")
    return df


# ----------------------------------------------------------------------
# Scorers — each takes the full DataFrame and returns a Series of scores
# aligned to df.index. Higher = predicted-higher-affinity (matches Mason
# / Absci convention; we Spearman-correlate vs neg_log_Kd which is also
# higher = tighter).
# ----------------------------------------------------------------------

def scorer_random(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(len(df)), index=df.index, name="random")


def scorer_cdrh3_length(df: pd.DataFrame) -> pd.Series:
    return df["heavy_cdr3"].str.len().astype(float).rename("cdrh3_length")


def scorer_vh_length(df: pd.DataFrame) -> pd.Series:
    return df["heavy_seq"].str.len().astype(float).rename("vh_length")


def scorer_esm2_pll(df: pd.DataFrame, device: str = "auto") -> pd.Series:
    """
    Mean per-position pseudo-log-likelihood of the heavy CDR-H3 conditioned
    on the rest of VH, under ESM-2 (facebook/esm2_t33_650M_UR50D).

    For each antibody:
        score = (1/L) * sum_{i in CDR-H3} log P(VH[i] | VH with position i masked)

    The mean (rather than sum) is necessary because CDR-H3 length varies
    6-15 across antibodies (especially within typhi-01); a sum would bias
    against longer CDR-H3s independently of binding affinity.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"[esm2_pll] loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mask_id = tok.mask_token_id
    scores = np.full(len(df), np.nan, dtype=np.float64)
    n_done = 0
    with torch.no_grad():
        for idx, row in df.iterrows():
            vh = str(row["heavy_seq"]).strip()
            cdr3 = str(row["heavy_cdr3"]).strip()
            if not vh or not cdr3:
                continue
            cdr3_start = vh.find(cdr3)
            if cdr3_start == -1:
                # CDR-H3 string isn't a literal substring of the VH; skip.
                continue
            cdr3_end = cdr3_start + len(cdr3)

            # Tokenise VH once; ESM-2 single-residue tokens
            enc = tok(vh, return_tensors="pt", add_special_tokens=True).to(device)
            input_ids = enc["input_ids"][0]                      # (L+2,)
            # VH residue i corresponds to token index i+1 (CLS at 0)

            total_logp = 0.0
            for j in range(cdr3_start, cdr3_end):
                masked = input_ids.clone()
                masked[j + 1] = mask_id
                out = model(input_ids=masked.unsqueeze(0))
                logits = out.logits[0, j + 1, :]
                logp = torch.log_softmax(logits, dim=-1)
                actual_id = int(input_ids[j + 1].item())
                total_logp += float(logp[actual_id].item())
            cdr_len = cdr3_end - cdr3_start
            scores[idx] = total_logp / cdr_len if cdr_len > 0 else np.nan

            n_done += 1
            if n_done % 50 == 0:
                print(f"[esm2_pll] {n_done}/{len(df)} antibodies scored")

    return pd.Series(scores, index=df.index, name="esm2_pll")


def _pll_loop_generic(df: pd.DataFrame, score_name: str,
                       per_position_logp: Callable[[str, int], float]) -> pd.Series:
    """Iterate antibodies, compute mean log P over each CDR-H3 position via the
    supplied per-position scoring callable. Used by AbLang / AbLang-2 scorers
    where each model exposes its own per-position-likelihood API.

    `per_position_logp(vh_seq, pos)` returns log P(vh_seq[pos] | vh_seq with
    pos masked) under the model.
    """
    scores = np.full(len(df), np.nan, dtype=np.float64)
    n_done = 0
    for idx, row in df.iterrows():
        vh = str(row["heavy_seq"]).strip()
        cdr3 = str(row["heavy_cdr3"]).strip()
        if not vh or not cdr3:
            continue
        cdr3_start = vh.find(cdr3)
        if cdr3_start == -1:
            continue
        cdr3_end = cdr3_start + len(cdr3)
        cdr_len = cdr3_end - cdr3_start
        if cdr_len < 1:
            continue
        total = 0.0
        for j in range(cdr3_start, cdr3_end):
            total += per_position_logp(vh, j)
        scores[idx] = total / cdr_len
        n_done += 1
        if n_done % 50 == 0:
            print(f"[{score_name}] {n_done}/{len(df)} antibodies scored")
    return pd.Series(scores, index=df.index, name=score_name)


def scorer_ablang_pll(df: pd.DataFrame, device: str = "auto") -> pd.Series:
    """Mean per-CDR-H3-position pseudo-log-likelihood under AbLang-1 (heavy).

    AbLang's mode='likelihood' returns shape (B, L+2, 20) of *logits*; we
    log-softmax across the 20-AA dim. AA→idx is shifted by -1 from the
    tokenizer's vocab_to_token (the <start> column is dropped from the
    likelihood output).
    """
    import ablang as ab
    import torch

    print(f"[ablang_pll] loading ablang heavy")
    model = ab.pretrained("heavy")
    model.freeze()

    # AA -> idx mapping for the (20,) likelihood vector
    aa_to_idx: dict[str, int] = {}
    tok = model.tokenizer
    if hasattr(tok, "vocab_to_token"):
        for aa, i in tok.vocab_to_token.items():
            if len(aa) == 1 and aa.isalpha():
                aa_to_idx[aa] = i - 1
    if len(aa_to_idx) < 20:
        # Fallback observed in the v2 scorer
        for i, aa in enumerate("MRHKDESTNQCGPAVIFYWL"):
            aa_to_idx[aa] = i

    def per_pos_logp(vh: str, pos: int) -> float:
        masked = vh[:pos] + "*" + vh[pos + 1:]
        lik = model([masked], mode="likelihood")  # (1, L+2, 20)
        token_pos = pos + 1  # +1 for <start>
        logits = torch.tensor(lik[0, token_pos, :], dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1).numpy()
        actual_aa = vh[pos]
        return float(log_probs[aa_to_idx[actual_aa]])

    return _pll_loop_generic(df, "ablang_pll", per_pos_logp)


def scorer_antiberty_pll(df: pd.DataFrame, device: str = "auto") -> pd.Series:
    """Mean per-CDR-H3-position pseudo-log-likelihood under AntiBERTy."""
    import os as _os
    import torch
    from transformers import BertForMaskedLM, BertTokenizer

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    import antiberty as _ab
    pkg_dir = _os.path.dirname(_ab.__file__)
    vocab_path = _os.path.join(pkg_dir, "trained_models", "vocab.txt")
    model_path = _os.path.join(pkg_dir, "trained_models", "AntiBERTy_md_smooth")
    print(f"[antiberty_pll] loading AntiBERTy from {model_path}")
    tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mask_token = tokenizer.mask_token  # "[MASK]"

    def per_pos_logp(vh: str, pos: int) -> float:
        # AntiBERTy expects space-separated AA tokens
        masked = list(vh)
        masked[pos] = mask_token
        seq = " ".join(masked)
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, pos + 1, :]  # +1 for [CLS]
        logp = torch.log_softmax(logits, dim=-1)
        actual_id = tokenizer.convert_tokens_to_ids(vh[pos])
        return float(logp[actual_id].item())

    return _pll_loop_generic(df, "antiberty_pll", per_pos_logp)


def scorer_ablang2_pll(df: pd.DataFrame, device: str = "auto") -> pd.Series:
    """Mean per-CDR-H3-position pseudo-log-likelihood under AbLang-2 (paired
    checkpoint, heavy-only mode via empty light chain).

    Per literature/ablang2.md, mode='likelihood' returns per-position log
    probabilities; we extract the actual-AA log prob at each CDR-H3 position.
    """
    import ablang2 as ab2
    import torch

    if device == "auto":
        # ablang2 runs fine on CPU; MPS support varies by version.
        device = "cpu"

    print(f"[ablang2_pll] loading ablang2-paired (device={device})")
    model = ab2.pretrained(model_to_use="ablang2-paired", device=device)
    model.freeze()

    # Discover AA -> idx mapping from the model's tokenizer if possible;
    # else use the v2 fallback (same vocab as AbLang-1 minus the leading
    # special token, in the order MRHKDESTNQCGPAVIFYWL).
    aa_to_idx: dict[str, int] = {}
    if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "vocab_to_token"):
        for aa, i in model.tokenizer.vocab_to_token.items():
            if len(aa) == 1 and aa.isalpha():
                aa_to_idx[aa] = i
    if len(aa_to_idx) < 20:
        for i, aa in enumerate("MRHKDESTNQCGPAVIFYWL"):
            aa_to_idx[aa] = i

    def per_pos_logp(vh: str, pos: int) -> float:
        masked = vh[:pos] + "*" + vh[pos + 1:]
        # AbLang-2 paired expects [VH, VL] pair; pass empty VL for heavy-only.
        out = model([[masked, ""]], mode="likelihood")
        # Expect shape (1, L+special, vocab); take heavy-only logits
        if isinstance(out, list):
            arr = np.asarray(out[0])
        else:
            arr = np.asarray(out)
        # Some ablang2 versions return shape (1, L+2, V); index position +1
        # for the start token.
        token_pos = pos + 1
        if arr.ndim == 3:
            vec = arr[0, token_pos, :]
        elif arr.ndim == 2:
            vec = arr[token_pos, :]
        else:
            raise RuntimeError(f"unexpected ablang2 output shape: {arr.shape}")
        # vec may be raw logits or already-softmaxed; log-softmax is safe
        # either way (idempotent on log-probs only if they sum to 1 in
        # log space, but for our ranking purposes the monotone transform
        # is fine).
        t = torch.tensor(vec, dtype=torch.float32)
        log_probs = torch.log_softmax(t, dim=-1).numpy()
        actual_aa = vh[pos]
        idx = aa_to_idx.get(actual_aa)
        if idx is None or idx >= len(log_probs):
            return np.nan
        return float(log_probs[idx])

    return _pll_loop_generic(df, "ablang2_pll", per_pos_logp)


SCORER_REGISTRY: dict[str, Callable[..., pd.Series]] = {
    "random": scorer_random,
    "cdrh3_length": scorer_cdrh3_length,
    "vh_length": scorer_vh_length,
    "esm2_pll": scorer_esm2_pll,
    "ablang_pll": scorer_ablang_pll,
    "antiberty_pll": scorer_antiberty_pll,
    "ablang2_pll": scorer_ablang2_pll,
}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def spearman_with_bootstrap(
    x: np.ndarray, y: np.ndarray, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED
) -> dict:
    """Spearman ρ + asymptotic two-sided p + bootstrap 95% CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return {"n": n, "rho": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot[b] = r
    ci_lo, ci_hi = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    return {"n": n, "rho": float(rho), "p": float(p), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def fisher_z_pool(rho_n_pairs: list[tuple[float, int]]) -> dict:
    """Fisher-z weighted average of correlations across strata.

    Each stratum contributes z_i = atanh(ρ_i) with weight w_i = n_i - 3.
    Pooled z is then a normal variate; back-transform with tanh.
    """
    if not rho_n_pairs:
        return {"rho": np.nan, "se": np.nan, "z": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n_total": 0}
    zs, ws = [], []
    for rho, n in rho_n_pairs:
        if not np.isfinite(rho) or n < 4 or abs(rho) >= 1:
            continue
        zs.append(np.arctanh(rho))
        ws.append(n - 3)
    if not zs:
        return {"rho": np.nan, "se": np.nan, "z": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n_total": 0}
    zs = np.array(zs)
    ws = np.array(ws, dtype=float)
    z_pool = (zs * ws).sum() / ws.sum()
    se = 1.0 / np.sqrt(ws.sum())
    z_stat = z_pool / se
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lo = np.tanh(z_pool - 1.96 * se)
    ci_hi = np.tanh(z_pool + 1.96 * se)
    return {
        "rho": float(np.tanh(z_pool)),
        "se": float(se),
        "z": float(z_stat),
        "p": float(p),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_total": int(sum(n for _, n in rho_n_pairs)),
    }


def random_null_per_antigen(
    df: pd.DataFrame, n_shuffles: int = DEFAULT_NULL_SHUFFLES, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Per-antigen empirical null distribution of Spearman ρ under random ranking."""
    rng = np.random.default_rng(seed)
    out_rows = []
    for antigen, group in df.groupby(ANTIGEN_COL):
        y = group[KD_COL].to_numpy()
        n = len(y)
        if n < 4:
            continue
        rhos = np.empty(n_shuffles)
        for s in range(n_shuffles):
            rhos[s] = stats.spearmanr(rng.permutation(y), y)[0]
        out_rows.append({
            "antigen": antigen,
            "n": n,
            "null_mean": float(rhos.mean()),
            "null_std": float(rhos.std()),
            "null_p025": float(np.quantile(rhos, 0.025)),
            "null_p975": float(np.quantile(rhos, 0.975)),
        })
    return pd.DataFrame(out_rows)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = SIG_P_THRESHOLD) -> np.ndarray:
    """Return BH-FDR-corrected p-values (a.k.a. q-values)."""
    pvals = np.asarray(pvals, dtype=float)
    finite = np.isfinite(pvals)
    out = np.full_like(pvals, np.nan)
    if not finite.any():
        return out
    ps = pvals[finite]
    order = np.argsort(ps)
    m = len(ps)
    ranked = ps[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    qfilled = np.empty(m)
    qfilled[order] = np.minimum(q, 1.0)
    out[finite] = qfilled
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def per_antigen_table(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sc in score_cols:
        for antigen, group in df.groupby(ANTIGEN_COL):
            r = spearman_with_bootstrap(group[sc].to_numpy(), group[KD_COL].to_numpy())
            rows.append({"scorer": sc, "antigen": antigen, **r})
    out = pd.DataFrame(rows)
    out["q_fdr"] = benjamini_hochberg(out["p"].to_numpy())
    out["sig_stat"] = (out["q_fdr"] < SIG_P_THRESHOLD)
    out["sig_meaningful"] = (out["rho"].abs() >= SIG_RHO_THRESHOLD)
    return out


def pooled_table(per_antigen: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sc in score_cols:
        sub = per_antigen[per_antigen.scorer == sc]
        rho_n = list(zip(sub["rho"].tolist(), sub["n"].tolist()))
        rows.append({"scorer": sc, **fisher_z_pool(rho_n)})
    return pd.DataFrame(rows)


def print_pretty(per_antigen: pd.DataFrame, pooled: pd.DataFrame, null_df: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("Per-antigen Spearman ρ vs neg_log_Kd  (95% bootstrap CI, BH-FDR p)")
    print("=" * 84)
    print(f"{'scorer':<16s} {'antigen':<16s} {'n':>4s}  {'rho':>7s}  {'CI 95%':>16s}  "
          f"{'p':>8s}  {'q_FDR':>8s}  flags")
    for _, r in per_antigen.iterrows():
        flags = []
        if r.sig_stat: flags.append("stat-sig")
        if r.sig_meaningful: flags.append("|rho|>0.2")
        print(f"{r.scorer:<16s} {r.antigen:<16s} {int(r.n):>4d}  "
              f"{r.rho:>+7.3f}  [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  "
              f"{r.p:>8.1e}  {r.q_fdr:>8.1e}  {' '.join(flags)}")

    print("\n" + "=" * 84)
    print("Pooled Spearman ρ (Fisher-z weighted across antigens, n_i - 3 weights)")
    print("=" * 84)
    print(f"{'scorer':<16s}  {'rho':>7s}  {'CI 95%':>16s}  {'z':>6s}  {'p':>8s}  {'n_total':>8s}")
    for _, r in pooled.iterrows():
        print(f"{r.scorer:<16s}  {r.rho:>+7.3f}  [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  "
              f"{r.z:>+6.2f}  {r.p:>8.1e}  {int(r.n_total):>8d}")

    print("\n" + "=" * 84)
    print(f"Random null distribution per antigen ({DEFAULT_NULL_SHUFFLES} shuffles)")
    print("=" * 84)
    print(f"{'antigen':<16s}  {'n':>5s}  {'null_mean':>10s}  {'null_std':>9s}  "
          f"{'null 95% band':>20s}")
    for _, r in null_df.iterrows():
        print(f"{r.antigen:<16s}  {int(r.n):>5d}  {r.null_mean:>+10.4f}  {r.null_std:>9.4f}  "
              f"[{r.null_p025:+.3f},{r.null_p975:+.3f}]")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scorers", default="random,cdrh3_length,vh_length",
                   help="Comma-separated scorer names, or 'all'.")
    p.add_argument("--device", default="auto")
    p.add_argument("--use-cached", action="store_true",
                   help="If set, reuse scores in results/kydab_scores.csv when "
                        "the scorer's column already exists.")
    p.add_argument("--strict-typhi", action="store_true",
                   help="Sensitivity analysis: restrict typhi-01 to PH7_2-only "
                        "rows (most-physiological pH; default is multi-pH "
                        "priority PH7_2>PH7_6>PH5_5).")
    p.add_argument("--out-suffix", default="",
                   help="Suffix for output filenames (e.g. '_strictpH72').")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.scorers == "all":
        sc_names = list(SCORER_REGISTRY)
    else:
        sc_names = [s.strip() for s in args.scorers.split(",") if s.strip()]
    unknown = [s for s in sc_names if s not in SCORER_REGISTRY]
    if unknown:
        print(f"[error] unknown scorers: {unknown}; known: {list(SCORER_REGISTRY)}",
              file=sys.stderr)
        return 2

    df = load_kydab(strict_typhi=args.strict_typhi)

    # Score cache: keyed by (project, heavy_seq). When the dedup or filter
    # policy changes the set of rows, cached scores for surviving antibodies
    # are reused; only new antibodies are re-scored.
    if args.use_cached and os.path.exists(SCORES_CACHE):
        cached = pd.read_csv(SCORES_CACHE)
    else:
        cached = pd.DataFrame()

    scores = df[[ANTIGEN_COL, "ID", "heavy_seq", "heavy_cdr3", KD_COL]].copy()
    for sc_name in sc_names:
        cached_col = None
        if args.use_cached and not cached.empty and sc_name in cached.columns \
                and {"project", "heavy_seq"}.issubset(cached.columns):
            lookup = cached.set_index([ANTIGEN_COL, "heavy_seq"])[sc_name]
            keys = list(zip(scores[ANTIGEN_COL], scores["heavy_seq"]))
            cached_col = pd.Series([lookup.get(k, np.nan) for k in keys],
                                   index=scores.index)
            n_hit = cached_col.notna().sum()
            n_miss = len(cached_col) - n_hit
            if n_miss == 0:
                print(f"[scorer] {sc_name}: using cached values for all "
                      f"{n_hit} antibodies")
                scores[sc_name] = cached_col.values
                continue
            print(f"[scorer] {sc_name}: cache hit {n_hit}/{len(cached_col)}; "
                  f"computing remaining {n_miss}")
        else:
            print(f"[scorer] {sc_name}: computing (no cache)")

        fn = SCORER_REGISTRY[sc_name]
        if sc_name in ("esm2_pll", "ablang_pll", "antiberty_pll", "ablang2_pll"):
            s = fn(df, device=args.device)
        elif sc_name == "random":
            s = fn(df, seed=args.seed)
        else:
            s = fn(df)
        if cached_col is not None:
            cached_col[cached_col.isna()] = s.values[cached_col.isna()]
            scores[sc_name] = cached_col.values
        else:
            scores[sc_name] = s.values

        # Incremental cache save so a later scorer's crash doesn't lose
        # earlier work (each PLM is ~10 min on MPS; re-running everything
        # because the last one in the list errored is wasteful).
        scores.to_csv(SCORES_CACHE, index=False)
        print(f"[save] partial scores after '{sc_name}' -> {SCORES_CACHE}")

    print(f"[save] final scores -> {SCORES_CACHE}")

    # Statistics
    per_antigen = per_antigen_table(scores, sc_names)
    pooled = pooled_table(per_antigen, sc_names)
    null_df = random_null_per_antigen(df, n_shuffles=DEFAULT_NULL_SHUFFLES, seed=args.seed)

    sfx = args.out_suffix
    per_antigen.to_csv(os.path.join(RESULTS_DIR, f"kydab_eval_per_antigen{sfx}.csv"), index=False)
    pooled.to_csv(os.path.join(RESULTS_DIR, f"kydab_eval_pooled{sfx}.csv"), index=False)
    null_df.to_csv(os.path.join(RESULTS_DIR, f"kydab_eval_null{sfx}.csv"), index=False)

    print_pretty(per_antigen, pooled, null_df)
    print(f"\n[save] per-antigen / pooled / null tables -> "
          f"{RESULTS_DIR}/kydab_eval_*{sfx}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
