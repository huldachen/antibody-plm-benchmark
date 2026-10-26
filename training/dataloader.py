"""
Datasets and split utilities for LoRA fine-tuning on Mason / Absci CDR-H3 libraries.

Inputs are raw CDR-H3 sequences (per the project spec); the tokenizer is the model's
own (e.g. facebook/esm2_t33_650M_UR50D). See lab_notebook/decisions/DEC-003 for why
raw CDR-H3 is the v3 starting point rather than the v2 embed-in-VH protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# torch is needed only to build tensors for the fine-tuning loop. The split and
# leak-check paths (leak_check.py) are pure pandas/sklearn, and the analysis env
# deliberately has no torch (DEC-011), so the import must not be load-bearing.
try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    Dataset = object
    _HAS_TORCH = False


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MASON_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "mason_cdrh3_enrichment.csv")
ABSCI_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "absci_her2_spr.csv")

Task = Literal["classification", "regression"]


@dataclass
class DatasetSpec:
    name: str
    task: Task
    seq_col: str
    label_col: str
    label_dtype: str          # "long" | "float32"; resolved to a torch dtype at use
    is_classification: bool


def _spec(dataset: str, task: Task) -> DatasetSpec:
    if dataset == "mason" and task == "classification":
        return DatasetSpec("mason", task, "AASeq", "AgClass", "long", True)
    if dataset == "mason" and task == "regression":
        return DatasetSpec("mason", task, "AASeq", "LogEnrichment", "float32", False)
    if dataset == "absci" and task == "classification":
        return DatasetSpec("absci", task, "HCDR3", "Binder", "long", True)
    if dataset == "absci" and task == "regression":
        return DatasetSpec("absci", task, "HCDR3", "neg_log_Kd", "float32", False)
    raise ValueError(f"Unsupported (dataset, task): ({dataset}, {task})")


def _dedup_by_sequence(
    df: pd.DataFrame, spec: DatasetSpec, verbose: bool = True
) -> pd.DataFrame:
    """
    Deduplicate by sequence so the same sequence can never land in two splits.

    Classification: require labels to agree within each sequence group; error
    listing examples otherwise. Drop duplicates keeping the first row.

    Regression: average the continuous label within each sequence group, then
    keep one row per unique sequence.
    """
    n_dup_rows = int(df.duplicated(subset=spec.seq_col).sum())
    if n_dup_rows == 0:
        return df

    if spec.is_classification:
        per_seq = df.groupby(spec.seq_col)[spec.label_col].nunique()
        ambiguous_seqs = set(per_seq[per_seq > 1].index)
        n_rows_before = len(df)
        if ambiguous_seqs:
            df = df[~df[spec.seq_col].isin(ambiguous_seqs)]
        out = df.drop_duplicates(subset=spec.seq_col, keep="first")
        if verbose:
            print(f"[dedup] {spec.name}/{spec.task}: {n_dup_rows} duplicate-"
                  f"sequence rows; dropped {len(ambiguous_seqs)} sequence(s) "
                  f"with disagreeing labels (removed {n_rows_before - len(df)} "
                  f"rows total); kept {len(out)} unique sequences with "
                  f"consistent labels.")
        return out

    # Regression: average label within sequence group
    agg = df.groupby(spec.seq_col)[spec.label_col].mean()
    out = df.drop_duplicates(subset=spec.seq_col, keep="first").copy()
    out[spec.label_col] = out[spec.seq_col].map(agg)
    if verbose:
        print(f"[dedup] {spec.name}/{spec.task}: averaged {spec.label_col} "
              f"across {n_dup_rows} duplicate rows; kept {len(out)} unique sequences.")
    return out


def load_dataframe(dataset: str, task: Task) -> tuple[pd.DataFrame, DatasetSpec]:
    """Load and filter the processed CSV for a given dataset/task."""
    spec = _spec(dataset, task)
    if dataset == "mason":
        df = pd.read_csv(MASON_CSV)
    elif dataset == "absci":
        df = pd.read_csv(ABSCI_CSV)
        df = df[df["SameLength"] == 1].copy()
        if task == "regression":
            df = df[df["Binder"] == 1].copy()
            df = df.dropna(subset=["neg_log_Kd"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = df.dropna(subset=[spec.seq_col, spec.label_col])
    df = _dedup_by_sequence(df, spec).reset_index(drop=True)
    return df, spec


class CDRH3Dataset(Dataset):
    """Wraps a DataFrame of CDR-H3 sequences + labels for a HF tokenizer."""

    def __init__(self, df: pd.DataFrame, tokenizer, spec: DatasetSpec, max_length: int = 32):
        if not _HAS_TORCH:
            raise ModuleNotFoundError(
                "CDRH3Dataset needs torch, which the analysis env omits by design "
                "(DEC-011). Install torch, or use load_dataframe/stratified_split "
                "only -- the leak-check diagnostic needs no torch."
            )
        self.sequences = df[spec.seq_col].tolist()
        labels = df[spec.label_col].to_numpy()
        dtype = torch.long if spec.is_classification else torch.float32
        self.labels = torch.tensor(labels, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.spec = spec

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
        }


def stratified_split(
    df: pd.DataFrame,
    spec: DatasetSpec,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    80/10/10 split, stratified by label.

    Classification: stratify by the integer label.
    Regression: stratify by quartile bins of the continuous label so train/val/test
    cover the same dynamic range.
    """
    if spec.is_classification:
        strat = df[spec.label_col].to_numpy()
    else:
        strat = pd.qcut(df[spec.label_col], q=4, labels=False, duplicates="drop").to_numpy()

    train_val, test_df = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=strat
    )
    val_within = val_frac / (1.0 - test_frac)
    if spec.is_classification:
        strat_tv = train_val[spec.label_col].to_numpy()
    else:
        strat_tv = pd.qcut(
            train_val[spec.label_col], q=4, labels=False, duplicates="drop"
        ).to_numpy()
    train_df, val_df = train_test_split(
        train_val, test_size=val_within, random_state=seed, stratify=strat_tv
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def label_summary(df: pd.DataFrame, spec: DatasetSpec) -> str:
    """One-line summary of label distribution, useful for log lines."""
    if spec.is_classification:
        counts = df[spec.label_col].value_counts().to_dict()
        return f"n={len(df)} class_counts={counts}"
    arr = df[spec.label_col].to_numpy()
    return f"n={len(df)} mean={arr.mean():.3f} std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}"

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def _onehot(seqs, length: int) -> np.ndarray:
    """(N, length*20) uint8 one-hot, for Hamming via dot product."""
    idx = {a: i for i, a in enumerate(AA_VOCAB)}
    out = np.zeros((len(seqs), length * 20), dtype=np.uint8)
    for r, s in enumerate(seqs):
        for c, aa in enumerate(s[:length]):
            j = idx.get(aa)
            if j is not None:
                out[r, c * 20 + j] = 1
    return out


def distance_split(
    df: pd.DataFrame,
    spec: DatasetSpec,
    d_link: int = 1,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    chunk: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-cluster-out split: no test sequence within `d_link` of any train one.

    A random split of a combinatorial CDR library puts near-duplicates of training
    variants into the test set, so a model can memorise rather than generalise --
    the failure mode this benchmark exists to measure. Here sequences are clustered
    into connected components under Hamming <= d_link (single linkage) and WHOLE
    components are assigned to train / val / test, which guarantees every test
    sequence is at least d_link+1 substitutions from every training sequence.

    Hamming is computed as L - (A @ A.T) on a one-hot encoding, in row chunks so
    the full N x N matrix is never materialised.

    Parameters
    ----------
    d_link : int
        Linkage threshold. d_link=1 clusters exact-and-1-apart sequences.
    seed : int
        Seeds the component shuffle, so the split is reproducible.

    Returns
    -------
    tuple of pd.DataFrame
        (train, val, test)
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    seqs = df[spec.seq_col].tolist()
    L = len(seqs[0])
    A = _onehot(seqs, L)
    n = len(seqs)

    rows, cols = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        matches = A[start:stop].astype(np.int16) @ A.T.astype(np.int16)
        ham = L - matches
        r, c = np.nonzero(ham <= d_link)
        r = r + start
        keep = r < c                      # upper triangle only
        rows.append(r[keep]); cols.append(c[keep])
    rows = np.concatenate(rows) if rows else np.array([], dtype=int)
    cols = np.concatenate(cols) if cols else np.array([], dtype=int)
    g = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_comp)
    sizes = np.bincount(labels, minlength=n_comp)
    want_test, want_val = int(test_frac * n), int(val_frac * n)

    # A component larger than the whole test budget must go to train, or the
    # split silently inverts. At d_link=2 this library percolates into a giant
    # core (~30% of sequences); sending it to eval would mean training on the
    # periphery and testing on the core, a distribution shift masquerading as a
    # generalisation test. Train-on-core / test-on-periphery is the honest
    # direction, and the asymmetry is reported rather than hidden.
    assign, got_test, got_val = {}, 0, 0
    for c in order:
        if got_test + sizes[c] <= want_test:
            assign[c] = "test"; got_test += sizes[c]
        elif got_val + sizes[c] <= want_val:
            assign[c] = "val"; got_val += sizes[c]
        else:
            assign[c] = "train"
    part = np.array([assign[c] for c in labels])

    out = tuple(df[part == k].reset_index(drop=True) for k in ("train", "val", "test"))
    print(f"[split] distance-aware d_link={d_link}: {n_comp} components; "
          f"train={len(out[0])} val={len(out[1])} test={len(out[2])}")
    return out
