#!/usr/bin/env python3
"""
Extract per-variant PLM embeddings for the Absci CDR-H3 pool (Group B i/ii).

Mean-pooled ESM-2 (facebook/esm2_t33_650M_UR50D, 1280-d) and AbLang heavy
seqcoding (768-d) embeddings of each HCDR3, cached as float32 .npy in row order
of results/absci_all_scored.csv. This is the one GPU step; downstream E2d reads
the cache so the few-shot embedding experiments need no PLM.

Consistent with the rest of the benchmark, embeddings are of the HCDR3 alone
(no VH framework context) — same input the v2/v3 scorers used. Documented
limitation, not an oversight.

Run (once):
    python experiments/extract_embeddings.py
Outputs:
    results/absci_esm2_emb.npy      (n, 1280) float32
    results/absci_ablang_emb.npy    (n, 768)  float32
Idempotent: skips a cache file that already exists.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP double-load guard

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SCORED_CSV = os.path.join(PROJECT_ROOT, "results", "absci_all_scored.csv")
ESM2_NPY = os.path.join(PROJECT_ROOT, "results", "absci_esm2_emb.npy")
ABLANG_NPY = os.path.join(PROJECT_ROOT, "results", "absci_ablang_emb.npy")
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
BATCH = 64


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_esm2(seqs) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = _device()
    print(f"  ESM-2 ({ESM2_MODEL}) on {dev} ...")
    tok = AutoTokenizer.from_pretrained(ESM2_MODEL)
    mdl = AutoModel.from_pretrained(ESM2_MODEL).to(dev).eval()
    embs = []
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc).last_hidden_state          # (b, L, 1280)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)       # mean over real tokens
        embs.append(pooled.cpu().float().numpy())
        print(f"    {min(i + BATCH, len(seqs))}/{len(seqs)}", end="\r")
    print()
    return np.vstack(embs).astype(np.float32)


def extract_ablang(seqs) -> np.ndarray:
    import ablang
    print("  AbLang heavy seqcoding ...")
    heavy = ablang.pretrained("heavy")
    heavy.freeze()
    embs = []
    for i in range(0, len(seqs), BATCH):
        batch = list(seqs[i:i + BATCH])
        embs.append(np.asarray(heavy(batch, mode="seqcoding")))
        print(f"    {min(i + BATCH, len(seqs))}/{len(seqs)}", end="\r")
    print()
    return np.vstack(embs).astype(np.float32)


def main() -> None:
    df = pd.read_csv(SCORED_CSV)
    seqs = df["HCDR3"].astype(str).tolist()
    print(f"Absci HCDR3 pool: n={len(seqs)}")

    if os.path.exists(ESM2_NPY):
        print(f"  [skip] {os.path.relpath(ESM2_NPY, PROJECT_ROOT)} exists")
    else:
        np.save(ESM2_NPY, extract_esm2(seqs))
        print(f"  saved {os.path.relpath(ESM2_NPY, PROJECT_ROOT)}")

    if os.path.exists(ABLANG_NPY):
        print(f"  [skip] {os.path.relpath(ABLANG_NPY, PROJECT_ROOT)} exists")
    else:
        np.save(ABLANG_NPY, extract_ablang(seqs))
        print(f"  saved {os.path.relpath(ABLANG_NPY, PROJECT_ROOT)}")

    for path in (ESM2_NPY, ABLANG_NPY):
        if os.path.exists(path):
            a = np.load(path)
            print(f"  {os.path.basename(path)}: shape={a.shape}, dtype={a.dtype}")


if __name__ == "__main__":
    main()
