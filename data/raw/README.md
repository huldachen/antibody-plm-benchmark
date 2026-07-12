# Data access

Raw datasets are **not redistributed** in this repository. The results tables in
`results/` ship our computed columns only (sequences, mutation counts, model
scores); the source authors' **measured values** (Absci/KyDab Kd, Mason binder
labels + enrichment) are not included. Download the data from the original sources
below, then restore the measured values with `scripts/merge_fitness.py` (see repo
README).

## Mason et al. (2021) — trastuzumab scFv CDR-H3, FACS binder/non-binder
- Source: Mason et al., *Nature Biomedical Engineering* (2021).
- Access: publicly available (see the paper's data availability).
- The shipped `results/mason_all_scored.csv` carries our computed columns only; the
  source `AgClass` label, `LogEnrichment`, and read counts are not redistributed.
- After download, place the CDR-H3 + label table under `data/raw/` and process with
  `python data/download_mason.py` (adjust the input path as needed), then restore the
  measured values into the shipped scores:
  `python scripts/merge_fitness.py mason --raw /path/to/mason_cdrh3.csv`
  (your file must contain `AASeq` and `AgClass`).

## Absci HER2 — trastuzumab CDR-H3, SPR Kd
- Source: Absci HER2 affinity dataset (Shanehsazzadeh et al.).
- Access: obtain the affinity table from the original release; treat the measured
  `Kd` as source data (do not redistribute).
- Restore the measured values into the shipped scores:
  `python scripts/merge_fitness.py absci --raw /path/to/absci_her2_spr.csv`
  (your file must contain `HCDR3` and `Kd_nM`).

## KyDab — multi-target natural-repertoire antibodies
- Source: KyDab (Zhou et al.); https://kydab.naturalantibody.com/
- Access: download per the resource's terms; treat measured `KD` as source data.
- Restore: `python scripts/merge_fitness.py kydab --raw /path/to/kydab.csv`.

## What ships in this repo
- `results/*.csv` — our computed scores + derived metrics (no source affinities).
- Model weights (ESM-2, AbLang, AntiBERTy, AntiFold) and PLM embeddings are **not**
  included; the code downloads / regenerates them (`experiments/extract_embeddings.py`,
  the scorers in `models/`).
- The trastuzumab–HER2 crystal (PDB 1N8Z) is fetched by
  `structures/fetch_and_inspect_1n8z.py` (not committed).
