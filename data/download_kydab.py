#!/usr/bin/env python3
"""
Download and process KyDab paired-antibody data with quantitative binding (Kd).

Source: Zhou et al., "KyDab — a comprehensive database of antibody discovery
selection campaigns", bioRxiv 2026, DOI 10.64898/2026.03.25.713450.
Portal: https://kydab.naturalantibody.com/
License: CC-BY (per bioRxiv metadata; portal terms-of-use page didn't render
in literature recon — verify before redistribution in github.ver).

For v3 (Priority D), KyDab is the out-of-distribution generalisation test —
train on trastuzumab/HER2 (Mason + Absci), evaluate on different antigens
with paired VH/VL natural-repertoire antibodies. Note that KyDab labels are
quantitative SPR Kd only (no binary binder/non-binder column; no native
non-binders).

URL discovery: the portal is a Dash app and CDN paths use inconsistent
internal codes (e.g. KV7019, malaria-csp-kv7018, typhi-kv7003). The exact
.annotated.csv URLs below were obtained by querying the Dash backend
endpoint:

    POST https://kydab.naturalantibody.com/_dash-update-component
    body = {"output": ".._pages_content.children..._pages_store.data..",
            "outputs": [{"id": "_pages_content", "property": "children"},
                        {"id": "_pages_store", "property": "data"}],
            "inputs": [{"id": "_pages_location", "property": "pathname",
                        "value": "/<project-slug>"}, ...],
            "changedPropIds": [...]}

That returns the rendered project-page layout, which contains the actual
.annotated.csv CDN URL. We pin the URLs here for reproducibility; if any
break in the future, requery the portal to refresh.

Output: data/processed/kydab_paired_with_affinity.csv — every row from
every project where 'KD (M)' is non-null. Convenience columns:
KD_nM, neg_log_Kd, project (the portal display slug).
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

# Portal-display-slug -> annotated.csv URL + Kd column name. Verified
# 2026-05-10 by querying the Dash backend (see module docstring) and then
# inspecting each project's annotated.csv schema directly. Schemas are NOT
# uniform across projects — Kd lives under different column names depending
# on assay and pH condition.
#
# kd_col is the column to extract as 'KD (M)' (molar). For typhi-01 the
# project ships SPR Kd at three pH conditions; we pick pH 7.2 as the
# closest to physiological (consistent with pertussis/malaria conditions).
#
# Counts in comments are observed (Total paired rows / rows with the named
# Kd column non-null). The literature recon also listed abaumani-01 (256)
# and sars-cov-2-02 (249) as having Kd but inspection showed their assays
# are ELISA fluorescence intensity and sort-enrichment ratios respectively
# — neither produces Kd. Both excluded.
KYDAB_PROJECTS = {
    "pertussis": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/pertussis/pertussis-data.annotated.csv",
        "kd_col": "KD (M)",
    },                                                          # 11290 / 422
    "malaria-csp-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/malaria-csp-kv7018/Malaria-CSP-KV7018.annotated.csv",
        "kd_col": "KD (M)",
    },                                                          # 18146 / 124
    "malaria-csp-02": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/malaria-csp-kv7021/Malaria-CSP-KV7021.annotated.csv",
        "kd_col": "KD (M)",
    },                                                          # 22427 / 110
    "typhi-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/typhi-kv7003/Typhi-KV7003.annotated.csv",
        # Multi-pH fallback: pick first non-null per row in this priority
        # order. Maximises typhi-01 contribution from 144 (PH7_2 only) to
        # 496 (union); records the source pH per row in `kd_col_source`.
        # Decision logged in entries/2026-05-10.md.
        "kd_col": ["KD_M_PH7_2", "KD_M_PH7_6", "KD_M_PH5_5"],
    },                                                          #  5658 / 496
}

# All 11 KyDab projects, audited 2026-05-10. The 7 below have NO Kd column
# (verified by schema inspection — see entries/2026-05-10.md). Pinned here
# for documentation and so that re-running download_kydab.py with a future
# --include-no-kd flag (not implemented) would have URLs ready.
KYDAB_PROJECTS_NO_KD = {
    "abaumani-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/KV7019/KV7019.annotated.csv",
        "reason": "ELISA fluorescence intensity only (no Kd; n=7383)",
    },
    "sars-cov-2-02": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/KV7027/KV7027.annotated.csv",
        "reason": "sort-enrichment Numerator/Denominator/Ratio (no Kd; n=9887)",
    },
    "malaria-csp-03": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/KV7033/KV7033.annotated.csv",
        "reason": "sequence-only, no measurement columns (n=6401)",
    },
    "rsv-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/KV7022/KV7022.annotated.csv",
        "reason": "sequence-only, no measurement columns (n=10751)",
    },
    "sars-cov-2-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/KV7036/KV7036.annotated.csv",
        "reason": "sequence-only, no measurement columns (n=26379)",
    },
    "influenza-01": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/influenza-kv7015/Influenza-KV7015.annotated.csv",
        "reason": "has_kd_measurement=False for all 4106 rows",
    },
    "rsv-02": {
        "url": "https://static-ag-specific-data.naturalantibody.com/"
               "assets/data/rsv-kv7014/RSV-KV7014.annotated.csv",
        "reason": "has_kd_measurement=False for all 1099 rows",
    },
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "kydab")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

KD_COL = "KD (M)"
PREFERRED_COLS = [
    "ID", "CLONE_ID", "ANIMAL_ID", "TISSUE",
    "heavy_seq", "heavy_cdr3", "light_seq", "light_cdr3",
    "ka", "k d", KD_COL,
    "heavy_v_gene", "heavy_d_gene", "heavy_j_gene", "heavy_locus",
    "light_v_gene", "light_j_gene", "light_locus",
    "Analyte", "antigen_name",
]


def download_annotated_csv(project: str, url: str) -> str | None:
    """Download <project>.annotated.csv if absent. Return local path or None."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, f"{project}.annotated.csv")
    if os.path.exists(dest):
        print(f"  {project}: cached ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    print(f"  {project}: downloading from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"    WARN: download failed for {project}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return None
    print(f"    saved to {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def process_one_project(project: str, csv_path: str,
                        kd_col: str | list[str]) -> pd.DataFrame | None:
    """Parse the project's annotated.csv and return only Kd-measured rows.

    `kd_col` may be a single column name or a priority-ordered list. If a
    list, each row's Kd is taken from the first column in the list that has
    a non-null value for that row. The source column is recorded in the
    output `kd_col_source` field.

    The chosen Kd column is renamed to the canonical 'KD (M)' so downstream
    code sees one consistent schema across projects.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    WARN: could not parse {project}: {e}")
        return None
    n_total = len(df)

    # Normalise to a list of candidate columns
    candidates = [kd_col] if isinstance(kd_col, str) else list(kd_col)
    present = [c for c in candidates if c in df.columns]
    if not present:
        print(f"    WARN: {project} has none of {candidates}; skipping. "
              f"Columns: {list(df.columns)[:8]}...")
        return None

    # Pick first non-null among the priority list per row
    chosen = df[present[0]].copy()
    source = pd.Series([present[0]] * len(df), index=df.index)
    for col in present[1:]:
        mask = chosen.isna() & df[col].notna()
        chosen.loc[mask] = df.loc[mask, col]
        source.loc[mask] = col

    keep_mask = chosen.notna()
    keep = df.loc[keep_mask].copy()
    keep[KD_COL] = chosen.loc[keep_mask].values
    keep["kd_col_source"] = source.loc[keep_mask].values
    cols = [c for c in PREFERRED_COLS if c in keep.columns]
    keep = keep[cols + ["kd_col_source"]]
    keep["project"] = project

    if len(present) > 1:
        breakdown = keep["kd_col_source"].value_counts().to_dict()
        print(f"    {project}: {n_total} paired rows, "
              f"{len(keep)} with Kd from {present} -> {breakdown}")
    else:
        print(f"    {project}: {n_total} paired rows, "
              f"{len(keep)} with non-null '{present[0]}' (-> '{KD_COL}')")
    return keep


def process_kydab() -> pd.DataFrame:
    """Download (cached) every project, combine Kd-measured rows."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    parts: list[pd.DataFrame] = []
    failed: list[str] = []

    for project, info in KYDAB_PROJECTS.items():
        csv_path = download_annotated_csv(project, info["url"])
        if csv_path is None:
            failed.append(project)
            continue
        out = process_one_project(project, csv_path, info["kd_col"])
        if out is None:
            failed.append(project)
        else:
            parts.append(out)

    if failed:
        print(f"\n  WARN: failed projects ({len(failed)}): {failed}")
    if not parts:
        raise RuntimeError("No KyDab projects produced data; aborting.")

    df = pd.concat(parts, ignore_index=True)

    # Convenience columns. KD column is in molar (M); convert.
    kd_m = pd.to_numeric(df[KD_COL], errors="coerce")
    df["KD_nM"] = kd_m * 1e9
    df["neg_log_Kd"] = -np.log10(kd_m)

    out_path = os.path.join(PROCESSED_DIR, "kydab_paired_with_affinity.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  Combined dataset saved: {out_path}")
    print(f"  Total rows: {len(df)}")
    print(f"\n  Per-project counts:")
    print(df["project"].value_counts().to_string())
    print(f"\n  KD (nM) summary:")
    print(df["KD_nM"].describe().to_string())
    if "heavy_cdr3" in df.columns:
        print(f"\n  Heavy CDR-H3 length distribution (top 10):")
        print(df["heavy_cdr3"].str.len().value_counts().sort_index().head(10).to_string())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("KyDab — paired VH/VL with quantitative binding (Kd)")
    print("=" * 60)

    print("\n[1/2] Downloading per-project annotated.csv files...")
    for proj, info in KYDAB_PROJECTS.items():
        download_annotated_csv(proj, info["url"])

    print("\n[2/2] Combining Kd-measured rows across projects...")
    df = process_kydab()

    print("\n" + "=" * 60)
    print("Done! KyDab data ready for v3 OOD generalisation experiments.")
    print("=" * 60)
