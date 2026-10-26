#!/usr/bin/env python3
"""
Single source of truth for where derived results live.

`results/` had grown to 40 flat CSVs, which reads as scatter and makes the
directory hard to navigate. They are now grouped by dataset. Scripts call
`result_path("absci_e2e_leakage.csv")` instead of hardcoding the directory, so
the physical layout is encoded in exactly one place and a future reorganisation
costs one edit here rather than ~28 across the codebase.

The subdirectory is derived from the filename prefix, which the naming
convention already carries (`<dataset>_<stage>_<what>.csv`).
"""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS = ("absci", "mason", "kydab")


def result_path(name: str) -> str:
    """Absolute path to a results file, grouped by dataset prefix.

    Parameters
    ----------
    name : str
        Bare filename, e.g. "absci_e2e_leakage.csv".

    Returns
    -------
    str
        Path under results/<dataset>/ when the prefix names a known dataset,
        otherwise directly under results/.

    Notes
    -----
    Falls back to a legacy flat path if the grouped one is absent but the flat
    file exists, so an older checkout or a partially-migrated tree still reads
    correctly instead of failing with a confusing FileNotFoundError.
    """
    prefix = name.split("_", 1)[0]
    grouped = (os.path.join(RESULTS_DIR, prefix, name) if prefix in DATASETS
               else os.path.join(RESULTS_DIR, name))
    if not os.path.exists(grouped):
        flat = os.path.join(RESULTS_DIR, name)
        if os.path.exists(flat):
            return flat
    return grouped
