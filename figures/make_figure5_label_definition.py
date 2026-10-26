#!/usr/bin/env python3
"""
F5 - the label definition, not the model, decides the verdict.

The headline claim needs a figure that shows the controlled comparison directly:
hold the dataset and the model scores fixed, change ONLY what counts as a
positive, and watch the mutation counter and the PLMs move in opposite
directions. A scatter of PLM-AUROC against MutCount-AUROC (figMech) shows the
mechanism but not the manipulation; a slope chart shows the manipulation.

Panel A is the controlled sweep: all three Absci conditions are the SAME 1266
variants scored by the SAME models -- only the positive-set definition changes.
Panel B is the task flip, where the dataset does change (Mason), so it is kept
visually separate rather than implied to be part of the controlled series.

Input : results/absci_mason_mechanism.csv  (AUROC, bootstrap CI, DeLong p)
Output: figures/F5_label_definition.png
Run   : python figures/make_figure5_label_definition.py   (plm env)
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.paths import result_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = result_path("absci_mason_mechanism.csv")
OUT = os.path.join(ROOT, "figures", "F5_label_definition.png")

MUT = ("MutCount", "#111111")
PLMS = [("ESM-2", "#C0392B"), ("AbLang", "#1F6FB2"), ("AntiBERTy", "#2E8B57")]
ABSCI = ["Absci binders", "Absci near-improvers", "Absci strict improvers"]
SHORT = {"Absci binders": "binders\n(n=420)",
         "Absci near-improvers": "near-improvers\n(n=42)",
         "Absci strict improvers": "strict improvers\n(n=5)"}


def _series(d, cond, name):
    r = d[d.condition == cond].iloc[0]
    key = "mutcount" if name == "MutCount" else name
    return (float(r[f"{key}_auroc"]), float(r[f"{key}_lo"]), float(r[f"{key}_hi"]))


def main() -> None:
    d = pd.read_csv(SRC)
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12.4, 5.3), gridspec_kw={"width_ratios": [2.15, 1]})

    # ---------- Panel A: controlled sweep ----------
    x = range(len(ABSCI))
    for name, colour in [MUT] + PLMS:
        ys, los, his = zip(*[_series(d, c, name) for c in ABSCI])
        lw = 2.8 if name == "MutCount" else 1.8
        axA.plot(x, ys, "o-", color=colour, lw=lw, ms=7 if name == "MutCount" else 5.5,
                 label=name, zorder=3 if name == "MutCount" else 2)
        axA.fill_between(x, los, his, color=colour, alpha=0.13, lw=0)
    axA.axhline(0.5, color="0.45", ls="--", lw=1.1, zorder=1)
    axA.text(0.02, 0.503, "chance", color="0.4", fontsize=8.5, va="bottom")
    axA.set_xticks(list(x)); axA.set_xticklabels([SHORT[c] for c in ABSCI], fontsize=9.5)
    axA.set_ylabel("AUROC (95% CI, stratified bootstrap)", fontsize=10)
    axA.set_xlabel("definition of a positive  —  same variants, same scores",
                   fontsize=10)
    axA.set_title("A. Only the label definition changes", fontsize=11.5, loc="left",
                  fontweight="bold")
    axA.set_ylim(0.1, 0.95)
    axA.legend(frameon=False, fontsize=9, loc="lower left")
    # The strict tier's intervals balloon because n_pos = 5. That is the honest
    # reason the headline rests on the n = 42 tier, so label it rather than
    # letting the reader assume the wide band is noise in the plot.
    axA.axvspan(1.72, 2.28, color="0.93", zorder=0)
    axA.text(2.0, 0.145, "n = 5: intervals too wide\nto carry a claim",
             ha="center", fontsize=8.2, color="0.42", style="italic")

    # annotate the decisive comparison
    r = d[d.condition == "Absci near-improvers"].iloc[0]
    axA.annotate(
        f"MutCount vs AbLang\nΔ = {float(r['AbLang_vs_mut_delta']):+.2f}, "
        f"DeLong p = {float(r['AbLang_vs_mut_p']):.0e}",
        xy=(1.03, 0.60), xytext=(1.30, 0.80), fontsize=8.8, color="0.25",
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    axA.annotate("", xy=(1, float(r["mutcount_auroc"])), xytext=(1, float(r["AbLang_auroc"])),
                 arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))

    # ---------- Panel B: task flip (different dataset) ----------
    conds = ["Mason binders", "Absci near-improvers"]
    labels = ["Mason\nbinders", "Absci\nimprovers"]
    xb = range(len(conds))
    for name, colour in [MUT] + PLMS:
        ys, los, his = zip(*[_series(d, c, name) for c in conds])
        axB.plot(xb, ys, "o-", color=colour, lw=2.4 if name == "MutCount" else 1.6,
                 ms=6 if name == "MutCount" else 5)
        axB.fill_between(xb, los, his, color=colour, alpha=0.13, lw=0)
    axB.axhline(0.5, color="0.45", ls="--", lw=1.1)
    axB.set_xticks(list(xb)); axB.set_xticklabels(labels, fontsize=9.5)
    axB.set_xlim(-0.35, 1.35); axB.set_ylim(0.1, 0.95)
    axB.set_title("B. …and the sign flips with the task", fontsize=11.5, loc="left",
                  fontweight="bold")
    axB.set_xlabel("retrieval target (dataset differs)", fontsize=10)

    for ax in (axA, axB):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("PLMs encode a germline-naturalness prior; affinity maturation "
                 "rewards parent-proximity", fontsize=10.5, color="0.3", y=0.015)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
