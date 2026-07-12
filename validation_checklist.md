# Validation & Robustness Checklist

Every skeptical check run on the Absci HER2 analysis, why it mattered, the
question it answered, and the outcome. Doubles as the Methods robustness record
and reviewer-rebuttal reference. Grouped by theme; artifacts named where known.

> **Basis note.** Several distance- and mutation-count checks below were first run
> on a CDR-H3-only basis and later re-verified on the corrected full-CDR basis
> (HCDR1+HCDR2+HCDR3) that the README and figures report, after we found the Absci
> variants also vary outside CDR-H3. Where the two bases give different numbers
> (leakage %, exact-duplicate count, partial ρ) both are shown; the conclusions are
> unchanged.

---

## A. Statistical rigor of the enrichment claims

### A1 — Tie-corrected Mann–Whitney (tie-shuffle consistency)
- **What:** Recompute MutCount's strict-tier improver-enrichment p-value with
  tie-consistent scoring (midrank MW; cross-checked by 20-shuffle averaging),
  not the deterministic stable-sort ranks.
- **Why:** The initial strict-tier significance (p = 0.028) rested on two
  improvers at NumMutations = 2 sitting inside a 20-way score tie — a stable-sort
  artifact. Recall was tie-corrected but the p-value wasn't; correcting one and
  not the other is inconsistent.
- **Question fixed:** *Is MutCount's strict-tier significance real or a tie artifact?*
- **Outcome:** p moved 0.028 → **0.075** (trend, not significant). Corrected in
  §4.1. Tie correction now applied to **every** rank statistic (DEC-007).
- **Artifact:** `absci_e1_tie_sensitivity.csv`.

### A2 — Near-improver tier (statistical power)
- **What:** Add a pre-registered secondary positive set — top-decile neg_log_Kd
  among the 420 binders (n = 42) — alongside the strict tier (Kd < parent, n = 5).
- **Why:** n = 5 is too small for stable rank statistics; a real signal can look
  non-significant purely from rarity.
- **Question fixed:** *Is the enrichment signal real, or n = 5 noise?*
- **Outcome:** On the robust tier MutCount enriches at **p = 3.2×10⁻⁷**, BLOSUM62
  at 1.9×10⁻³ — the signal is real; the strict-tier "only trends" is a rarity
  effect, not absence.
- **Artifact:** `absci_near_improver_metrics.csv`.

### A3 — Metric choice: discovery (top-K) vs global (median rank)
- **What:** Report top-K / recall / enrichment, not median improver rank, as the
  headline.
- **Why:** BLOSUM62 has the *best* median improver rank yet **zero** recall@100 —
  median rewards "moderately high on average," but a synthesis campaign only cares
  about the extreme top-K.
- **Question fixed:** *Which metric matches the actual discovery decision?*
- **Outcome:** Median rank dropped as a headline; discovery metrics (Recall@K,
  P@K, AUROC) lead. Seeded the whole "correlation ≠ discovery" thesis.

---

## B. Structure arm (does inverse folding beat counting?)

### B1 — Fixed-backbone vs per-variant structure (setup check)
- **What:** Confirm whether each variant got its own structure or all shared the
  WT crystal backbone.
- **Why:** If all variants are scored against one WT backbone, the score can only
  reward WT-compatibility ≈ low mutation count — the result would be a setup
  artifact, not a statement about structure.
- **Question fixed:** *Is "structure = a mutation counter" a real finding or an
  artifact of fixed-backbone scoring?*
- **Outcome:** It **is** fixed-backbone (shared 1N8Z). Claim scoped precisely to
  *fixed-backbone* IF; per-variant structural modelling flagged as the only regime
  that could add signal, left to Future Work.
- **Artifact:** `structures/score_antifold_crystal.py`, `absci_e4_crystal_metrics.csv`.

### B2 — Partial correlation controlling for mutation count
- **What:** Among binders (n = 420), correlate AntiFold score with fitness before
  and after partialling out mutation count.
- **Why:** A raw correlation can be entirely an edit-distance confound.
- **Question fixed:** *Does inverse folding add improver signal independent of
  counting?*
- **Outcome:** raw ρ = +0.097 → **partial ρ = −0.054 (n.s.)** on the CDR-H3 basis
  (`absci_e4_partial_corr.csv`); re-scored on the corrected full-CDR basis reported
  in the README/figures, raw ρ = +0.135 → **partial ρ = +0.003 / −0.074 (n.s.)** —
  same conclusion. AntiFold vs mutation count **ρ = −0.73** (−0.71 full-CDR). No —
  fixed-backbone IF is mechanistically a counter. ESM-2's partial ρ is *nominally*
  anti on the CDR-H3 basis (−0.110, p = 0.023) but does **not** survive the full-CDR
  re-scoring, so we do not rest the mechanism on it and lead instead with the
  model-free two-prior divergence (Fig F2).
- **Artifact:** `absci_e4_partial_corr.csv` (CDR-H3 basis),
  `absci_e4b_fullcdr.csv` (full-CDR basis; F3).

---

## C. Leakage & generalization (the keystone)

### C1 — Leakage quantification (eval→train distance)
- **What:** Measure how close held-out eval variants sit to training variants in
  CDR-H3 Hamming distance.
- **Why:** Combinatorial libraries share mutations; a random split can put near-
  duplicates of training variants in the test set → the model memorizes, not
  generalizes.
- **Question fixed:** *Is the supervised "generalization" actually near-memorization?*
- **Outcome:** **39%** of eval within Hamming ≤ 2 of training; **14 exact
  duplicates**; the 5 improvers only **[2,2,3,3,4]** from the nearest training
  neighbour (CDR-H3 basis). On the corrected full-CDR basis this eases to **27%**
  within Hamming ≤ 2 and **0 exact duplicates** (Fig F1): near-duplication is real
  but was overstated by CDR-H3-only distance, and the net advantage inflation from
  leakage is ≈1.4×.
- **Artifact:** `absci_e2e_leakage.csv` (Fig F1).

### C2 — Distance-aware (leave-cluster-out) split
- **What:** Re-evaluate with eval Hamming-separated from training (cluster CDR-H3
  into components at Hamming ≤ D_LINK, assign whole clusters to eval vs pool).
- **Why:** The only way to measure generalization to genuinely novel sequences.
- **Question fixed:** *Does Ridge still beat MutCount once near-duplicates are removed?*
- **Outcome:** Advantage **~halved** (strict Recall@100 0.99 → 0.70 at N=500). The
  "labels crush counting" headline was substantially leakage.
- **Artifact:** `absci_e2f_distance_d{1,2}_summary.csv`.

### C3 — Split-topology bracket (D_LINK = 1 vs 2)
- **What:** Report two linkage thresholds as a bracket.
- **Why:** At D_LINK = 2 the library percolates into one giant core → eval, with
  singletons → train (train-on-periphery/test-on-core), a distribution-shifted
  split that likely *understates* the model.
- **Question fixed:** *Is the distance-split result an artifact of clustering topology?*
- **Outcome:** Bracketed; conclusions reported across both, not a single split.

### C4 — Distance-stratified generalization curve (seed-averaged)
- **What:** Bin eval variants by actual min-distance-to-training and plot the
  metric per bin, averaged over 20 independent draws with per-bin CIs.
- **Why:** Conditions directly on novelty (immune to the giant-component artifact)
  and cleanly separates memorization from generalization.
- **Question fixed:** *How much of the edge is memorization vs real generalization?*
- **Outcome:** Global ρ **decays 0.81 → 0.46** with distance but stays above
  MutCount (~0.20). Clean decomposition.
- **Artifact:** `absci_e2g_dist_stratified.csv` (Fig F4).

### C5 — Both metrics on the *same* strata (ρ AND improver-AUROC)
- **What:** Compute the discovery metric (near-AUROC) on the identical distance
  bins and binder subsets as ρ.
- **Why:** A global-correlation edge may not convert into a discovery edge; must
  compare like-for-like.
- **Question fixed:** *Does the generalization edge (ρ) become a discovery edge (AUROC)?*
- **Outcome:** No. AUROC gap **collapses 0.36 → 0.00 (d=4) → 0.07 (d≥5)** while ρ
  gap persists. This *is* the correlation-vs-discovery thesis, on the supervised model.

---

## D. Locking the thesis (three pre-write checks)

### D1 — Seed-averaging (not one idiosyncratic split)
- **What:** Confirm the curve is averaged over 20 train/eval draws (16 at d=1,
  where 4 draws lacked ≥2 near-improvers), CIs carried through.
- **Question fixed:** *Is the curve one lucky split?*
- **Outcome:** Clean; stated in the figure caption.

### D2 — Realistic label budget (N = 50–100, not just 500)  ← corrected the thesis
- **What:** Re-report leakage-free discovery at the budget actually being sold
  (few labels), not only the generous N = 500.
- **Why:** The paper's pitch is "few labels"; the headline must be at that budget.
- **Question fixed:** *Does the conclusion hold at the realistic budget?*
- **Outcome:** **Forced a correction.** Strict Recall@100 shows Ridge beating
  MutCount even leakage-free at N = 50–100 (0.56–0.75 vs 0.40). Prevented locking
  the over-flattened "counting always wins." Thesis became explicitly metric- *and*
  budget-dependent, with the robustness × stringency hierarchy so the noisy n=5
  tier never carries the headline.

### D3 — Within-bin mutation-count range (fairness / mechanism)
- **What:** Report the mutation-count span in each distance bin.
- **Why:** The d≥5 ρ edge could be "counting has no variance in a narrow mut-count
  band," not real generalization.
- **Question fixed:** *Is Ridge's d≥5 edge generalization or a counting-saturation artifact?*
- **Outcome:** At d≥5 mut range is 4–13, so MutCount keeps modest variance
  (ρ = 0.20); Ridge's higher ρ (0.46) is position-identity signal spread across
  the bulk, not the top. Stated plainly rather than left for a reviewer.

---

## E. Recommended but NOT yet run (before submission)

### E1 — Multiple-comparisons handling
- **What / why:** Many MW tests (≥6 models × 2 tiers × splits). Report p-values as
  nominal; state which survive Bonferroni. Headline effects (3×10⁻⁷) survive;
  borderline anti-ranking (AbLang 0.034) and AntiFold enrichment (0.012) may not.
- **Question it will fix:** *Are the borderline claims robust to multiplicity?*

### E2 — Second-dataset replication (Mason, zero-shot only)
- **What / why:** Run the zero-shot anti-ranking + trivial-baseline enrichment on
  Mason (same target, binary FACS → binder retrieval). Cheap partial-generalization
  check without full KyDab/E5.
- **Question it will fix:** *Does the zero-shot pattern replicate on a second assay?*

### E3 — Near-tier threshold sensitivity
- **What / why:** Confirm enrichment conclusions are robust to the top-decile
  cutoff (5% / 20%). Appendix.
- **Question it will fix:** *Is the near-tier result an artifact of the 10% threshold?*

---

*Provenance note: every check names its committed artifact above; the scored and
metric tables live in `results/`.*
