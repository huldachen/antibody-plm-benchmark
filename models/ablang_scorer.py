#!/usr/bin/env python3
"""
AbLang Masked Marginal Scorer for Antibody Variant Fitness Prediction.

AbLang (Olsen et al., Oxford) is an antibody-specific language model
trained on the Observed Antibody Space (OAS) database. It uses a
BERT-like architecture specifically designed for antibody sequences.

Scoring uses the same masked marginal approach as ESM-2:
  score = Σ [log P(mutant_aa | context) - log P(wt_aa | context)]
  for all mutated positions.

AbLang has separate models for heavy and light chains (ablang-H, ablang-L).
For this benchmark (heavy chain CDR-H3), we use ablang-H.

Install: pip install ablang
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

try:
    import ablang
    ABLANG_AVAILABLE = True
except ImportError:
    ABLANG_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Full trastuzumab VH domain sequence
WT_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
WT_CDR_H3 = "WGGDGFYAMD"
CDR_H3_START = 98
CDR_H3_END = 108


class AbLangScorer:
    """
    AbLang masked marginal scorer for antibody CDR-H3 variants.

    Uses AbLang heavy chain model (ablang-H) to compute masked marginal
    scores. CDR-H3 variants are embedded in the full trastuzumab VH context.

    AbLang outputs likelihoods as (batch, seq_len+2, 20) where:
      - +2 accounts for <start> and <stop> tokens
      - 20 columns correspond to amino acids in the model's vocabulary order
    The AA-to-index mapping is extracted from the tokenizer's vocab_to_token dict.
    """

    def __init__(self, chain: str = "heavy", device: Optional[str] = None):
        """
        Initialize AbLang scorer.

        Args:
            chain: 'heavy' or 'light' (default: 'heavy' for CDR-H3)
            device: torch device ('cuda' or 'cpu'). If None, auto-detect.
        """
        if not ABLANG_AVAILABLE:
            raise ImportError(
                "ablang is required. Install with: pip install ablang"
            )
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required. Install with: pip install torch"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # AbLang uses 'heavy' or 'light' chain models
        self.model = ablang.pretrained(chain)
        self.model.freeze()

        # Build AA-to-index mapping for AbLang's likelihood output.
        #
        # AbLang's tokenizer vocab_to_token maps: M:1, R:2, ..., L:20
        # But mode='likelihood' returns shape (batch, seq+2, 20) where the
        # 20 columns are indices 0-19 — shifted by -1 from vocab_to_token.
        # The <start> token (index 0 in full vocab) is excluded from the
        # likelihood output, so AA indices become: M:0, R:1, ..., L:19.
        self.aa_to_idx = {}
        tokenizer = self.model.tokenizer
        if hasattr(tokenizer, 'vocab_to_token'):
            for aa, idx in tokenizer.vocab_to_token.items():
                if len(aa) == 1 and aa.isalpha():
                    self.aa_to_idx[aa] = idx - 1  # shift by -1 for likelihood output
        else:
            # Fallback: empirically determined order (MRHKDESTNQCGPAVIFYWL)
            fallback_order = "MRHKDESTNQCGPAVIFYWL"
            for i, aa in enumerate(fallback_order):
                self.aa_to_idx[aa] = i

        if len(self.aa_to_idx) < 20:
            warnings.warn(
                f"Only found {len(self.aa_to_idx)} AA mappings in AbLang tokenizer. "
                f"Expected 20. Using fallback order."
            )
            fallback_order = "MRHKDESTNQCGPAVIFYWL"
            self.aa_to_idx = {aa: i for i, aa in enumerate(fallback_order)}

    def _embed_in_context(self, cdr_h3_seq: str, cdr_start: int, cdr_end: int) -> str:
        """Embed CDR-H3 variant in full VH context."""
        expected_len = cdr_end - cdr_start
        if len(cdr_h3_seq) != expected_len:
            raise ValueError(
                f"CDR-H3 must be {expected_len} residues for positions "
                f"{cdr_start}-{cdr_end}, got {len(cdr_h3_seq)}"
            )
        return WT_VH[:cdr_start] + cdr_h3_seq + WT_VH[cdr_end:]

    def score_variant(self, cdr_h3_seq: str, wt_cdr_h3: str = WT_CDR_H3) -> float:
        """
        Score a CDR-H3 variant using AbLang masked marginal scoring.

        For each mutated position:
          1. Mask the position with '*' (AbLang mask token)
          2. Run AbLang to get log probabilities
          3. Compute log P(mutant) - log P(wt)
        Sum across all mutations.

        Args:
            cdr_h3_seq: CDR-H3 variant sequence (must match wt_cdr_h3 length).
            wt_cdr_h3: Wild-type CDR-H3 sequence.

        Returns:
            Masked marginal score (float).
        """
        cdr_len = len(wt_cdr_h3)
        if len(cdr_h3_seq) != cdr_len:
            raise ValueError(
                f"CDR-H3 variant ({len(cdr_h3_seq)} res) must match WT length ({cdr_len} res)"
            )

        # Determine CDR-H3 position in VH
        cdr_start = WT_VH.find(wt_cdr_h3)
        if cdr_start == -1:
            raise ValueError(f"WT CDR-H3 '{wt_cdr_h3}' not found in VH sequence")
        cdr_end = cdr_start + cdr_len

        vh_variant = self._embed_in_context(cdr_h3_seq, cdr_start, cdr_end)
        total_score = 0.0

        for i in range(cdr_len):
            wt_aa = wt_cdr_h3[i]
            mut_aa = cdr_h3_seq[i]

            if wt_aa == mut_aa:
                continue

            pos_in_vh = cdr_start + i

            # Create masked sequence (AbLang uses '*' as mask token)
            masked_vh = vh_variant[:pos_in_vh] + '*' + vh_variant[pos_in_vh + 1:]

            # Get log probabilities at masked position
            log_probs = self._get_log_probs_at_position(masked_vh, pos_in_vh)

            # Compute log-odds using our AA-to-index mapping
            wt_idx = self.aa_to_idx[wt_aa]
            mut_idx = self.aa_to_idx[mut_aa]

            log_odds = log_probs[mut_idx] - log_probs[wt_idx]
            total_score += log_odds

        return float(total_score)

    def _get_log_probs_at_position(self, masked_sequence: str,
                                    position: int) -> np.ndarray:
        """
        Get log probabilities for all amino acids at a masked position.

        AbLang outputs shape (1, seq_len+2, 20):
          - +2 for <start> and <stop> tokens
          - 20 amino acid log-likelihoods
        Position in output = position_in_seq + 1 (for <start> token).

        Args:
            masked_sequence: VH sequence with one '*' mask token.
            position: 0-indexed position of the mask in the sequence.

        Returns:
            Array of shape (20,) with log probabilities for each amino acid.
        """
        # AbLang likelihood mode: returns numpy array (batch, seq_len+2, 20)
        likelihoods = self.model([masked_sequence], mode='likelihood')

        # +1 offset for the <start> token prepended by AbLang
        token_position = position + 1
        pos_logits = likelihoods[0, token_position, :]

        # Convert to log-softmax (likelihoods may be raw logits)
        log_probs = torch.nn.functional.log_softmax(
            torch.tensor(pos_logits, dtype=torch.float32), dim=-1
        ).numpy()

        return log_probs

    def score_variants(self, df: pd.DataFrame, seq_col: str = "AASeq",
                       score_col: str = "ablang_score") -> pd.DataFrame:
        """Score multiple CDR-H3 variants from a DataFrame."""
        scores = []
        for idx, row in df.iterrows():
            try:
                score = self.score_variant(row[seq_col])
                scores.append(score)
            except Exception as e:
                scores.append(np.nan)
                warnings.warn(f"Failed to score {row[seq_col]}: {e}")

        result = df.copy()
        result[score_col] = scores
        return result


def score_variants_from_df(df: pd.DataFrame, seq_col: str = "AASeq",
                           score_col: str = "ablang_score",
                           device: Optional[str] = None) -> pd.DataFrame:
    """Convenience function: create scorer and score a DataFrame."""
    scorer = AbLangScorer(chain="heavy", device=device)
    return scorer.score_variants(df, seq_col=seq_col, score_col=score_col)


if __name__ == "__main__":
    assert len(WT_VH) == 120, f"WT_VH should be 120 residues, got {len(WT_VH)}"
    assert WT_VH[CDR_H3_START:CDR_H3_END] == WT_CDR_H3
    print("AbLang Scorer — Self-Test")
    print(f"  WT VH: {len(WT_VH)} residues")
    print(f"  CDR-H3 [{CDR_H3_START}:{CDR_H3_END}]: {WT_CDR_H3}")
    print(f"  AbLang available: {ABLANG_AVAILABLE}")
    if not ABLANG_AVAILABLE:
        print("  Install with: pip install ablang")
    print("  All assertions passed.")
