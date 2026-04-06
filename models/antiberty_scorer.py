#!/usr/bin/env python3
"""
AntiBERTy Masked Marginal Scorer for Antibody Variant Fitness Prediction.

AntiBERTy (Ruffolo et al., 2021) is an antibody-specific masked language
model trained on ~558M natural antibody sequences from the Observed
Antibody Space (OAS) database.

Scoring uses the same masked marginal approach as ESM-2 and AbLang:
  score = Σ [log P(mutant_aa | context) - log P(wt_aa | context)]
  for all mutated positions.

Install: pip install antiberty
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from antiberty import AntiBERTyRunner
    ANTIBERTY_AVAILABLE = True
except ImportError:
    ANTIBERTY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Full trastuzumab VH domain sequence
WT_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
WT_CDR_H3 = "WGGDGFYAMD"
CDR_H3_START = 98
CDR_H3_END = 108


class AntiBERTyScorer:
    """
    AntiBERTy masked marginal scorer for antibody CDR-H3 variants.

    Uses the AntiBERTy model from HuggingFace to compute masked marginal
    scores for CDR-H3 variants embedded in the full trastuzumab VH context.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize AntiBERTy scorer.

        Args:
            device: torch device ('cuda' or 'cpu'). If None, auto-detect.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required. Install with: pip install torch"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load AntiBERTy from bundled pip package checkpoint
        # (HF repo Exscientia/AntiBERTy is gated; the antiberty pip package
        #  bundles the weights locally under trained_models/)
        import antiberty as _ab
        _pkg_dir = os.path.dirname(_ab.__file__)
        _model_path = os.path.join(_pkg_dir, "trained_models", "AntiBERTy_md_smooth")
        _vocab_path = os.path.join(_pkg_dir, "trained_models", "vocab.txt")

        from transformers import BertForMaskedLM, BertTokenizer
        self.tokenizer = BertTokenizer(_vocab_path, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(_model_path).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def _embed_in_context(self, cdr_h3_seq: str, cdr_start: int, cdr_end: int) -> str:
        """Embed CDR-H3 variant in full VH context."""
        expected_len = cdr_end - cdr_start
        if len(cdr_h3_seq) != expected_len:
            raise ValueError(
                f"CDR-H3 must be {expected_len} residues for positions "
                f"{cdr_start}-{cdr_end}, got {len(cdr_h3_seq)}"
            )
        return WT_VH[:cdr_start] + cdr_h3_seq + WT_VH[cdr_end:]

    def _prepare_sequence(self, sequence: str) -> str:
        """
        Prepare sequence for AntiBERTy tokenization.

        AntiBERTy expects space-separated amino acid tokens.
        """
        return " ".join(list(sequence))

    def _get_log_probs_at_position(self, sequence: str,
                                    mask_position: int) -> np.ndarray:
        """
        Get log probabilities at a masked position.

        Args:
            sequence: VH sequence with one position to be masked.
            mask_position: 0-indexed position in the sequence to mask.

        Returns:
            Log probability array over the full vocabulary.
        """
        # Create masked sequence
        seq_list = list(sequence)
        seq_list[mask_position] = self.tokenizer.mask_token or "<mask>"
        masked_seq = " ".join(seq_list)

        # Tokenize
        inputs = self.tokenizer(
            masked_seq, return_tensors="pt", add_special_tokens=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits at masked position
        # +1 for CLS token
        logits = outputs.logits[0, mask_position + 1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs.cpu().numpy()

    def _aa_to_token_idx(self, aa: str) -> int:
        """Convert amino acid to token index."""
        token_id = self.tokenizer.convert_tokens_to_ids(aa)
        if token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Amino acid '{aa}' not recognized by AntiBERTy")
        return token_id

    def score_variant(self, cdr_h3_seq: str,
                      wt_cdr_h3: str = WT_CDR_H3) -> float:
        """
        Score a CDR-H3 variant using AntiBERTy masked marginal scoring.

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

            # Get log probabilities at masked position
            log_probs = self._get_log_probs_at_position(vh_variant, pos_in_vh)

            # Compute log-odds
            wt_idx = self._aa_to_token_idx(wt_aa)
            mut_idx = self._aa_to_token_idx(mut_aa)
            log_odds = log_probs[mut_idx] - log_probs[wt_idx]

            total_score += log_odds

        return float(total_score)

    def score_variants(self, df: pd.DataFrame, seq_col: str = "AASeq",
                       score_col: str = "antiberty_score") -> pd.DataFrame:
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
                           score_col: str = "antiberty_score",
                           device: Optional[str] = None) -> pd.DataFrame:
    """Convenience function: create scorer and score a DataFrame."""
    scorer = AntiBERTyScorer(device=device)
    return scorer.score_variants(df, seq_col=seq_col, score_col=score_col)


if __name__ == "__main__":
    assert len(WT_VH) == 120, f"WT_VH should be 120 residues, got {len(WT_VH)}"
    assert WT_VH[CDR_H3_START:CDR_H3_END] == WT_CDR_H3
    print("AntiBERTy Scorer — Self-Test")
    print(f"  WT VH: {len(WT_VH)} residues")
    print(f"  CDR-H3 [{CDR_H3_START}:{CDR_H3_END}]: {WT_CDR_H3}")
    print(f"  AntiBERTy available: {ANTIBERTY_AVAILABLE}")
    print(f"  Transformers available: {TRANSFORMERS_AVAILABLE}")
    if not ANTIBERTY_AVAILABLE:
        print("  Install with: pip install antiberty")
    print("  All assertions passed.")
