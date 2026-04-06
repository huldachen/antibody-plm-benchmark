#!/usr/bin/env python3
"""
ESM-2 Masked Marginal Scorer for Antibody Variant Fitness Prediction

This module implements masked marginal scoring using ESM-2 (facebook/esm2_t33_650M_UR50D)
for predicting antibody CDR-H3 variant fitness. The scorer embeds CDR-H3 variants in the
full trastuzumab VH context and uses masked prediction to compute log-odds scores.

Masked Marginal Scoring (the standard for fitness prediction):
  - For each mutated position: mask it, run model, extract log P(mutant_aa) - log P(wt_aa)
  - Sum across all mutated positions
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Full trastuzumab VH domain sequence
WT_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

# CDR-H3 loop (10 residues): positions 98-107 (0-indexed) in WT_VH
# Verify: WT_VH[98:108] = "WGGDGFYAMD"
WT_CDR_H3 = "WGGDGFYAMD"
CDR_H3_START = 98
CDR_H3_END = 108  # exclusive


class ESM2Scorer:
    """
    ESM-2 Masked Marginal Scorer for antibody CDR-H3 variant fitness prediction.

    Uses the facebook/esm2_t33_650M_UR50D model to score variants by:
    1. Embedding the CDR-H3 sequence in the full VH domain context
    2. For each mutated position: masking it, running the model, and extracting
       the log-odds score (log P(mutant_aa) - log P(wt_aa))
    3. Summing across all mutated positions

    Attributes:
        model: The ESM-2 masked language model
        tokenizer: The ESM-2 tokenizer
        device: torch device (cuda or cpu)
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: Optional[str] = None):
        """
        Initialize the ESM2Scorer by loading model and tokenizer.

        Args:
            model_name: HuggingFace model identifier (default: facebook/esm2_t33_650M_UR50D)
            device: torch device to use ('cuda' or 'cpu'). If None, auto-detect.

        Raises:
            ImportError: If torch or transformers is not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for ESM2Scorer. "
                "Install with: pip install torch"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for ESM2Scorer. "
                "Install with: pip install transformers"
            )

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Ensure we're not tracking gradients
        for param in self.model.parameters():
            param.requires_grad = False

    def _embed_in_context(self, cdr_h3_seq: str, cdr_start: int = CDR_H3_START, cdr_end: int = CDR_H3_END) -> str:
        """
        Embed a CDR-H3 sequence in the full VH domain context by replacing
        the wild-type CDR-H3 region with the variant sequence.

        Args:
            cdr_h3_seq: CDR-H3 sequence (must be same length as wt region)
            cdr_start: Start index of CDR-H3 in VH (0-indexed, inclusive)
            cdr_end: End index of CDR-H3 in VH (0-indexed, exclusive)

        Returns:
            Full VH domain sequence with the variant CDR-H3 embedded
        """
        expected_len = cdr_end - cdr_start
        if len(cdr_h3_seq) != expected_len:
            raise ValueError(
                f"CDR-H3 sequence must be {expected_len} residues for positions "
                f"{cdr_start}-{cdr_end}, got {len(cdr_h3_seq)}"
            )

        # Replace CDR-H3 in full VH sequence
        vh_with_variant = WT_VH[:cdr_start] + cdr_h3_seq + WT_VH[cdr_end:]
        return vh_with_variant

    def _get_masked_logits(self, sequence: str, mask_position: int) -> np.ndarray:
        """
        Get the log probabilities for all amino acids at a masked position.

        Args:
            sequence: The full VH domain sequence with one position masked
            mask_position: The absolute position in the sequence that is masked

        Returns:
            Array of shape (20,) with log probabilities for each amino acid
        """
        # Tokenize the sequence
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Find the mask token position (accounting for special tokens)
        # The tokenizer adds <cls> at the beginning and <eos> at the end
        mask_token_position = mask_position + 1  # +1 for <cls> token

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits at the masked position
        logits = outputs.logits[0, mask_token_position, :]  # shape: (vocabulary_size,)

        # Convert to log probabilities using log_softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs.cpu().numpy()

    def _aa_to_token_idx(self, aa: str) -> int:
        """
        Convert amino acid to ESM-2 vocabulary index.

        Args:
            aa: Single amino acid character

        Returns:
            Token index in ESM-2 vocabulary

        Raises:
            ValueError: If amino acid is not recognized
        """
        # ESM-2 uses single-letter amino acid codes
        # Standard amino acids map directly
        token = self.tokenizer.convert_tokens_to_ids(aa)
        if token is None or token == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Amino acid '{aa}' not recognized. "
                f"Expected one of: ACDEFGHIKLMNPQRSTVWY"
            )
        return token

    def score_variant(self, cdr_h3_seq: str, wt_cdr_h3: str = WT_CDR_H3) -> float:
        """
        Score a CDR-H3 variant using masked marginal scoring.

        Masked Marginal Scoring procedure:
        1. For each position with a mutation (cdr_h3_seq[i] != wt_cdr_h3[i]):
           a. Mask that position in the full VH sequence
           b. Run ESM-2 to get log probabilities
           c. Extract: log P(mutant_aa) - log P(wt_aa)
        2. Sum the log-odds across all mutated positions

        Args:
            cdr_h3_seq: CDR-H3 variant sequence (must be same length as wt_cdr_h3)
            wt_cdr_h3: Wild-type CDR-H3 sequence (default: WGGDGFYAMD)

        Returns:
            Masked marginal score (float). Higher scores indicate better fitness.

        Raises:
            ValueError: If sequence length doesn't match WT or contains invalid AAs
        """
        cdr_len = len(wt_cdr_h3)
        if len(cdr_h3_seq) != cdr_len:
            raise ValueError(
                f"CDR-H3 variant ({len(cdr_h3_seq)} res) must match WT length ({cdr_len} res)"
            )

        # Determine CDR-H3 start/end in VH based on WT sequence
        cdr_start = WT_VH.find(wt_cdr_h3)
        if cdr_start == -1:
            raise ValueError(f"WT CDR-H3 '{wt_cdr_h3}' not found in VH sequence")
        cdr_end = cdr_start + cdr_len

        # Embed variant in full VH context
        vh_with_variant = self._embed_in_context(cdr_h3_seq, cdr_start, cdr_end)

        total_score = 0.0

        # Score each position
        for i in range(cdr_len):
            wt_aa = wt_cdr_h3[i]
            variant_aa = cdr_h3_seq[i]

            # Only score positions with mutations
            if wt_aa == variant_aa:
                continue

            # Position in full VH sequence
            pos_in_vh = cdr_start + i

            # Create masked sequence (replace position i with mask token)
            masked_vh = vh_with_variant[:pos_in_vh] + "<mask>" + vh_with_variant[pos_in_vh + 1:]

            # Get log probabilities at this position
            log_probs = self._get_masked_logits(masked_vh, pos_in_vh)

            # Get token indices for wild-type and variant amino acids
            wt_idx = self._aa_to_token_idx(wt_aa)
            var_idx = self._aa_to_token_idx(variant_aa)

            # Compute log-odds: log P(variant_aa) - log P(wt_aa)
            log_odds = log_probs[var_idx] - log_probs[wt_idx]
            total_score += log_odds

        return float(total_score)

    def score_variants(
        self,
        df: pd.DataFrame,
        seq_col: str = "AASeq",
        wt_col: Optional[str] = None,
        score_col: str = "esm2_score"
    ) -> pd.DataFrame:
        """
        Score multiple CDR-H3 variants from a DataFrame.

        Args:
            df: Input DataFrame with CDR-H3 sequences
            seq_col: Column name containing CDR-H3 sequences (default: "AASeq")
            wt_col: Optional column name for WT sequences (default: uses WT_CDR_H3 for all)
            score_col: Output column name for scores (default: "esm2_score")

        Returns:
            DataFrame with added score column
        """
        scores = []
        errors = []

        for idx, row in df.iterrows():
            try:
                cdr_seq = row[seq_col]
                wt_seq = row[wt_col] if wt_col and wt_col in df.columns else WT_CDR_H3
                score = self.score_variant(cdr_seq, wt_seq)
                scores.append(score)
                errors.append(None)
            except Exception as e:
                scores.append(np.nan)
                errors.append(str(e))

        result_df = df.copy()
        result_df[score_col] = scores

        # Add error column if there are any errors
        if any(e is not None for e in errors):
            result_df[f"{score_col}_error"] = errors
            n_errors = sum(1 for e in errors if e is not None)
            warnings.warn(f"Failed to score {n_errors}/{len(df)} sequences")

        return result_df


def score_variants_from_df(
    df: pd.DataFrame,
    seq_col: str = "AASeq",
    wt_col: Optional[str] = None,
    score_col: str = "esm2_score",
    device: Optional[str] = None,
    model_name: str = "facebook/esm2_t33_650M_UR50D"
) -> pd.DataFrame:
    """
    Convenience function to score variants from a DataFrame.

    Creates an ESM2Scorer, scores all variants, and returns the result.

    Args:
        df: Input DataFrame with CDR-H3 sequences
        seq_col: Column name containing CDR-H3 sequences (default: "AASeq")
        wt_col: Optional column name for WT sequences
        score_col: Output column name for scores (default: "esm2_score")
        device: torch device ('cuda' or 'cpu', default: auto-detect)
        model_name: HuggingFace model identifier

    Returns:
        DataFrame with added score column

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'AASeq': ['WGGDGFYAMD', 'WGGDGFYAMH', 'WGGDGFYAMI']
        ... })
        >>> scored_df = score_variants_from_df(df)
    """
    scorer = ESM2Scorer(model_name=model_name, device=device)
    return scorer.score_variants(df, seq_col=seq_col, wt_col=wt_col, score_col=score_col)


# Self-test: Verify the module structure and VH sequence positioning
if __name__ == "__main__":
    # Verify WT_VH and CDR-H3 positioning
    assert len(WT_VH) == 120, f"WT_VH should be 120 residues, got {len(WT_VH)}"
    assert WT_VH[CDR_H3_START:CDR_H3_END] == WT_CDR_H3, \
        f"CDR-H3 mismatch at positions {CDR_H3_START}-{CDR_H3_END}"
    assert WT_CDR_H3 == "WGGDGFYAMD", "WT_CDR_H3 sequence mismatch"

    print("✓ WT_VH sequence length: 120")
    print(f"✓ CDR-H3 (positions {CDR_H3_START}-{CDR_H3_END}): {WT_CDR_H3}")
    print("✓ All assertions passed")

    # Test the module can be imported if torch/transformers are available
    if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
        print("\n✓ torch and transformers are available")
        print("  (To test full functionality, run: python esm2.py --test)")
    else:
        missing = []
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install torch transformers")
