"""Deterministic six-frame DNA → amino acid translation."""

from __future__ import annotations

import torch
import torch.nn as nn

from triframe.utils.codon_table import build_codon_lookup_tensor, PAD_ID


class SixFrameTranslator(nn.Module):
    """Translates DNA nucleotide IDs into 6 reading frames of amino acid IDs.

    For each DNA read:
      - Forward frames 0, 1, 2: translate starting at offsets 0, 1, 2.
      - Reverse frames 3, 4, 5: reverse-complement the DNA, then translate at
        offsets 0, 1, 2.

    This module has **no learnable parameters**.

    Amino acid vocabulary (23 tokens):
        20 standard AAs (alphabetical A=0 … V=19), STOP=20, X=21, PAD=22.

    Args:
        max_aa_length: Maximum amino acid sequence length (for output padding).

    Inputs:
        nucleotide_ids: (batch, seq_len) nucleotide IDs [A=0, C=1, G=2, T=3, N=4].

    Outputs:
        aa_ids:        (batch, 6, max_aa_len) amino acid IDs (padded with PAD=22).
        frame_lengths: (batch, 6) actual AA length per frame before padding.
    """

    _RC_MAP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

    def __init__(self, max_aa_length: int = 512):
        super().__init__()
        self.max_aa_length = max_aa_length
        self.register_buffer("codon_table", build_codon_lookup_tensor())
        self.register_buffer("rc_map", self._RC_MAP)

    def _reverse_complement(self, nuc_ids: torch.Tensor) -> torch.Tensor:
        """Compute the reverse complement of nucleotide ID sequences.

        Args:
            nuc_ids: (batch, seq_len) nucleotide IDs.

        Returns:
            (batch, seq_len) reverse-complemented nucleotide IDs.
        """
        complemented = self.rc_map[nuc_ids]
        return complemented.flip(dims=[1])

    def _translate_frame(
        self, nuc_ids: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate a single reading frame.

        Args:
            nuc_ids: (batch, seq_len) nucleotide IDs.
            offset: Frame offset (0, 1, or 2).

        Returns:
            aa_ids: (batch, max_aa_len) padded amino acid IDs.
            lengths: (batch,) actual AA lengths.
        """
        B, L = nuc_ids.shape
        usable = L - offset
        n_codons = usable // 3

        if n_codons <= 0:
            aa_ids = nuc_ids.new_full((B, self.max_aa_length), PAD_ID)
            lengths = nuc_ids.new_zeros(B)
            return aa_ids, lengths

        start = offset
        end = start + n_codons * 3

        codons = nuc_ids[:, start:end].reshape(B, n_codons, 3)  # (B, n_codons, 3)
        n1 = codons[:, :, 0]
        n2 = codons[:, :, 1]
        n3 = codons[:, :, 2]

        aa = self.codon_table[n1, n2, n3]  # (B, n_codons)

        actual_len = min(n_codons, self.max_aa_length)
        aa = aa[:, :actual_len]

        # Always pad to exactly max_aa_length with PAD tokens
        if actual_len < self.max_aa_length:
            pad = nuc_ids.new_full((B, self.max_aa_length - actual_len), PAD_ID)
            aa = torch.cat([aa, pad], dim=1)

        lengths = torch.full((B,), actual_len, device=nuc_ids.device, dtype=torch.long)
        return aa, lengths

    def forward(
        self, nucleotide_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate all 6 reading frames.

        Args:
            nucleotide_ids: (batch, seq_len) nucleotide IDs.

        Returns:
            aa_ids:        (batch, 6, max_aa_len) amino acid IDs.
            frame_lengths: (batch, 6) actual AA lengths per frame.
        """
        rc_ids = self._reverse_complement(nucleotide_ids)

        all_aa = []
        all_lengths = []

        for offset in range(3):
            aa, ln = self._translate_frame(nucleotide_ids, offset)
            all_aa.append(aa)
            all_lengths.append(ln)

        for offset in range(3):
            aa, ln = self._translate_frame(rc_ids, offset)
            all_aa.append(aa)
            all_lengths.append(ln)

        aa_ids = torch.stack(all_aa, dim=1)        # (B, 6, max_aa_len)
        frame_lengths = torch.stack(all_lengths, dim=1)  # (B, 6)
        return aa_ids, frame_lengths
