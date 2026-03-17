"""Shared-weight transformer encoder for amino acid reading frames."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PreNormTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer.

    ``LayerNorm → MHA → residual → LayerNorm → FFN → residual``
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model).
            key_padding_mask: (batch, seq_len) True where padded.

        Returns:
            (batch, seq_len, d_model)
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.drop1(h)

        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class FrameEncoder(nn.Module):
    """Encodes amino acid sequences from each reading frame using shared weights.

    A single pre-norm transformer is applied identically to all 6 frames. The
    frames are processed together in a single batch for efficiency.

    Args:
        d_frame: Model hidden dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        d_ff_ratio: Feed-forward expansion ratio.
        max_aa_length: Maximum amino acid sequence length.
        dropout: Dropout probability.
        aa_vocab_size: Amino acid vocabulary size (default 23).

    Inputs:
        aa_ids:        (batch, 6, aa_seq_len) amino acid IDs.
        frame_lengths: (batch, 6) actual AA lengths per frame.

    Output:
        (batch, 6, aa_seq_len, d_frame) per-position frame features.
    """

    def __init__(
        self,
        d_frame: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff_ratio: int = 4,
        max_aa_length: int = 512,
        dropout: float = 0.1,
        aa_vocab_size: int = 23,
    ):
        super().__init__()
        self.d_frame = d_frame

        self.embedding = nn.Embedding(aa_vocab_size, d_frame)
        self.pos_embedding = nn.Embedding(max_aa_length, d_frame)

        d_ff = d_frame * d_ff_ratio
        self.layers = nn.ModuleList(
            [
                _PreNormTransformerLayer(d_frame, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_frame)

    def forward(
        self, aa_ids: torch.Tensor, frame_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            aa_ids:        (batch, 6, aa_seq_len) amino acid IDs.
            frame_lengths: (batch, 6) actual AA lengths per frame.

        Returns:
            (batch, 6, aa_seq_len, d_frame) frame features.
        """
        B, F, S = aa_ids.shape

        flat_ids = aa_ids.reshape(B * F, S)                    # (B*6, S)
        flat_lengths = frame_lengths.reshape(B * F)            # (B*6,)

        positions = torch.arange(S, device=aa_ids.device).unsqueeze(0)
        x = self.embedding(flat_ids) + self.pos_embedding(positions)  # (B*6, S, d)

        pad_mask = torch.arange(S, device=aa_ids.device).unsqueeze(0) >= flat_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        x = self.final_norm(x)
        return x.reshape(B, F, S, self.d_frame)
