"""Dilated causal convolution stack for nucleotide-level encoding."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualConvBlock(nn.Module):
    """Single dilated convolution block with residual connection.

    ``x → Conv1d → GroupNorm → GELU → Dropout → + x``
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.norm = nn.GroupNorm(8, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, d_model, seq_len) — channel-first for Conv1d.

        Returns:
            (batch, d_model, seq_len)
        """
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class NucleotideEncoder(nn.Module):
    """Encodes raw DNA reads at nucleotide resolution using dilated convolutions.

    Multi-scale dilations capture dinucleotide frequencies, codon patterns,
    regulatory motifs, and GC-content windows.

    Args:
        d_nucleotide: Embedding / convolution hidden dimension.
        n_layers: Number of residual convolution blocks.
        kernel_size: Convolution kernel width.
        max_read_length: Maximum supported DNA read length (for positional encoding).
        dropout: Dropout probability.

    Input:
        nucleotide_ids: (batch, seq_len) of nucleotide token IDs
                        [A=0, C=1, G=2, T=3, N=4].

    Output:
        (batch, seq_len, d_nucleotide) nucleotide-level features.
    """

    def __init__(
        self,
        d_nucleotide: int = 384,
        n_layers: int = 8,
        kernel_size: int = 7,
        max_read_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(5, d_nucleotide)
        self.pos_embedding = nn.Embedding(max_read_length, d_nucleotide)

        max_dilation_exp = min(n_layers, 8)
        self.blocks = nn.ModuleList(
            [
                _ResidualConvBlock(
                    d_nucleotide,
                    kernel_size,
                    dilation=2 ** (i % max_dilation_exp),
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_nucleotide)

    def forward(self, nucleotide_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            nucleotide_ids: (batch, seq_len) nucleotide token IDs.

        Returns:
            (batch, seq_len, d_nucleotide) encoded features.
        """
        B, L = nucleotide_ids.shape
        positions = torch.arange(L, device=nucleotide_ids.device).unsqueeze(0)

        x = self.embedding(nucleotide_ids) + self.pos_embedding(positions)

        x = x.transpose(1, 2)  # (B, d, L) for Conv1d
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, L, d)

        x = self.final_norm(x)
        return x
