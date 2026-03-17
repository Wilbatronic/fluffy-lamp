"""Frame Attention with Gating — identifies coding frames and produces a gated aggregate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionPool(nn.Module):
    """Learned-query attention pooling: reduces a variable-length sequence to a single vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = math.sqrt(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool a sequence to a single vector via learned-query attention.

        Args:
            x: (batch, seq_len, d_model).
            mask: (batch, seq_len) True where padded.

        Returns:
            (batch, d_model)
        """
        scores = (self.query * x).sum(dim=-1) / self.scale  # (B, S)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, S, 1)
        return (weights * x).sum(dim=1)  # (B, d_model)


class _PreNormTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer for cross-frame self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop1(h)
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class FrameAttention(nn.Module):
    """Cross-frame attention that identifies coding frame(s) and produces a
    gated aggregate representation.

    Architecture:
        1. Attention-pool each frame's sequence to a single vector.
        2. Cross-frame self-attention over the 6 frame vectors (with frame
           position embeddings).
        3. Gating: softmax over a learned scalar per frame, weighted sum.

    Args:
        d_frame: Hidden dimension.
        n_layers: Number of cross-frame transformer layers.
        n_heads: Attention heads (for the cross-frame layers).
        dropout: Dropout probability.

    Inputs:
        frame_features: (batch, 6, aa_seq_len, d_frame) per-frame encoder outputs.
        frame_lengths:  (batch, 6) actual AA lengths per frame.

    Outputs:
        output:      (batch, d_frame) aggregated frame representation.
        frame_gates: (batch, 6) interpretable per-frame confidence (sums to 1).
    """

    def __init__(
        self,
        d_frame: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = _AttentionPool(d_frame)
        self.frame_pos = nn.Embedding(6, d_frame)

        self.layers = nn.ModuleList(
            [_PreNormTransformerLayer(d_frame, n_heads, dropout) for _ in range(n_layers)]
        )
        self.gate_proj = nn.Linear(d_frame, 1)

    def forward(
        self, frame_features: torch.Tensor, frame_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            frame_features: (batch, 6, aa_seq_len, d_frame).
            frame_lengths:  (batch, 6) actual AA lengths.

        Returns:
            output:      (batch, d_frame).
            frame_gates: (batch, 6).
        """
        B, N_FRAMES, S, D = frame_features.shape

        pooled = []
        for f in range(N_FRAMES):
            mask = torch.arange(S, device=frame_features.device).unsqueeze(0) >= frame_lengths[:, f : f + 1]
            pooled.append(self.pool(frame_features[:, f], mask))  # (B, D)
        frame_vectors = torch.stack(pooled, dim=1)  # (B, 6, D)

        frame_idx = torch.arange(N_FRAMES, device=frame_features.device)
        frame_vectors = frame_vectors + self.frame_pos(frame_idx).unsqueeze(0)

        for layer in self.layers:
            frame_vectors = layer(frame_vectors)

        gate_logits = self.gate_proj(frame_vectors).squeeze(-1)  # (B, 6)
        frame_gates = F.softmax(gate_logits, dim=-1)              # (B, 6)

        output = (frame_gates.unsqueeze(-1) * frame_vectors).sum(dim=1)  # (B, D)
        return output, frame_gates
