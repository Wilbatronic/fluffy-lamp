"""Cross-Resolution Attention — fuses nucleotide-level and codon-level features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: queries attend to key/value sequence.

    ``LayerNorm → CrossAttn → residual → LayerNorm → FFN → residual``

    Uses separate Q, K, V projections for full control over different source
    dimensions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        d_ff = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            queries: (batch, n_q, d_model).
            kv:      (batch, n_kv, d_model).
            kv_mask: (batch, n_kv) True where padded.

        Returns:
            (batch, n_q, d_model)
        """
        B, N_Q, D = queries.shape
        N_KV = kv.shape[1]

        q = self.q_proj(self.norm_q(queries))
        k = self.k_proj(self.norm_kv(kv))
        v = self.v_proj(kv)

        q = q.view(B, N_Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_KV, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_KV, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, N_Q, N_KV)

        if kv_mask is not None:
            scores = scores.masked_fill(
                kv_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N_Q, D)
        out = self.resid_drop(self.out_proj(out))

        queries = queries + out

        queries = queries + self.ff(self.norm_ff(queries))
        return queries


class CrossResolutionFusion(nn.Module):
    """Fuses nucleotide-level and codon-level features via cross-resolution attention.

    Learned query tokens, conditioned on the frame representation, attend to the
    projected nucleotide sequence. Multiple queries extract different functional
    aspects before being mean-pooled.

    Args:
        d_nucleotide: Nucleotide feature dimension (input).
        d_frame: Frame / output feature dimension.
        n_layers: Number of cross-attention layers.
        n_heads: Attention heads per cross-attention layer.
        n_queries: Number of learned query tokens.
        dropout: Dropout probability.

    Inputs:
        nuc_features: (batch, nuc_seq_len, d_nucleotide) from NucleotideEncoder.
        frame_repr:   (batch, d_frame) from FrameAttention.
        nuc_mask:     (batch, nuc_seq_len) boolean padding mask (True = pad).

    Output:
        fused: (batch, d_frame) final fused representation.
    """

    def __init__(
        self,
        d_nucleotide: int = 384,
        d_frame: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        n_queries: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nuc_proj = nn.Linear(d_nucleotide, d_frame)
        self.learned_queries = nn.Parameter(torch.randn(n_queries, d_frame))

        self.layers = nn.ModuleList(
            [_CrossAttentionLayer(d_frame, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_frame)

    def forward(
        self,
        nuc_features: torch.Tensor,
        frame_repr: torch.Tensor,
        nuc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            nuc_features: (batch, nuc_seq_len, d_nucleotide).
            frame_repr:   (batch, d_frame).
            nuc_mask:     (batch, nuc_seq_len) True where padded.

        Returns:
            (batch, d_frame)
        """
        B = nuc_features.shape[0]
        kv = self.nuc_proj(nuc_features)  # (B, L, d_frame)

        queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_q, d)
        queries = queries + frame_repr.unsqueeze(1)                     # condition on frame

        for layer in self.layers:
            queries = layer(queries, kv, kv_mask=nuc_mask)

        pooled = queries.mean(dim=1)  # (B, d_frame)

        fused = self.final_norm(pooled + frame_repr)
        return fused
