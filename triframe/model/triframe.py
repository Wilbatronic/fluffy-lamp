"""TriFrame: Frame-Resolving Model for Assembly-Free Metagenomic Functional Annotation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from triframe.model.config import TriFrameConfig
from triframe.model.nucleotide_encoder import NucleotideEncoder
from triframe.model.frame_translator import SixFrameTranslator
from triframe.model.frame_encoder import FrameEncoder
from triframe.model.frame_attention import FrameAttention
from triframe.model.cross_resolution import CrossResolutionFusion
from triframe.model.heads import TriFrameHeads


def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a boolean padding mask (True = padded position).

    Args:
        lengths: (batch,) actual sequence lengths.
        max_len: Maximum length in the batch.

    Returns:
        (batch, max_len) boolean mask.
    """
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


class TriFrameModel(nn.Module):
    """TriFrame: Frame-Resolving Model for Assembly-Free Metagenomic
    Functional Annotation.

    Given a raw DNA read as nucleotide IDs, the model:
        1. Encodes nucleotides with dilated causal convolutions.
        2. Deterministically translates 6 reading frames.
        3. Encodes each amino acid frame with a shared-weight transformer.
        4. Identifies the coding frame(s) via frame attention with gating.
        5. Fuses nucleotide and codon representations via cross-resolution
           attention.
        6. Predicts functional annotations through multi-task heads.

    Args:
        config: ``TriFrameConfig`` instance.

    Inputs:
        nucleotide_ids: (batch, seq_len) DNA read as nucleotide IDs.
        lengths:        (batch,) actual read lengths before padding.

    Outputs:
        dict containing all prediction logits plus ``frame_gates`` for
        interpretability.
    """

    def __init__(self, config: TriFrameConfig):
        super().__init__()
        self.config = config

        self.nucleotide_encoder = NucleotideEncoder(
            d_nucleotide=config.d_nucleotide,
            n_layers=config.n_nuc_layers,
            kernel_size=config.nuc_kernel_size,
            max_read_length=config.max_read_length,
            dropout=config.dropout,
        )

        self.frame_translator = SixFrameTranslator(
            max_aa_length=config.max_aa_length,
        )

        self.frame_encoder = FrameEncoder(
            d_frame=config.d_frame,
            n_layers=config.n_frame_layers,
            n_heads=config.n_frame_heads,
            d_ff_ratio=config.d_ff_ratio,
            max_aa_length=config.max_aa_length,
            dropout=config.dropout,
        )

        self.frame_attention = FrameAttention(
            d_frame=config.d_frame,
            n_layers=config.n_frame_attn_layers,
            n_heads=config.n_frame_heads,
            dropout=config.dropout,
        )

        self.cross_resolution = CrossResolutionFusion(
            d_nucleotide=config.d_nucleotide,
            d_frame=config.d_frame,
            n_layers=config.n_fusion_layers,
            n_heads=config.n_frame_heads,
            dropout=config.dropout,
        )

        self.heads = TriFrameHeads(
            d_frame=config.d_frame,
            n_ec_level1=config.n_ec_level1,
            n_ec_level2=config.n_ec_level2,
            n_ec_level3=config.n_ec_level3,
            n_ec_level4=config.n_ec_level4,
            n_kegg_orthologs=config.n_kegg_orthologs,
            n_cog_categories=config.n_cog_categories,
        )

    def forward(
        self,
        nucleotide_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            nucleotide_ids: (batch, seq_len) nucleotide IDs.
            lengths:        (batch,) actual read lengths.

        Returns:
            dict of prediction logits and ``frame_gates``.
        """
        nuc_features = self.nucleotide_encoder(nucleotide_ids)

        aa_ids, frame_lengths = self.frame_translator(nucleotide_ids)

        frame_features = self.frame_encoder(aa_ids, frame_lengths)

        frame_repr, frame_gates = self.frame_attention(frame_features, frame_lengths)

        nuc_mask = _make_padding_mask(lengths, nucleotide_ids.size(1))
        fused = self.cross_resolution(nuc_features, frame_repr, nuc_mask)

        predictions = self.heads(fused)
        predictions["frame_gates"] = frame_gates
        return predictions

    def count_parameters(self) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: TriFrameConfig) -> TriFrameModel:
        """Instantiate a model from a config object."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str | Path, config: TriFrameConfig | None = None) -> TriFrameModel:
        """Load a model from a saved state dict.

        Args:
            path: Path to the saved ``.pt`` state dict.
            config: Optional config; if ``None``, attempts to load
                    ``config.yaml`` from the same directory.

        Returns:
            Initialised ``TriFrameModel`` with loaded weights.
        """
        path = Path(path)
        if config is None:
            config_path = path.parent / "config.yaml"
            if config_path.exists():
                config = TriFrameConfig.from_yaml(config_path)
            else:
                raise ValueError(
                    f"No config provided and {config_path} not found."
                )
        model = cls(config)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model
