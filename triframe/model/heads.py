"""Multi-task prediction heads for functional annotation."""

from __future__ import annotations

import torch
import torch.nn as nn


class TriFrameHeads(nn.Module):
    """Multi-task prediction heads for TriFrame.

    Each head is a simple linear projection from the fused representation to
    class logits.  EC numbers use a hierarchical layout (4 independent levels).
    KEGG and COG are multi-label (sigmoid at inference; logits returned here).

    Args:
        d_frame: Input feature dimension.
        n_ec_level1: EC top-level classes.
        n_ec_level2: EC second-level subclasses.
        n_ec_level3: EC third-level sub-subclasses.
        n_ec_level4: EC fourth-level serial numbers.
        n_kegg_orthologs: Number of KEGG orthologs (multi-label).
        n_cog_categories: Number of COG categories (multi-label).

    Input:
        x: (batch, d_frame) fused representation.

    Output:
        dict with keys:
            ``coding_logits``:  (batch, 2) coding vs non-coding.
            ``frame_logits``:   (batch, 7) which of 6 frames is correct, or none.
            ``ec_logits``:      dict with ``level1`` … ``level4`` tensors.
            ``kegg_logits``:    (batch, n_kegg_orthologs) multi-label logits.
            ``cog_logits``:     (batch, n_cog_categories) multi-label logits.
    """

    def __init__(
        self,
        d_frame: int = 512,
        n_ec_level1: int = 7,
        n_ec_level2: int = 68,
        n_ec_level3: int = 264,
        n_ec_level4: int = 5000,
        n_kegg_orthologs: int = 10000,
        n_cog_categories: int = 26,
    ):
        super().__init__()
        self.coding_head = nn.Linear(d_frame, 2)
        self.frame_head = nn.Linear(d_frame, 7)

        self.ec_head1 = nn.Linear(d_frame, n_ec_level1)
        self.ec_head2 = nn.Linear(d_frame, n_ec_level2)
        self.ec_head3 = nn.Linear(d_frame, n_ec_level3)
        self.ec_head4 = nn.Linear(d_frame, n_ec_level4)

        self.kegg_head = nn.Linear(d_frame, n_kegg_orthologs)
        self.cog_head = nn.Linear(d_frame, n_cog_categories)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        """Forward pass.

        Args:
            x: (batch, d_frame).

        Returns:
            Dictionary of prediction logits.
        """
        return {
            "coding_logits": self.coding_head(x),
            "frame_logits": self.frame_head(x),
            "ec_logits": {
                "level1": self.ec_head1(x),
                "level2": self.ec_head2(x),
                "level3": self.ec_head3(x),
                "level4": self.ec_head4(x),
            },
            "kegg_logits": self.kegg_head(x),
            "cog_logits": self.cog_head(x),
        }
