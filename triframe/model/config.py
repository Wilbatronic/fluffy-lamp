"""TriFrame model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TriFrameConfig:
    """Configuration for the TriFrame model.

    Attributes are grouped by sub-module. Use factory classmethods ``small()``,
    ``base()``, and ``large()`` for pre-defined parameter budgets, or load from
    YAML with ``from_yaml(path)``.
    """

    # Nucleotide encoder
    d_nucleotide: int = 384
    n_nuc_layers: int = 8
    nuc_kernel_size: int = 7
    max_read_length: int = 2048

    # Frame encoder
    d_frame: int = 512
    n_frame_layers: int = 6
    n_frame_heads: int = 8
    d_ff_ratio: int = 4
    max_aa_length: int = 512

    # Frame attention
    n_frame_attn_layers: int = 3

    # Cross-resolution fusion
    n_fusion_layers: int = 3

    # Prediction heads
    n_ec_level1: int = 7
    n_ec_level2: int = 68
    n_ec_level3: int = 264
    n_ec_level4: int = 5000
    n_kegg_orthologs: int = 10000
    n_cog_categories: int = 26

    # General
    dropout: float = 0.1
    num_frames: int = 6

    # ------------------------------------------------------------------ IO ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> TriFrameConfig:
        """Load a configuration from a YAML file."""
        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    # ----------------------------------------------------------- factories ---

    @classmethod
    def small(cls) -> TriFrameConfig:
        """~11M parameter configuration."""
        return cls(
            d_nucleotide=192,
            d_frame=256,
            n_nuc_layers=4,
            n_frame_layers=4,
            n_frame_heads=4,
            n_frame_attn_layers=2,
            n_fusion_layers=2,
            n_kegg_orthologs=5000,
        )

    @classmethod
    def base(cls) -> TriFrameConfig:
        """~56M parameter configuration (defaults)."""
        return cls()

    @classmethod
    def large(cls) -> TriFrameConfig:
        """~162M parameter configuration."""
        return cls(
            d_nucleotide=512,
            d_frame=768,
            n_nuc_layers=12,
            n_frame_layers=8,
            n_frame_heads=12,
            n_frame_attn_layers=4,
            n_fusion_layers=4,
            n_kegg_orthologs=25000,
        )
