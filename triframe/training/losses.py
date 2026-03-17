"""Multi-task loss functions for TriFrame."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriFrameLoss(nn.Module):
    """Multi-task loss for TriFrame predictions.

    Computes weighted sum of 5 loss components:
    1. Coding classification (CrossEntropyLoss) - ignore -1
    2. Frame classification (CrossEntropyLoss) - only for is_coding==1, ignore -1
    3. Hierarchical EC number classification (sum of 4 CrossEntropyLoss) - ignore ""
    4. KEGG ortholog multi-label (BCEWithLogitsLoss) - ignore ""
    5. COG category multi-label (BCEWithLogitsLoss) - ignore ""

    Args:
        coding_weight: Weight for coding loss component.
        frame_weight: Weight for frame loss component.
        ec_weight: Weight for EC number loss component.
        kegg_weight: Weight for KEGG loss component.
        cog_weight: Weight for COG loss component.
        n_kegg_orthologs: Number of KEGG orthologs (for target building).
        n_cog_categories: Number of COG categories (for target building).
    """

    def __init__(
        self,
        coding_weight: float = 1.0,
        frame_weight: float = 1.0,
        ec_weight: float = 0.5,
        kegg_weight: float = 1.0,
        cog_weight: float = 0.5,
        n_kegg_orthologs: int = 10000,
        n_cog_categories: int = 26,
    ):
        super().__init__()
        self.coding_weight = coding_weight
        self.frame_weight = frame_weight
        self.ec_weight = ec_weight
        self.kegg_weight = kegg_weight
        self.cog_weight = cog_weight
        self.n_kegg = n_kegg_orthologs
        self.n_cog = n_cog_categories

        self.coding_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.frame_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.ec_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.kegg_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.cog_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def _parse_ec_labels(self, ec_strings: list[str]) -> torch.Tensor:
        """Parse EC number strings into 4 integer levels.

        Args:
            ec_strings: List of EC strings like "2.7.1.1" or "".

        Returns:
            (batch, 4) tensor of integer levels, -1 for missing.
        """
        batch_size = len(ec_strings)
        levels = torch.full((batch_size, 4), -1, dtype=torch.long)

        for i, ec in enumerate(ec_strings):
            if not ec or ec.strip() == "":
                continue
            parts = ec.strip().split(".")
            for j, part in enumerate(parts[:4]):
                try:
                    levels[i, j] = int(part)
                except (ValueError, IndexError):
                    levels[i, j] = -1
        return levels

    def _parse_kegg_labels(
        self, kegg_strings: list[str], device: torch.device, n_kegg: int
    ) -> torch.Tensor:
        """Parse KEGG KO strings into multi-hot tensor.

        Args:
            kegg_strings: List of KEGG strings like "K00844,K00845" or "".
            device: Target device for the tensor.
            n_kegg: Number of KEGG orthologs (inferred from predictions).

        Returns:
            (batch, n_kegg) multi-hot float tensor.
        """
        batch_size = len(kegg_strings)
        targets = torch.zeros(batch_size, n_kegg, dtype=torch.float, device=device)

        for i, ko_str in enumerate(kegg_strings):
            if not ko_str or ko_str.strip() == "":
                continue
            kos = [k.strip() for k in ko_str.split(",") if k.strip()]
            for ko in kos:
                idx = self._kegg_to_index(ko, n_kegg)
                if 0 <= idx < n_kegg:
                    targets[i, idx] = 1.0
        return targets

    def _parse_cog_labels(
        self, cog_strings: list[str], device: torch.device, n_cog: int
    ) -> torch.Tensor:
        """Parse COG category strings into multi-hot tensor.

        Args:
            cog_strings: List of COG strings like "G,E" or "".
            device: Target device for the tensor.
            n_cog: Number of COG categories (inferred from predictions).

        Returns:
            (batch, n_cog) multi-hot float tensor.
        """
        batch_size = len(cog_strings)
        targets = torch.zeros(batch_size, n_cog, dtype=torch.float, device=device)

        for i, cog_str in enumerate(cog_strings):
            if not cog_str or cog_str.strip() == "":
                continue
            cats = [c.strip().upper() for c in cog_str.split(",") if c.strip()]
            for cat in cats:
                idx = self._cog_to_index(cat)
                if 0 <= idx < n_cog:
                    targets[i, idx] = 1.0
        return targets

    def _kegg_to_index(self, ko: str, n_kegg: int) -> int:
        """Convert KEGG KO ID (e.g., 'K00001') to index.

        Uses simple hash-based mapping for consistency.

        Args:
            ko: KEGG KO ID string.
            n_kegg: Number of KEGG orthologs (for modulo operation).
        """
        try:
            if ko.startswith("K") and len(ko) >= 6:
                num = int(ko[1:])
                return num % n_kegg
        except (ValueError, IndexError):
            pass
        return -1

    def _cog_to_index(self, cat: str) -> int:
        """Convert COG category letter to index.

        Standard COG categories: A-Z (26 letters).
        """
        if len(cat) == 1 and "A" <= cat <= "Z":
            return ord(cat) - ord("A")
        return -1

    def _kegg_index_to_id(self, idx: int) -> str:
        """Convert index back to KEGG KO ID format."""
        return f"K{idx:05d}"

    def _cog_index_to_category(self, idx: int) -> str:
        """Convert index back to COG category letter."""
        return chr(ord("A") + idx)

    def forward(
        self, predictions: dict, labels: dict | None
    ) -> tuple[torch.Tensor, dict]:
        """Compute multi-task loss.

        Args:
            predictions: Dict from model forward pass with keys:
                - coding_logits: (batch, 2)
                - frame_logits: (batch, 7)
                - ec_logits: dict with 'level1', 'level2', 'level3', 'level4'
                - kegg_logits: (batch, n_kegg)
                - cog_logits: (batch, n_cog)
            labels: Dict with keys:
                - is_coding: (batch,) 0/1/-1
                - reading_frame: (batch,) 0-5 or -1
                - ec_number: list[str]
                - kegg_ko: list[str]
                - cog_category: list[str]
                Or None if no labels available.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        device = predictions["coding_logits"].device
        batch_size = predictions["coding_logits"].size(0)

        losses = {
            "coding": torch.tensor(0.0, device=device),
            "frame": torch.tensor(0.0, device=device),
            "ec": torch.tensor(0.0, device=device),
            "kegg": torch.tensor(0.0, device=device),
            "cog": torch.tensor(0.0, device=device),
        }

        if labels is None:
            total = sum(losses.values())
            return total, losses

        # 1. Coding loss
        is_coding = labels["is_coding"].to(device)
        has_coding_label = (is_coding >= 0).any()
        if has_coding_label:
            losses["coding"] = self.coding_criterion(
                predictions["coding_logits"], is_coding
            )

        # 2. Frame loss - only for coding samples
        reading_frame = labels["reading_frame"].to(device)
        is_coding_mask = is_coding == 1
        has_frame_label = (reading_frame >= 0) & is_coding_mask
        if has_frame_label.any():
            frame_logits = predictions["frame_logits"][has_frame_label]
            frame_targets = reading_frame[has_frame_label]
            losses["frame"] = self.frame_criterion(frame_logits, frame_targets)

        # 3. EC loss - hierarchical 4 levels
        ec_levels = self._parse_ec_labels(labels["ec_number"]).to(device)
        has_ec = (ec_levels >= 0).any(dim=1)
        if has_ec.any():
            ec_loss_sum = torch.tensor(0.0, device=device)
            ec_count = 0
            for level_idx, level_key in enumerate(["level1", "level2", "level3", "level4"]):
                level_targets = ec_levels[has_ec, level_idx]
                level_logits = predictions["ec_logits"][level_key][has_ec]
                if (level_targets >= 0).any():
                    ec_loss_sum = ec_loss_sum + self.ec_criterion(
                        level_logits, level_targets
                    )
                    ec_count += 1
            if ec_count > 0:
                losses["ec"] = ec_loss_sum / ec_count

        # 4. KEGG loss - multi-label BCE
        n_kegg = predictions["kegg_logits"].shape[-1]
        kegg_targets = self._parse_kegg_labels(labels["kegg_ko"], device, n_kegg)
        has_kegg = kegg_targets.sum(dim=1) > 0
        if has_kegg.any():
            kegg_logits_masked = predictions["kegg_logits"][has_kegg]
            kegg_targets_masked = kegg_targets[has_kegg]
            losses["kegg"] = self.kegg_criterion(kegg_logits_masked, kegg_targets_masked)

        # 5. COG loss - multi-label BCE
        n_cog = predictions["cog_logits"].shape[-1]
        cog_targets = self._parse_cog_labels(labels["cog_category"], device, n_cog)
        has_cog = cog_targets.sum(dim=1) > 0
        if has_cog.any():
            cog_logits_masked = predictions["cog_logits"][has_cog]
            cog_targets_masked = cog_targets[has_cog]
            losses["cog"] = self.cog_criterion(cog_logits_masked, cog_targets_masked)

        total = (
            self.coding_weight * losses["coding"]
            + self.frame_weight * losses["frame"]
            + self.ec_weight * losses["ec"]
            + self.kegg_weight * losses["kegg"]
            + self.cog_weight * losses["cog"]
        )

        return total, losses
