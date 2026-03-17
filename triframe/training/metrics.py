"""Evaluation metrics for TriFrame."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override


class TriFrameMetrics:
    """Accumulate and compute evaluation metrics for TriFrame.

    Tracks:
    - Coding accuracy (binary classification)
    - Frame accuracy (only for is_coding==1 samples)
    - EC hierarchical accuracy (4 levels)
    - KEGG F1 (macro), Precision@1, Precision@3, Precision@5
    - COG F1 (macro)
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        # Coding
        self.coding_correct = 0
        self.coding_total = 0

        # Frame
        self.frame_correct = 0
        self.frame_total = 0

        # EC levels
        self.ec_correct = [0, 0, 0, 0]
        self.ec_total = [0, 0, 0, 0]

        # KEGG - for F1 and precision@k
        self.kegg_tp = 0
        self.kegg_fp = 0
        self.kegg_fn = 0
        self.kegg_prec_at_1_correct = 0
        self.kegg_prec_at_3_correct = 0
        self.kegg_prec_at_5_correct = 0
        self.kegg_total = 0

        # COG - for F1
        self.cog_tp = 0
        self.cog_fp = 0
        self.cog_fn = 0
        self.cog_total = 0

    @override
    def update(self, predictions: dict, labels: dict | None) -> None:
        """Update metrics with a batch of predictions and labels.

        Args:
            predictions: Dict from model with logits and gates.
            labels: Dict with ground truth labels, or None.
        """
        if labels is None:
            return

        device = predictions["coding_logits"].device

        # Coding accuracy
        is_coding = labels["is_coding"].to(device)
        coding_mask = is_coding >= 0
        if coding_mask.any():
            coding_preds = predictions["coding_logits"][coding_mask].argmax(dim=-1)
            coding_targets = is_coding[coding_mask]
            self.coding_correct += (coding_preds == coding_targets).sum().item()
            self.coding_total += coding_mask.sum().item()

        # Frame accuracy (only for coding samples with valid frame labels)
        reading_frame = labels["reading_frame"].to(device)
        frame_mask = (is_coding == 1) & (reading_frame >= 0)
        if frame_mask.any():
            frame_preds = predictions["frame_logits"][frame_mask].argmax(dim=-1)
            frame_targets = reading_frame[frame_mask]
            self.frame_correct += (frame_preds == frame_targets).sum().item()
            self.frame_total += frame_mask.sum().item()

        # EC accuracy (4 hierarchical levels)
        for level_idx, level_key in enumerate(["level1", "level2", "level3", "level4"]):
            level_logits = predictions["ec_logits"][level_key]
            # Parse EC labels for this batch
            ec_parsed = self._parse_ec_labels(labels["ec_number"])
            ec_targets = ec_parsed[:, level_idx].to(device)
            ec_mask = ec_targets >= 0
            if ec_mask.any():
                level_preds = level_logits[ec_mask].argmax(dim=-1)
                level_targets = ec_targets[ec_mask]
                self.ec_correct[level_idx] += (level_preds == level_targets).sum().item()
                self.ec_total[level_idx] += ec_mask.sum().item()

        # KEGG metrics
        kegg_logits = predictions["kegg_logits"]
        kegg_targets = self._parse_kegg_labels(labels["kegg_ko"], kegg_logits.size(-1), device)
        has_kegg = kegg_targets.sum(dim=1) > 0
        if has_kegg.any():
            kegg_preds = torch.sigmoid(kegg_logits[has_kegg])
            kegg_targets_masked = kegg_targets[has_kegg]
            self._update_kegg_metrics(kegg_preds, kegg_targets_masked)
            self.kegg_total += has_kegg.sum().item()

        # COG metrics
        cog_logits = predictions["cog_logits"]
        cog_targets = self._parse_cog_labels(labels["cog_category"], cog_logits.size(-1), device)
        has_cog = cog_targets.sum(dim=1) > 0
        if has_cog.any():
            cog_preds = torch.sigmoid(cog_logits[has_cog])
            cog_targets_masked = cog_targets[has_cog]
            self._update_cog_metrics(cog_preds, cog_targets_masked)
            self.cog_total += has_cog.sum().item()

    def _parse_ec_labels(self, ec_strings: list[str]) -> torch.Tensor:
        """Parse EC number strings into 4 integer levels."""
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
        self, kegg_strings: list[str], n_kegg: int, device: torch.device
    ) -> torch.Tensor:
        """Parse KEGG KO strings into multi-hot tensor."""
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
        self, cog_strings: list[str], n_cog: int, device: torch.device
    ) -> torch.Tensor:
        """Parse COG category strings into multi-hot tensor."""
        batch_size = len(cog_strings)
        targets = torch.zeros(batch_size, n_cog, dtype=torch.float, device=device)

        for i, cog_str in enumerate(cog_strings):
            if not cog_str or cog_str.strip() == "":
                continue
            cats = [c.strip().upper() for c in cog_str.split(",") if c.strip()]
            for cat in cats:
                idx = self._cog_to_index(cat, n_cog)
                if 0 <= idx < n_cog:
                    targets[i, idx] = 1.0
        return targets

    def _kegg_to_index(self, ko: str, n_kegg: int) -> int:
        """Convert KEGG KO ID to index."""
        try:
            if ko.startswith("K") and len(ko) >= 6:
                num = int(ko[1:])
                return num % n_kegg
        except (ValueError, IndexError):
            pass
        return -1

    def _cog_to_index(self, cat: str, n_cog: int) -> int:
        """Convert COG category to index."""
        if len(cat) == 1 and "A" <= cat <= "Z":
            return ord(cat) - ord("A")
        return -1

    def _update_kegg_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update KEGG metrics with a batch."""
        # Binary predictions (threshold 0.5) for F1
        binary_preds = (preds > 0.5).float()

        # TP, FP, FN for macro F1
        tp = (binary_preds * targets).sum()
        fp = (binary_preds * (1 - targets)).sum()
        fn = ((1 - binary_preds) * targets).sum()

        self.kegg_tp += tp.item()
        self.kegg_fp += fp.item()
        self.kegg_fn += fn.item()

        # Precision@k
        for i in range(preds.size(0)):
            pred_scores = preds[i]
            target_labels = targets[i]

            # Top-k indices
            _, top_indices = torch.topk(pred_scores, k=min(5, pred_scores.size(0)))

            # Precision@1
            if target_labels[top_indices[0]] > 0.5:
                self.kegg_prec_at_1_correct += 1

            # Precision@3
            if target_labels[top_indices[:3]].sum() > 0:
                self.kegg_prec_at_3_correct += 1

            # Precision@5
            if target_labels[top_indices].sum() > 0:
                self.kegg_prec_at_5_correct += 1

    def _update_cog_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update COG metrics with a batch."""
        binary_preds = (preds > 0.5).float()

        tp = (binary_preds * targets).sum()
        fp = (binary_preds * (1 - targets)).sum()
        fn = ((1 - binary_preds) * targets).sum()

        self.cog_tp += tp.item()
        self.cog_fp += fp.item()
        self.cog_fn += fn.item()

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dict with metric names and float values.
        """
        metrics = {}

        # Coding accuracy
        if self.coding_total > 0:
            metrics["coding_accuracy"] = self.coding_correct / self.coding_total
        else:
            metrics["coding_accuracy"] = 0.0

        # Frame accuracy
        if self.frame_total > 0:
            metrics["frame_accuracy"] = self.frame_correct / self.frame_total
        else:
            metrics["frame_accuracy"] = 0.0

        # EC accuracies
        for level_idx in range(4):
            key = f"ec_level{level_idx + 1}_accuracy"
            if self.ec_total[level_idx] > 0:
                metrics[key] = self.ec_correct[level_idx] / self.ec_total[level_idx]
            else:
                metrics[key] = 0.0

        # KEGG F1 (macro)
        if self.kegg_total > 0:
            precision = self.kegg_tp / (self.kegg_tp + self.kegg_fp + 1e-8)
            recall = self.kegg_tp / (self.kegg_tp + self.kegg_fn + 1e-8)
            if precision + recall > 0:
                metrics["kegg_f1"] = 2 * precision * recall / (precision + recall)
            else:
                metrics["kegg_f1"] = 0.0

            # Precision@k
            metrics["kegg_precision@1"] = self.kegg_prec_at_1_correct / self.kegg_total
            metrics["kegg_precision@3"] = self.kegg_prec_at_3_correct / self.kegg_total
            metrics["kegg_precision@5"] = self.kegg_prec_at_5_correct / self.kegg_total
        else:
            metrics["kegg_f1"] = 0.0
            metrics["kegg_precision@1"] = 0.0
            metrics["kegg_precision@3"] = 0.0
            metrics["kegg_precision@5"] = 0.0

        # COG F1 (macro)
        if self.cog_total > 0:
            precision = self.cog_tp / (self.cog_tp + self.cog_fp + 1e-8)
            recall = self.cog_tp / (self.cog_tp + self.cog_fn + 1e-8)
            if precision + recall > 0:
                metrics["cog_f1"] = 2 * precision * recall / (precision + recall)
            else:
                metrics["cog_f1"] = 0.0
        else:
            metrics["cog_f1"] = 0.0

        return metrics
