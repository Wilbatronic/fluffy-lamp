"""Batch collation with padding for TriFrame datasets."""

from __future__ import annotations

import torch


class TriFrameCollator:
    """Collates variable-length DNA reads into padded batches.

    Pads nucleotide IDs with N (4) to the longest sequence in the batch.

    Returns:
        dict with:
            ``nucleotide_ids``: (batch, max_len) padded with 4 (N).
            ``lengths``:        (batch,) actual read lengths.
            ``labels``:         dict of label tensors if labels exist, else ``None``.
    """

    def __call__(self, batch: list[dict]) -> dict:
        lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
        max_len = int(lengths.max().item())

        nuc_ids = torch.full((len(batch), max_len), 4, dtype=torch.long)
        for i, sample in enumerate(batch):
            seq = sample["nucleotide_ids"]
            nuc_ids[i, : len(seq)] = seq

        result: dict = {
            "nucleotide_ids": nuc_ids,
            "lengths": lengths,
        }

        if "labels" in batch[0]:
            is_coding = torch.tensor(
                [s["labels"]["is_coding"] for s in batch], dtype=torch.long
            )
            reading_frame = torch.tensor(
                [s["labels"]["reading_frame"] for s in batch], dtype=torch.long
            )
            result["labels"] = {
                "is_coding": is_coding,
                "reading_frame": reading_frame,
                "ec_number": [s["labels"].get("ec_number", "") for s in batch],
                "kegg_ko": [s["labels"].get("kegg_ko", "") for s in batch],
                "cog_category": [s["labels"].get("cog_category", "") for s in batch],
            }
        else:
            result["labels"] = None

        return result
