"""Inference API for TriFrame."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.data import DNATokenizer


class TriFramePredictor:
    """Predictor for TriFrame functional annotation.

    Handles tokenization, batching, model inference, and post-processing
    to produce human-readable predictions.
    """

    def __init__(self, model: TriFrameModel, device: str = "auto"):
        """Initialize predictor with a model.

        Args:
            model: Trained TriFrameModel instance.
            device: Device for inference ('auto', 'cuda', 'mps', 'cpu').
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = DNATokenizer()

        # Extract vocab sizes from model config
        config = model.config
        self.n_kegg = config.n_kegg_orthologs
        self.n_cog = config.n_cog_categories
        self.n_ec_level1 = config.n_ec_level1
        self.n_ec_level2 = config.n_ec_level2
        self.n_ec_level3 = config.n_ec_level3
        self.n_ec_level4 = config.n_ec_level4

    def _get_device(self, device_str: str) -> torch.device:
        """Determine the device to use."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "TriFramePredictor":
        """Load predictor from a checkpoint file.

        Args:
            path: Path to checkpoint .pt file.
            device: Device for inference.

        Returns:
            Initialized TriFramePredictor.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Load config and create model
        config = checkpoint["model_config"]
        model = TriFrameModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, device)

    def predict_reads(
        self, sequences: list[str], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """Predict functional annotations for a list of DNA sequences.

        Args:
            sequences: List of DNA strings.
            batch_size: Batch size for inference.

        Returns:
            List of prediction dicts, one per sequence.
        """
        results = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_results = self._predict_batch(batch_seqs)
            results.extend(batch_results)

        return results

    def _predict_batch(self, sequences: list[str]) -> list[dict[str, Any]]:
        """Predict a batch of sequences."""
        # Tokenize
        tokenized = [self.tokenizer.encode(seq) for seq in sequences]

        # Pad
        max_len = max(len(ids) for ids in tokenized)
        nucleotide_ids = torch.full((len(sequences), max_len), 4, dtype=torch.long)
        for i, ids in enumerate(tokenized):
            nucleotide_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)

        # Move to device
        nucleotide_ids = nucleotide_ids.to(self.device)
        lengths = lengths.to(self.device)

        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                predictions = self.model(nucleotide_ids, lengths)

        # Post-process
        results = []
        for i in range(len(sequences)):
            result = self._postprocess_single(predictions, i, sequences[i])
            results.append(result)

        return results

    def _postprocess_single(
        self, predictions: dict, idx: int, sequence: str
    ) -> dict[str, Any]:
        """Post-process predictions for a single sample."""
        # Coding prediction
        coding_probs = F.softmax(predictions["coding_logits"][idx], dim=-1)
        is_coding = coding_probs[1].item() > 0.5
        coding_confidence = coding_probs[1].item() if is_coding else coding_probs[0].item()

        # Frame prediction
        frame_probs = F.softmax(predictions["frame_logits"][idx], dim=-1)
        predicted_frame = frame_probs.argmax().item()
        # If frame 6 (index 6) has highest prob, it means "no coding frame"
        if predicted_frame == 6 or not is_coding:
            predicted_frame = -1

        # Frame confidences from gates
        frame_gates = predictions["frame_gates"][idx]
        frame_confidences = frame_gates.cpu().tolist()

        # EC number prediction (hierarchical)
        ec_probs = {
            level: F.softmax(predictions["ec_logits"][level][idx], dim=-1)
            for level in ["level1", "level2", "level3", "level4"]
        }

        ec_levels = [
            ec_probs["level1"].argmax().item(),
            ec_probs["level2"].argmax().item(),
            ec_probs["level3"].argmax().item(),
            ec_probs["level4"].argmax().item(),
        ]

        # Build EC string
        if is_coding and all(0 <= l for l in ec_levels):
            ec_number = ".".join(str(l) for l in ec_levels)
            ec_confidence = min(ec_probs[f"level{i+1}"][ec_levels[i]].item() for i in range(4))
        else:
            ec_number = None
            ec_confidence = 0.0

        # KEGG prediction (multi-label)
        kegg_probs = torch.sigmoid(predictions["kegg_logits"][idx])
        kegg_threshold = 0.5
        kegg_indices = (kegg_probs > kegg_threshold).nonzero(as_tuple=True)[0].cpu().tolist()

        kegg_orthologs = [self._kegg_index_to_id(idx) for idx in kegg_indices]
        kegg_scores = [kegg_probs[idx].item() for idx in kegg_indices]

        # Sort by score
        if kegg_scores:
            sorted_pairs = sorted(zip(kegg_orthologs, kegg_scores), key=lambda x: x[1], reverse=True)
            kegg_orthologs = [p[0] for p in sorted_pairs]
            kegg_scores = [p[1] for p in sorted_pairs]

        # COG prediction (multi-label)
        cog_probs = torch.sigmoid(predictions["cog_logits"][idx])
        cog_threshold = 0.5
        cog_indices = (cog_probs > cog_threshold).nonzero(as_tuple=True)[0].cpu().tolist()

        cog_categories = [self._cog_index_to_category(idx) for idx in cog_indices]
        cog_scores = [cog_probs[idx].item() for idx in cog_indices]

        # Sort by score
        if cog_scores:
            sorted_pairs = sorted(zip(cog_categories, cog_scores), key=lambda x: x[1], reverse=True)
            cog_categories = [p[0] for p in sorted_pairs]
            cog_scores = [p[1] for p in sorted_pairs]

        # Generate read ID
        read_id = f"read_{hash(sequence) & 0xFFFFFFFF:08x}"

        return {
            "read_id": read_id,
            "is_coding": is_coding,
            "coding_confidence": coding_confidence,
            "predicted_frame": predicted_frame,
            "frame_confidences": frame_confidences,
            "ec_number": ec_number,
            "ec_confidence": ec_confidence,
            "kegg_orthologs": kegg_orthologs,
            "kegg_scores": kegg_scores,
            "cog_categories": cog_categories,
            "cog_scores": cog_scores,
        }

    def _kegg_to_index(self, ko: str) -> int:
        """Convert KEGG KO ID to index."""
        try:
            if ko.startswith("K") and len(ko) >= 6:
                num = int(ko[1:])
                return num % self.n_kegg
        except (ValueError, IndexError):
            pass
        return -1

    def _kegg_index_to_id(self, idx: int) -> str:
        """Convert index to KEGG KO ID."""
        return f"K{idx:05d}"

    def _cog_to_index(self, cat: str) -> int:
        """Convert COG category to index."""
        if len(cat) == 1 and "A" <= cat <= "Z":
            return ord(cat) - ord("A")
        return -1

    def _cog_index_to_category(self, idx: int) -> str:
        """Convert index to COG category."""
        return chr(ord("A") + idx)

    def predict_fasta(
        self,
        fasta_path: str,
        batch_size: int = 32,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict annotations for sequences in a FASTA file.

        Args:
            fasta_path: Path to input FASTA file.
            batch_size: Batch size for inference.
            output_path: Optional path to write TSV output.

        Returns:
            List of prediction dicts.
        """
        # Read FASTA
        sequences = []
        read_ids = []

        with open(fasta_path) as f:
            current_id = None
            current_seq_parts = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    # Save previous sequence
                    if current_id is not None:
                        sequences.append("".join(current_seq_parts))
                        read_ids.append(current_id)
                    # Start new sequence
                    current_id = line[1:].split()[0]
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)

            # Save last sequence
            if current_id is not None:
                sequences.append("".join(current_seq_parts))
                read_ids.append(current_id)

        # Run predictions
        results = self.predict_reads(sequences, batch_size=batch_size)

        # Update read IDs from FASTA
        for i, read_id in enumerate(read_ids):
            results[i]["read_id"] = read_id

        # Write TSV if requested
        if output_path is not None:
            self._write_tsv(results, output_path)

        return results

    def _write_tsv(self, results: list[dict[str, Any]], output_path: str) -> None:
        """Write predictions to TSV file."""
        fieldnames = [
            "read_id",
            "is_coding",
            "coding_confidence",
            "predicted_frame",
            "frame_confidences",
            "ec_number",
            "ec_confidence",
            "kegg_orthologs",
            "kegg_scores",
            "cog_categories",
            "cog_scores",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for result in results:
                row = {
                    "read_id": result["read_id"],
                    "is_coding": int(result["is_coding"]),
                    "coding_confidence": f"{result['coding_confidence']:.4f}",
                    "predicted_frame": result["predicted_frame"],
                    "frame_confidences": ",".join(f"{c:.4f}" for c in result["frame_confidences"]),
                    "ec_number": result["ec_number"] if result["ec_number"] else "",
                    "ec_confidence": f"{result['ec_confidence']:.4f}",
                    "kegg_orthologs": ",".join(result["kegg_orthologs"]),
                    "kegg_scores": ",".join(f"{s:.4f}" for s in result["kegg_scores"]),
                    "cog_categories": ",".join(result["cog_categories"]),
                    "cog_scores": ",".join(f"{s:.4f}" for s in result["cog_scores"]),
                }
                writer.writerow(row)

        print(f"Predictions written to: {output_path}")
