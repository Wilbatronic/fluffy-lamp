#!/usr/bin/env python3
"""CLI script for evaluating TriFrame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.data import FASTAReadDataset, TriFrameCollator
from triframe.training import TriFrameLoss, TriFrameMetrics
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TriFrame model on test data."
    )

    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-data", required=True, help="Path to test FASTA file"
    )
    parser.add_argument(
        "--test-labels", required=True, help="Path to test labels TSV"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Evaluation batch size"
    )
    parser.add_argument(
        "--device", default="auto", help="Device (auto/cuda/mps/cpu)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional JSON output file for metrics"
    )

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Determine the device to use."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    """Run evaluation and return metrics."""
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create model
    config = checkpoint["model_config"]
    model = TriFrameModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Load test dataset
    print(f"Loading test data from {args.test_data}")
    test_dataset = FASTAReadDataset(
        fasta_path=args.test_data,
        label_path=args.test_labels,
        max_read_length=config.max_read_length,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloader
    collator = TriFrameCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Create loss and metrics
    loss_fn = TriFrameLoss(
        n_kegg_orthologs=config.n_kegg_orthologs,
        n_cog_categories=config.n_cog_categories,
    ).to(device)

    metrics = TriFrameMetrics()

    # Evaluation loop
    total_loss = 0.0
    num_batches = 0

    print("\nRunning evaluation...")
    for batch in test_loader:
        nucleotide_ids = batch["nucleotide_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"]
        if labels is not None:
            labels = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in labels.items()
            }

        # Forward pass
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(nucleotide_ids, lengths)
            loss, _ = loss_fn(predictions, labels)

        # Update metrics
        metrics.update(predictions, labels)
        total_loss += loss.item()
        num_batches += 1

    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    results = metrics.compute()
    results["loss"] = avg_loss

    return results


def main() -> int:
    args = parse_args()

    results = evaluate(args)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Loss:                  {results['loss']:.4f}")
    print(f"Coding Accuracy:       {results['coding_accuracy']:.4f}")
    print(f"Frame Accuracy:        {results['frame_accuracy']:.4f}")
    print(f"EC Level 1 Accuracy:   {results['ec_level1_accuracy']:.4f}")
    print(f"EC Level 2 Accuracy:   {results['ec_level2_accuracy']:.4f}")
    print(f"EC Level 3 Accuracy:   {results['ec_level3_accuracy']:.4f}")
    print(f"EC Level 4 Accuracy:   {results['ec_level4_accuracy']:.4f}")
    print(f"KEGG F1 (macro):       {results['kegg_f1']:.4f}")
    print(f"KEGG Precision@1:      {results['kegg_precision@1']:.4f}")
    print(f"KEGG Precision@3:      {results['kegg_precision@3']:.4f}")
    print(f"KEGG Precision@5:      {results['kegg_precision@5']:.4f}")
    print(f"COG F1 (macro):        {results['cog_f1']:.4f}")
    print("=" * 50)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
