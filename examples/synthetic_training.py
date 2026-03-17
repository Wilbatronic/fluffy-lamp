#!/usr/bin/env python3
"""Full pipeline example: synthetic data generation and training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and print status."""
    print(f"\n{'='*50}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False

    return True


def main() -> int:
    """Run the full synthetic training pipeline."""
    print("TriFrame Synthetic Training Example")
    print("=" * 50)

    # Directories
    data_dir = Path("data/synthetic_example")
    output_dir = Path("outputs/synthetic_example")

    # Step 1: Generate synthetic data
    if not run_command(
        [
            sys.executable, "scripts/generate_synthetic_data.py",
            "--output-dir", str(data_dir),
            "--n-sequences", "1000",
            "--min-length", "150",
            "--max-length", "300",
            "--noncoding-fraction", "0.3",
            "--seed", "42",
        ],
        "Generate synthetic training data",
    ):
        return 1

    # Step 2: Train for 2 epochs (quick demo)
    if not run_command(
        [
            sys.executable, "scripts/train.py",
            "--train-data", str(data_dir / "train.fasta"),
            "--train-labels", str(data_dir / "train_labels.tsv"),
            "--val-data", str(data_dir / "val.fasta"),
            "--val-labels", str(data_dir / "val_labels.tsv"),
            "--output-dir", str(output_dir),
            "--model-size", "small",
            "--epochs", "2",
            "--batch-size", "32",
            "--lr", "1e-4",
            "--save-every-n-epochs", "1",
            "--eval-every-n-steps", "100",
            "--no-amp",  # Disable AMP for CPU compatibility
        ],
        "Train TriFrame model (2 epochs)",
    ):
        return 1

    # Step 3: Evaluate on test set
    if not run_command(
        [
            sys.executable, "scripts/evaluate.py",
            "--checkpoint", str(output_dir / "best_model.pt"),
            "--test-data", str(data_dir / "test.fasta"),
            "--test-labels", str(data_dir / "test_labels.tsv"),
            "--batch-size", "32",
        ],
        "Evaluate on test set",
    ):
        return 1

    # Step 4: Run inference on a few samples
    if not run_command(
        [
            sys.executable, "scripts/predict.py",
            "--checkpoint", str(output_dir / "best_model.pt"),
            "--input", str(data_dir / "test.fasta"),
            "--output", str(output_dir / "predictions.tsv"),
            "--batch-size", "32",
        ],
        "Run inference on test samples",
    ):
        return 1

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print(f"Data: {data_dir}")
    print(f"Model: {output_dir}")
    print(f"Predictions: {output_dir / 'predictions.tsv'}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
