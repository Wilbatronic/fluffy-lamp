#!/usr/bin/env python3
"""CLI script for training TriFrame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.data import FASTAReadDataset
from triframe.training import TriFrameTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TriFrame model on annotated DNA reads."
    )

    # Data arguments
    parser.add_argument(
        "--train-data", required=True, help="Path to training FASTA file"
    )
    parser.add_argument(
        "--train-labels", required=True, help="Path to training labels TSV"
    )
    parser.add_argument(
        "--val-data", required=True, help="Path to validation FASTA file"
    )
    parser.add_argument(
        "--val-labels", required=True, help="Path to validation labels TSV"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (default: small preset)",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large"],
        default="small",
        help="Model size preset (ignored if --config provided)",
    )

    # Training configuration
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.1, help="Warmup fraction"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument(
        "--save-every-n-epochs", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--eval-every-n-steps", type=int, default=500, help="Validate every N steps"
    )

    # Loss weights
    parser.add_argument(
        "--coding-weight", type=float, default=1.0, help="Coding loss weight"
    )
    parser.add_argument(
        "--frame-weight", type=float, default=1.0, help="Frame loss weight"
    )
    parser.add_argument(
        "--ec-weight", type=float, default=0.5, help="EC loss weight"
    )
    parser.add_argument(
        "--kegg-weight", type=float, default=1.0, help="KEGG loss weight"
    )
    parser.add_argument(
        "--cog-weight", type=float, default=0.5, help="COG loss weight"
    )

    # Other arguments
    parser.add_argument(
        "--device", default="auto", help="Device (auto/cuda/mps/cpu)"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load model config
    if args.config:
        print(f"Loading model config from {args.config}")
        model_config = TriFrameConfig.from_yaml(args.config)
    else:
        print(f"Using {args.model_size} model preset")
        if args.model_size == "small":
            model_config = TriFrameConfig.small()
        elif args.model_size == "base":
            model_config = TriFrameConfig.base()
        else:
            model_config = TriFrameConfig.large()

    # Create model
    print("Initializing model...")
    model = TriFrameModel(model_config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = FASTAReadDataset(
        fasta_path=args.train_data,
        label_path=args.train_labels,
        max_read_length=model_config.max_read_length,
    )
    val_dataset = FASTAReadDataset(
        fasta_path=args.val_data,
        label_path=args.val_labels,
        max_read_length=model_config.max_read_length,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.epochs,
        warmup_fraction=args.warmup_fraction,
        coding_loss_weight=args.coding_weight,
        frame_loss_weight=args.frame_weight,
        ec_loss_weight=args.ec_weight,
        kegg_loss_weight=args.kegg_weight,
        cog_loss_weight=args.cog_weight,
        output_dir=args.output_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        eval_every_n_steps=args.eval_every_n_steps,
        device=args.device,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    # Create trainer
    trainer = TriFrameTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
    )

    # Resume if requested
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    final_metrics = trainer.train()

    print("\nFinal validation metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
