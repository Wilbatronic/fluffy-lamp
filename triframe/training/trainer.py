"""Training loop for TriFrame."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.data import TriFrameCollator
from triframe.training.losses import TriFrameLoss
from triframe.training.metrics import TriFrameMetrics
from triframe.training.scheduler import get_cosine_schedule_with_warmup


@dataclass
class TrainingConfig:
    """Configuration for TriFrame training.

    Attributes:
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate before update.
        max_grad_norm: Max gradient norm for clipping.
        num_epochs: Total training epochs.
        warmup_fraction: Fraction of training for linear warmup.
        coding_loss_weight: Weight for coding loss component.
        frame_loss_weight: Weight for frame loss component.
        ec_loss_weight: Weight for EC loss component.
        kegg_loss_weight: Weight for KEGG loss component.
        cog_loss_weight: Weight for COG loss component.
        output_dir: Directory to save checkpoints and logs.
        save_every_n_epochs: Save checkpoint every N epochs.
        eval_every_n_steps: Run validation every N steps.
        device: Device to train on ('auto', 'cuda', 'mps', 'cpu').
        use_amp: Use automatic mixed precision.
        num_workers: DataLoader workers.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_epochs: int = 50
    warmup_fraction: float = 0.1
    coding_loss_weight: float = 1.0
    frame_loss_weight: float = 1.0
    ec_loss_weight: float = 0.5
    kegg_loss_weight: float = 1.0
    cog_loss_weight: float = 0.5
    output_dir: str = "outputs"
    save_every_n_epochs: int = 5
    eval_every_n_steps: int = 500
    device: str = "auto"
    use_amp: bool = True
    num_workers: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TriFrameTrainer:
    """Trainer for TriFrame model.

    Handles training loop with mixed precision, gradient accumulation,
    gradient clipping, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: TriFrameModel,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = self._get_device(config.device)
        self.model.to(self.device)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataLoaders
        self.collator = TriFrameCollator()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Get model config for loss
        model_config = model.config
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_fraction)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss and metrics
        self.loss_fn = TriFrameLoss(
            coding_weight=config.coding_loss_weight,
            frame_weight=config.frame_loss_weight,
            ec_weight=config.ec_loss_weight,
            kegg_weight=config.kegg_loss_weight,
            cog_weight=config.cog_loss_weight,
            n_kegg_orthologs=model_config.n_kegg_orthologs,
            n_cog_categories=model_config.n_cog_categories,
        ).to(self.device)

        self.train_metrics = TriFrameMetrics()
        self.val_metrics = TriFrameMetrics()

        # AMP scaler
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0

        # Logging
        self.log_file = self.output_dir / "training_log.jsonl"

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

    def train(self) -> dict[str, float]:
        """Run full training loop.

        Returns:
            Dict with final validation metrics.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Total steps: {len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch()

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Final evaluation
        final_metrics = self.evaluate()
        self.save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 3600:.2f} hours")
        print(f"Best KEGG F1: {self.best_metric:.4f}")

        return final_metrics

    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        epoch_loss = 0.0
        steps_in_epoch = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            nucleotide_ids = batch["nucleotide_ids"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            labels = batch["labels"]
            if labels is not None:
                labels = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in labels.items()
                }

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                predictions = self.model(nucleotide_ids, lengths)
                loss, loss_components = self.loss_fn(predictions, labels)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % 100 == 0:
                    throughput = (
                        self.config.batch_size
                        * self.config.gradient_accumulation_steps
                        * 100
                        / (time.time() - start_time)
                    )
                    lr = self.scheduler.get_last_lr()[0]
                    self._log_step(
                        step=self.global_step,
                        loss=loss.item() * self.config.gradient_accumulation_steps,
                        loss_components=loss_components,
                        lr=lr,
                        throughput=throughput,
                    )
                    start_time = time.time()

                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self.evaluate()
                    self.model.train()

                    # Save best model
                    kegg_f1 = val_metrics.get("kegg_f1", 0.0)
                    if kegg_f1 > self.best_metric:
                        self.best_metric = kegg_f1
                        self.save_checkpoint("best_model.pt")
                        print(f"  New best model! KEGG F1: {kegg_f1:.4f}")

            # Update metrics (on unscaled loss)
            with torch.no_grad():
                self.train_metrics.update(
                    {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()},
                    labels,
                )
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                steps_in_epoch += 1

        avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
        metrics = self.train_metrics.compute()
        metrics["loss"] = avg_loss

        print(f"Epoch {self.epoch + 1}/{self.config.num_epochs} - Train loss: {avg_loss:.4f}")

        return metrics

    def _log_step(
        self,
        step: int,
        loss: float,
        loss_components: dict,
        lr: float,
        throughput: float,
    ) -> None:
        """Log training step."""
        log_entry = {
            "step": step,
            "epoch": self.epoch + 1,
            "loss": loss,
            "loss_components": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_components.items()},
            "learning_rate": lr,
            "throughput": throughput,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"Step {step} | Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Throughput: {throughput:.1f} reads/sec"
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            nucleotide_ids = batch["nucleotide_ids"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            labels = batch["labels"]
            if labels is not None:
                labels = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in labels.items()
                }

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                predictions = self.model(nucleotide_ids, lengths)
                loss, _ = self.loss_fn(predictions, labels)

            self.val_metrics.update(predictions, labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = self.val_metrics.compute()
        metrics["loss"] = avg_loss

        print(f"  Validation - Loss: {avg_loss:.4f}, Coding Acc: {metrics['coding_accuracy']:.3f}, KEGG F1: {metrics['kegg_f1']:.3f}")

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "training_config": self.config.to_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]

        print(f"Loaded checkpoint from {path}")
        print(f"  Resumed at epoch {self.epoch + 1}, step {self.global_step}")
