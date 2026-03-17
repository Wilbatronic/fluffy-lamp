"""Learning rate schedulers for TriFrame."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    The learning rate schedule follows:
    1. Linear warmup from 0 to initial LR over `num_warmup_steps`
    2. Cosine decay from initial LR to `min_lr_ratio * initial LR` over remaining steps

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a ratio of initial LR (default 0.0).

    Returns:
        LambdaLR scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return decayed

    return LambdaLR(optimizer, lr_lambda)
