import math

import torch.optim


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup.

    Linear warmup from 0 → max_lr over warmup_steps, then cosine decay to
    min_lr over the remaining steps. Held at min_lr after max_steps.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
