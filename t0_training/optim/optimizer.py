import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW with weight decay disabled for 1D parameters (norms, biases).

    2D+ parameters (weight matrices) get weight_decay; 1D parameters
    (RMSNorm scale vectors, bias terms) get weight_decay=0.0.
    """
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if p.ndim < 2 and p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )
