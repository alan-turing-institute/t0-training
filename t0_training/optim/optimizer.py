import torch.nn as nn
from olmo_core.optim import SkipStepAdamW


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> SkipStepAdamW:
    """SkipStepAdamW with weight decay disabled for 1D parameters (norms, biases).

    2D+ parameters (weight matrices) get weight_decay; 1D parameters
    (RMSNorm scale vectors, bias terms) get weight_decay=0.0. 

    SkipStepAdamW skips the update whenever the caller-supplied loss or grad norm for a step
    is a >6-sigma outlier over the trailing 128 steps. 
    The caller must set `.latest_loss` and `.latest_grad_norm` before each `.step()`; see Trainer.train().
    """
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if p.ndim < 2 and p.requires_grad]

    return SkipStepAdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )
