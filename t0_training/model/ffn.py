import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


class FFN(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up   = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
