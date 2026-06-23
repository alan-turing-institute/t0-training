import torch
import torch.nn as nn

from .attention import Attention
from .config import TransformerConfig
from .ffn import FFN
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FFN(config)
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Reordered-norm: norm is applied to the sublayer output, inside the residual branch
        x = x + self.attn_norm(self.attn(x, cos, sin))
        x = x + self.ffn_norm(self.ffn(x))
        return x
