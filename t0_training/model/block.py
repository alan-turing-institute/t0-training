import torch
import torch.nn as nn

from .attention import Attention
from .config import TransformerConfig
from .ffn import FFN
from .norm import RMSNorm

# https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/transformer/block.py

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = Attention(config, window_size=config.window_size_for_layer(layer_idx))
        self.ffn = FFN(config)
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Reordered-norm: norm is applied to the sublayer output, inside the residual branch

        # attention -> norm -> ffn -> norm -> residual
        x = x + self.attn_norm(self.attn(x, cos, sin))
        x = x + self.ffn_norm(self.ffn(x))
        return x
