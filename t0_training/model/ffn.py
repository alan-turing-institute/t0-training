import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/feed_forward.py

class FFN(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # gate/up/down are matrix multiplications.

        # gate/up take d_model in and outputs ffn_hidden dim (~8/3 * d_model for SwiGLU - i.e. upscale to larger dimension)
        # down takes ffn_hidden dim in and outputs d_model
        self.gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up   = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate -> silu -> multiply with up -> down
        
        # each value in up(x) is weighted between -0.28 and itself (output of silu(gate(x)))
        return self.down(F.silu(self.gate(x)) * self.up(x))
