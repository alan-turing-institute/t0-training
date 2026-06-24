import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import TransformerConfig
from .norm import RMSNorm
from .rope import precompute_freqs

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/transformer/model.py

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding and LM head weights — they share the same matrix
        self.lm_head.weight = self.embedding.weight

        # Precompute RoPE tables once; registered as buffers so they move with the model
        cos, sin = precompute_freqs(config.d_model // config.n_heads, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T) integer token IDs
        x = self.embedding(tokens)              # (B, T, d_model)

        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        x = self.norm(x)                        # final norm before LM head
        return self.lm_head(x)                  # (B, T, vocab_size)
