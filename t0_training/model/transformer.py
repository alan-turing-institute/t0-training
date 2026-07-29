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

        self.embedding = nn.Embedding(config.vocab_size, config.d_model) # shape [vocab_size, d_model]
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.n_layers)]
        ) 
        self.norm = RMSNorm(config.d_model)
        # Untied from the embedding: OLMo-core's olmo2/olmo3 factories build
        # lm_head.w_out as an independent matrix (tie_word_embeddings=False),
        # so we match that here rather than sharing embedding.weight.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE tables once; registered as buffers so they move with the model
        cos, sin = precompute_freqs(config.d_model // config.n_heads, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.init_weights()

    def init_weights(self, std: float = 0.02) -> None:
        # Matches OLMo-core's default InitMethod.normal: truncated normal,
        # std=0.02, applied to every embedding and linear layer.
        # https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/transformer/init.py
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        for block in self.blocks:
            for lin in (
                block.attn.wq, block.attn.wk, block.attn.wv, block.attn.wo,
                block.ffn.gate, block.ffn.up, block.ffn.down,
            ):
                nn.init.trunc_normal_(lin.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T) integer token IDs
        x = self.embedding(tokens)              # (B, T, d_model)

        for block in self.blocks:
            x = block(x, self.cos, self.sin)    # (B, T, d_model)

        x = self.norm(x)                        # final norm before LM head
        return self.lm_head(x)                  # now upsize to (B, T, vocab_size)
