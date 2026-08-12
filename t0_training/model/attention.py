import torch
import torch.nn as nn

from .config import TransformerConfig
from .norm import RMSNorm
from .rope import apply_rotary

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/attention/__init__.py

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, window_size: int | None = None):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # KV repeat factor for GQA
        self.window_size = window_size  # None = full attention; else causal window of this many tokens

        self.wq = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)

        if config.qk_norm:
            # OLMo3 (use_head_qk_norm=False in olmo-core): normalise Q/K over the
            # full projection width, so the RMS statistic is shared across heads
            # (Qwen3-style per-head norm would use RMSNorm(d_head) instead).
            self.q_norm = RMSNorm(config.n_heads * self.d_head)
            self.k_norm = RMSNorm(config.n_kv_heads * self.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape

        # Project to Q, K, V
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # QK norm (applied to the flat projection, before the head split — OLMo3 style)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_kv_heads, self.d_head)
        v = v.view(B, T, self.n_kv_heads, self.d_head)

        # Rotary embeddings
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # GQA: repeat K and V heads to match Q head count
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # FlashAttention: expects (B, T, n_heads, d_head), returns same shape.
        # window_size=(w-1, 0) restricts each query to the preceding w tokens
        # (causal sliding window); (-1, -1) is full (global) causal attention.
        window_size_tuple = (self.window_size - 1, 0) if self.window_size is not None else (-1, -1)
        from flash_attn import flash_attn_func  # noqa: PLC0415
        out = flash_attn_func(q, k, v, causal=True, window_size=window_size_tuple)

        # Merge heads and project out
        out = out.reshape(B, T, self.n_heads * self.d_head)
        return self.wo(out)
