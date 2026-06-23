import torch
import torch.nn as nn

from .config import TransformerConfig
from .norm import RMSNorm
from .rope import apply_rotary


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # KV repeat factor for GQA

        self.wq = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head)

        # QK norm (applied per-head before RoPE)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Rotary embeddings
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # GQA: repeat K and V heads to match Q head count
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # FlashAttention: expects (B, T, n_heads, d_head), returns same shape
        from flash_attn import flash_attn_func  # noqa: PLC0415
        out = flash_attn_func(q, k, v, causal=True)

        # Merge heads and project out
        out = out.reshape(B, T, self.n_heads * self.d_head)
        return self.wo(out)
