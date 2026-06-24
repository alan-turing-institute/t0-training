import torch

# https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/rope.py#L88

def precompute_freqs(d_head: int, max_seq_len: int, theta: float = 500_000.0):
    # Inverse frequencies, one per pair: shape (d_head // 2,)
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / d_head))

    # Outer product with positions: shape (max_seq_len, d_head // 2)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    # Duplicate so shape is (max_seq_len, d_head) — matches x directly
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Expects B, T, n_heads, d_head as input shape
    # Split the head dim in half and rotate: (x1, x2) -> (-x2, x1)
    # This uses half-split pairing: pair i = (dim i, dim i + d_head//2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   (B, T, n_heads, d_head)
    # cos/sin: (max_seq_len, d_head) -> slice to (T, d_head), broadcast over B and n_heads
    T = x.shape[1]
    cos = cos[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head)
    sin = sin[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head)
    return (x * cos + _rotate_half(x) * sin).to(x.dtype)
