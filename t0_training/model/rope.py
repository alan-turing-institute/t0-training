import math

import torch

# https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/rope.py#L88

def precompute_freqs(d_head: int, max_seq_len: int, theta: float = 500_000.0):
    # Inverse frequencies, one per pair: shape (d_head // 2,)
    i = torch.arange(0, d_head, 2, dtype=torch.float32) # 0, 2, 4, .., d_head-2
    inv_freq = 1.0 / (theta ** (i / d_head)) # radians per token

    # Outer product with positions: shape (max_seq_len, d_head // 2)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq) # raidians

    # Duplicate so shape is (max_seq_len, d_head) — matches x directly
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


# YaRN (https://arxiv.org/abs/2309.00071). Matches olmo_core's
# YaRNRoPEScalingConfig.compute_scaled_inv_freq exactly:
# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/rope.py


def _yarn_find_correction_dim(
    num_rotations: float, d_head: int, theta: float, original_max_position_embeddings: int,
) -> float:
    """Finds the dimension at which the frequency correction should be applied for YaRN scaling."""
    # Wavelength needed to complete `num_rotations` rotations during the original training
    wavelength = original_max_position_embeddings / num_rotations # tokens per 1 full rotation = tokens per 2*pi radians

    # Need to find pair index j for this wavelength:
    # inv_freq = 1.0 / (theta ** (i / d_head)) 
    # wavelength = 2*pi / inv_freq = 2*pi * theta**(i/d_head)
    # Rearrange to get i = d_head * log(wavelength / (2*pi)) / log(theta)
    i = d_head * math.log(wavelength / (2 * math.pi)) / math.log(theta)
    # j = i/2 (i is the dimension index, j is the pair index)
    return i / 2


def _yarn_find_correction_range(
    beta_fast: float, beta_slow: float, d_head: int, theta: float,
    original_max_position_embeddings: int,
) -> tuple[int, int]:
    """Returns the low and high pair indices over which ramp should be applied."""
    half_dim = d_head // 2 # no of pairs

    low = math.floor(_yarn_find_correction_dim(beta_fast, d_head, theta, original_max_position_embeddings)) # beta_fast = leave pairs which rotate at least this many times unscaled
    high = math.ceil(_yarn_find_correction_dim(beta_slow, d_head, theta, original_max_position_embeddings)) # beta_slow = pairs which rotate fewer than this many times are fully scaled by `factor`
    return max(low, 0), min(high, half_dim - 1)


def _yarn_linear_ramp(low: float, high: float, half_dim: int) -> torch.Tensor:
    """Returns a linear ramp from 0 to 1 between low and high, clamped to [0, 1]. 0 region is unscaled (extrapolation), 1 region is fully scaled (interpolation)."""
    span = max(high - low, 1e-3)  # 1e-3 avoid division by zero
    idx = torch.arange(half_dim, dtype=torch.float32)

    # (idx - low)/span <= 0 for indices below the ramp region
    # 0 <= (idx - low)/span <= 1 for indices within the ramp region
    # (idx - low)/span >= 1 for indices above the ramp region
    return ((idx - low) / span).clamp_(0, 1)


def precompute_freqs_yarn(
    d_head: int, max_seq_len: int, theta: float,
    factor: float, beta_fast: float, beta_slow: float,
    original_max_position_embeddings: int, attention_factor: float,
):
    """Precompute YaRN-scaled RoPE cos/sin tables.

    Same shape and purpose as `precompute_freqs`, but blends each frequency
    pair between its unscaled ("extrapolation") and `factor`-divided
    ("interpolation") version depending on how many full rotations it
    completed within the original training context, then rescales the
    result by `attention_factor`. 
    
    See https://arxiv.org/abs/2309.00071.

    Parameters
    ----------
    d_head : int
        Dimension of each attention head
    max_seq_len : int
        Max sequence length
    theta : float
        RoPE base frequency
    factor : float
        YaRN scaling factor (e.g. 8 if going from 8192 to 65536 context length)
    beta_fast : float
        Rotation-count threshold marking the high-frequency (short-wavelength)
        edge of the ramp - pairs that completed at least this many full
        rotations during training are left unscaled
    beta_slow : float
        Rotation-count threshold marking the low-frequency (long-wavelength)
        edge of the ramp - pairs that completed fewer than this many full
        rotations during training are fully scaled by `factor`
    original_max_position_embeddings : int
        Sequence length the base model was trained with originally
    attention_factor : float
        Constant multiplier applied to the output cos/sin tables (not the
        frequencies) to rescale attention magnitude, compensating for the
        effect of the frequency scaling above
    """
    # Inverse frequencies, one per pair: shape (d_head // 2,)
    # same as `precompute_freqs` above
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    inv_freq_extrapolation = 1.0 / (theta ** (i / d_head))

    # scale frequencies by YaRN scaling factor (lower freq = higher wavelength)
    inv_freq_interpolation = inv_freq_extrapolation / factor

    # find low/high indices of the ramp region and apply ramp
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, d_head, theta, original_max_position_embeddings)
    ramp = _yarn_linear_ramp(low, high, d_head // 2)

    # ramp = 0 for unscaled pairs, ramp = 1 for fully scaled pairs.
    inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)

    # Outer product with positions: shape (max_seq_len, d_head // 2)
    # Same as `precompute_freqs` above
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    # attention_factor scales the rotation output, not the frequencies themselves
    return freqs.cos() * attention_factor, freqs.sin() * attention_factor


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
