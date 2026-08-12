"""Tests for model modules (norm, rope, ffn, attention, block, transformer).

Where possible, numerical outputs are validated against the equivalent
olmo-core reference implementation using matching weights.
"""

import pytest
import torch
import torch.nn as nn

from t0_training.model.config import TransformerConfig
from t0_training.model.ffn import FFN
from t0_training.model.norm import RMSNorm
from t0_training.model.rope import apply_rotary, precompute_freqs

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# ---------------------------------------------------------------------------
# Tiny config used across all tests — fits comfortably on CPU
# ---------------------------------------------------------------------------

_CFG = TransformerConfig(
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    ffn_hidden_dim=128,
    vocab_size=512,
    max_seq_len=32,
    qk_norm=True,
)

_D_HEAD = _CFG.d_model // _CFG.n_heads  # 16


# ===========================================================================
# RMSNorm  (model/norm.py)
# ref: https://github.com/allenai/OLMo-core/blob/fa6c5014/src/olmo_core/nn/layer_norm.py#L210
# ===========================================================================


def test_rmsnorm_output_shape():
    norm = RMSNorm(_CFG.d_model)
    x = torch.randn(2, 10, _CFG.d_model)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_input_returns_weight():
    """With all-ones input the norm divides by ~1, so output ≈ weight."""
    norm = RMSNorm(4)
    nn.init.constant_(norm.weight, 2.0)
    x = torch.ones(1, 1, 4)
    out = norm(x)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5)


def test_rmsnorm_works_on_higher_rank_tensor():
    """RMSNorm should work on higher-rank tensors, e.g. (B, T, n_heads, d_head)."""
    norm = RMSNorm(_D_HEAD)
    x = torch.randn(2, 8, _CFG.n_heads, _D_HEAD)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_matches_olmo_core():
    """Numerical match against olmo_core RMSNorm (no bias, no full-precision upcast)."""
    from olmo_core.nn.layer_norm import RMSNorm as OlmoRMSNorm

    d = 64
    our_norm = RMSNorm(d, eps=1e-5)
    olmo_norm = OlmoRMSNorm(size=d, eps=1e-5, elementwise_affine=True, bias=False, full_precision=False)
    # Share weights
    with torch.no_grad():
        olmo_norm.weight.copy_(our_norm.weight)

    x = torch.randn(2, 10, d)
    our_out = our_norm(x)
    olmo_out = olmo_norm(x)
    assert torch.allclose(our_out, olmo_out, atol=1e-5), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


def test_rmsnorm_no_bias():
    """RMSNorm should have no bias parameter."""
    norm = RMSNorm(16)
    param_names = {n for n, _ in norm.named_parameters()}
    assert "bias" not in param_names
    assert "weight" in param_names


# ===========================================================================
# RoPE  (model/rope.py)
# ref: https://github.com/allenai/OLMo-core/blob/fa6c5014/src/olmo_core/nn/rope.py#L88
# ===========================================================================


def test_precompute_freqs_shape():
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    assert cos.shape == (_CFG.max_seq_len, _D_HEAD)
    assert sin.shape == (_CFG.max_seq_len, _D_HEAD)


def test_precompute_freqs_cos_sin_in_range():
    cos, sin = precompute_freqs(_D_HEAD, 64, 10_000.0)
    assert cos.abs().max() <= 1.0 + 1e-6
    assert sin.abs().max() <= 1.0 + 1e-6


def test_precompute_freqs_matches_olmo_core():
    """Frequency tables should match olmo_core's RotaryEmbedding computation."""
    from olmo_core.nn.rope import RotaryEmbedding

    d_head = 16
    seq_len = 32
    theta = 10_000

    our_cos, our_sin = precompute_freqs(d_head, seq_len, float(theta))

    olmo_rope = RotaryEmbedding(head_size=d_head, theta=theta, full_precision=True)
    olmo_sin, olmo_cos = olmo_rope._get_rotary_embedding(seq_len, torch.device("cpu"))

    assert torch.allclose(our_cos, olmo_cos.float(), atol=1e-5), (
        f"cos max diff: {(our_cos - olmo_cos.float()).abs().max().item()}"
    )
    assert torch.allclose(our_sin, olmo_sin.float(), atol=1e-5), (
        f"sin max diff: {(our_sin - olmo_sin.float()).abs().max().item()}"
    )


def test_apply_rotary_shape():
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    x = torch.randn(2, 8, _CFG.n_heads, _D_HEAD)
    out = apply_rotary(x, cos, sin)
    assert out.shape == x.shape


def test_apply_rotary_dtype_preserved():
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    x = torch.randn(2, 8, _CFG.n_heads, _D_HEAD, dtype=torch.bfloat16)
    out = apply_rotary(x, cos, sin)
    assert out.dtype == torch.bfloat16


def test_apply_rotary_matches_olmo_core():
    """apply_rotary should produce the same result as olmo_core after transposing head dims."""
    from olmo_core.nn.rope import RotaryEmbedding

    d_head = 16
    seq_len = 8
    n_heads = 4
    B = 2
    theta = 10_000

    our_cos, our_sin = precompute_freqs(d_head, seq_len, float(theta))

    olmo_rope = RotaryEmbedding(head_size=d_head, theta=theta, full_precision=True)
    olmo_sin, olmo_cos = olmo_rope._get_rotary_embedding(seq_len, torch.device("cpu"))

    x = torch.randn(B, seq_len, n_heads, d_head)

    our_out = apply_rotary(x, our_cos, our_sin)  # (B, T, nh, hs)

    # olmo expects (B, nh, T, hs)
    x_olmo = x.permute(0, 2, 1, 3).contiguous()
    olmo_out = olmo_rope._apply_rotary_pos_emb(olmo_sin, olmo_cos, x_olmo)
    olmo_out = olmo_out.permute(0, 2, 1, 3)  # back to (B, T, nh, hs)

    assert torch.allclose(our_out, olmo_out, atol=1e-5), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


# ===========================================================================
# FFN  (model/ffn.py)
# ref: https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/feed_forward.py
# ===========================================================================


def test_ffn_output_shape():
    ffn = FFN(_CFG)
    x = torch.randn(2, 10, _CFG.d_model)
    assert ffn(x).shape == x.shape


def test_ffn_no_bias():
    ffn = FFN(_CFG)
    for name, mod in ffn.named_modules():
        if isinstance(mod, nn.Linear):
            assert mod.bias is None, f"{name} should have no bias"


def test_ffn_matches_olmo_core():
    """FFN output should match olmo_core FeedForward with identical weights.

    olmo_core naming: w1=gate, w3=up, w2=down
    forward: w2(silu(w1(x)) * w3(x))
    our naming: gate, up, down
    forward: down(silu(gate(x)) * up(x))
    """
    from olmo_core.nn.feed_forward import FeedForward

    d_model = 64
    hidden = 128
    cfg = TransformerConfig(
        d_model=d_model, n_heads=4, n_kv_heads=2, n_layers=2,
        ffn_hidden_dim=hidden, vocab_size=512, max_seq_len=32,
    )
    our_ffn = FFN(cfg)
    olmo_ffn = FeedForward(d_model=d_model, hidden_size=hidden, bias=False)

    with torch.no_grad():
        olmo_ffn.w1.weight.copy_(our_ffn.gate.weight)  # gate → w1
        olmo_ffn.w3.weight.copy_(our_ffn.up.weight)    # up   → w3
        olmo_ffn.w2.weight.copy_(our_ffn.down.weight)  # down → w2

    x = torch.randn(2, 10, d_model)
    our_out = our_ffn(x)
    olmo_out = olmo_ffn(x)
    assert torch.allclose(our_out, olmo_out, atol=1e-5), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


# ===========================================================================
# TransformerConfig sliding-window helper (model/config.py)
# ref: https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/attention/__init__.py
# ===========================================================================


def test_window_size_for_layer_none_when_disabled():
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=8,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32,
    )
    assert all(cfg.window_size_for_layer(i) is None for i in range(8))


def test_window_size_for_layer_matches_olmo3_pattern():
    """pattern=[4096,4096,4096,-1], force_full_attention_on_last_layer=True:
    every 4th layer is full attention, and the last layer is always full."""
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=8,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32,
        sliding_window_pattern=(4096, 4096, 4096, -1),
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
    )
    expected = [4096, 4096, 4096, None, 4096, 4096, 4096, None]
    assert [cfg.window_size_for_layer(i) for i in range(8)] == expected


def test_window_size_for_layer_force_full_first_layer():
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=5,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32,
        sliding_window_pattern=(128,),
        force_full_attention_on_first_layer=True,
        force_full_attention_on_last_layer=False,
    )
    assert cfg.window_size_for_layer(0) is None
    assert cfg.window_size_for_layer(1) == 128
    assert cfg.window_size_for_layer(4) == 128  # last layer not forced full here


# ===========================================================================
# Attention  (model/attention.py) — requires CUDA for flash_attn
# ===========================================================================


@requires_gpu
def test_attention_output_shape():
    from t0_training.model.attention import Attention

    attn = Attention(_CFG).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, _CFG.d_model, dtype=torch.bfloat16, device="cuda")
    out = attn(x, cos, sin)
    assert out.shape == x.shape


@requires_gpu
def test_attention_output_finite():
    from t0_training.model.attention import Attention

    attn = Attention(_CFG).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, _CFG.d_model, dtype=torch.bfloat16, device="cuda")
    out = attn(x, cos, sin)
    assert torch.isfinite(out).all()


@requires_gpu
def test_attention_gqa_config():
    """GQA config (n_kv_heads < n_heads) should run without error."""
    from t0_training.model.attention import Attention

    cfg = TransformerConfig(
        d_model=64, n_heads=8, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32,
    )
    attn = Attention(cfg).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, cfg.d_model, dtype=torch.bfloat16, device="cuda")
    out = attn(x, cos, sin)
    assert out.shape == x.shape


@requires_gpu
def test_attention_causal_mask():
    """Future tokens should not influence past token outputs.

    With a causal mask, the output at position t depends only on positions ≤ t.
    We verify this by zeroing out all tokens after position t and checking that
    the output at position t is unchanged.
    """
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    # flash_attn only supports fp16/bf16
    attn = Attention(cfg).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    x = torch.randn(1, 8, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out_full = attn(x, cos, sin).float()

    # Zero out positions 4 onwards and rerun
    x2 = x.clone()
    x2[:, 4:, :] = 0.0
    out_masked = attn(x2, cos, sin).float()

    # First 4 positions should be identical
    assert torch.allclose(out_full[:, :4, :], out_masked[:, :4, :], atol=1e-4), (
        f"max diff: {(out_full[:, :4, :] - out_masked[:, :4, :]).abs().max().item()}"
    )


@requires_gpu
def test_attention_sliding_window_ignores_tokens_outside_window():
    """With window_size=W, output at position t depends only on positions in
    [t - W + 1, t]. Changing a token well outside that window must not change
    the output at t."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    window = 3
    # flash_attn only supports fp16/bf16
    attn = Attention(cfg, window_size=window).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    T = 10
    x = torch.randn(1, T, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out_full = attn(x, cos, sin).float()

    # Position 8's window is [6, 8]; position 0 is well outside it.
    x2 = x.clone()
    x2[:, 0, :] = 0.0
    out_modified = attn(x2, cos, sin).float()

    assert torch.allclose(out_full[:, 8, :], out_modified[:, 8, :], atol=1e-4), (
        f"max diff: {(out_full[:, 8, :] - out_modified[:, 8, :]).abs().max().item()}"
    )


@requires_gpu
def test_attention_sliding_window_boundary():
    """Pin the exact window boundary (catches off-by-one in the flash-attn
    window_size tuple): with window W, position t attends to [t - W + 1, t]
    inclusive. Perturbing position t-W+1 (oldest inside the window) must change
    the output at t; perturbing position t-W (just outside) must not."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    window, t, T = 4, 8, 10
    attn = Attention(cfg, window_size=window).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    x = torch.randn(1, T, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out = attn(x, cos, sin).float()

    # Position t-W = 4 is just OUTSIDE the window of position 8: no effect.
    x_outside = x.clone()
    x_outside[:, t - window, :] += 1.0
    out_outside = attn(x_outside, cos, sin).float()
    assert torch.allclose(out[:, t, :], out_outside[:, t, :], atol=1e-4), (
        f"position {t - window} leaked into the window; "
        f"max diff: {(out[:, t, :] - out_outside[:, t, :]).abs().max().item()}"
    )

    # Position t-W+1 = 5 is the oldest position INSIDE the window: must matter.
    x_inside = x.clone()
    x_inside[:, t - window + 1, :] += 1.0
    out_inside = attn(x_inside, cos, sin).float()
    assert not torch.allclose(out[:, t, :], out_inside[:, t, :], atol=1e-4), (
        f"position {t - window + 1} should be inside the window but had no effect"
    )


@requires_gpu
def test_attention_full_attention_uses_all_prior_tokens():
    """Sanity check for the test above: without a window, an early-token change
    DOES propagate to a later position's output."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    # flash_attn only supports fp16/bf16
    attn = Attention(cfg, window_size=None).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    T = 10
    x = torch.randn(1, T, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out_full = attn(x, cos, sin).float()

    x2 = x.clone()
    x2[:, 0, :] = 0.0
    out_modified = attn(x2, cos, sin).float()

    assert not torch.allclose(out_full[:, 8, :], out_modified[:, 8, :], atol=1e-4)


def _matching_olmo_attention(cfg, window_size, our_attn):
    """Build an olmo_core Attention mirroring our Attention (no bias, flash
    backend, full-width qk-norm when enabled) with copied weights, on CUDA in
    bf16."""
    from olmo_core.nn.attention import Attention as OlmoAttention, AttentionBackendName
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.rope import RoPEConfig

    olmo_attn = OlmoAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        bias=False,
        rope=RoPEConfig(theta=int(cfg.rope_theta)),
        # OLMo3 qk-norm: RMSNorm over the full projection width (use_head_qk_norm=False)
        qk_norm=(
            LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
            if cfg.qk_norm else None
        ),
        backend=AttentionBackendName.flash_2,
        window_size=window_size,
        dtype=torch.bfloat16,
        init_device="cuda",
    )
    with torch.no_grad():
        olmo_attn.w_q.weight.copy_(our_attn.wq.weight)
        olmo_attn.w_k.weight.copy_(our_attn.wk.weight)
        olmo_attn.w_v.weight.copy_(our_attn.wv.weight)
        olmo_attn.w_out.weight.copy_(our_attn.wo.weight)
        if cfg.qk_norm:
            olmo_attn.q_norm.weight.copy_(our_attn.q_norm.weight)
            olmo_attn.k_norm.weight.copy_(our_attn.k_norm.weight)
    return olmo_attn


@requires_gpu
def test_attention_matches_olmo_core():
    """Full (global) attention: numerical match against olmo_core Attention
    with identical weights, GQA, and RoPE (qk_norm off to isolate the
    attention math; see test_attention_qk_norm_matches_olmo_core)."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    attn = Attention(cfg, window_size=None).cuda().to(torch.bfloat16)
    olmo_attn = _matching_olmo_attention(cfg, None, attn)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    x = torch.randn(2, 16, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    our_out = attn(x, cos, sin).float()
    olmo_out = olmo_attn(x).float()

    assert torch.allclose(our_out, olmo_out, atol=2e-2), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


@requires_gpu
def test_attention_qk_norm_matches_olmo_core():
    """qk-norm enabled: numerical match against olmo_core Attention. OLMo3
    normalises Q/K over the full projection width with a shared RMS statistic
    (use_head_qk_norm=False), which model/attention.py mirrors."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=True,
    )
    attn = Attention(cfg, window_size=None).cuda().to(torch.bfloat16)
    # Non-trivial norm weights so a per-head vs full-width mix-up can't hide
    # behind all-ones initialisation.
    with torch.no_grad():
        attn.q_norm.weight.uniform_(0.5, 1.5)
        attn.k_norm.weight.uniform_(0.5, 1.5)
    olmo_attn = _matching_olmo_attention(cfg, None, attn)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    x = torch.randn(2, 16, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    our_out = attn(x, cos, sin).float()
    olmo_out = olmo_attn(x).float()

    assert torch.allclose(our_out, olmo_out, atol=2e-2), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


@requires_gpu
def test_attention_sliding_window_matches_olmo_core():
    """Sliding-window attention: numerical match against olmo_core Attention
    with the same window_size, using T > window so the window actually binds."""
    from t0_training.model.attention import Attention

    torch.manual_seed(0)
    cfg = TransformerConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
        ffn_hidden_dim=128, vocab_size=512, max_seq_len=32, qk_norm=False,
    )
    window = 4
    attn = Attention(cfg, window_size=window).cuda().to(torch.bfloat16)
    olmo_attn = _matching_olmo_attention(cfg, window, attn)
    cos, sin = precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()

    x = torch.randn(2, 16, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    our_out = attn(x, cos, sin).float()
    olmo_out = olmo_attn(x).float()

    assert torch.allclose(our_out, olmo_out, atol=2e-2), (
        f"max diff: {(our_out - olmo_out).abs().max().item()}"
    )


# ===========================================================================
# TransformerBlock  (model/block.py)
# ref: https://github.com/allenai/OLMo-core/blob/fa6c5014/src/olmo_core/nn/transformer/block.py
# ===========================================================================


@requires_gpu
def test_block_output_shape():
    from t0_training.model.block import TransformerBlock

    block = TransformerBlock(_CFG).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, _CFG.d_model, dtype=torch.bfloat16, device="cuda")
    out = block(x, cos, sin)
    assert out.shape == x.shape


@requires_gpu
def test_block_residual_connection():
    """Block output should differ from input (residual is non-trivial)."""
    from t0_training.model.block import TransformerBlock

    torch.manual_seed(1)
    block = TransformerBlock(_CFG).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, _CFG.d_model, dtype=torch.bfloat16, device="cuda")
    out = block(x, cos, sin)
    assert not torch.allclose(out, x)


@requires_gpu
def test_block_output_finite():
    from t0_training.model.block import TransformerBlock

    block = TransformerBlock(_CFG).cuda().to(torch.bfloat16)
    cos, sin = precompute_freqs(_D_HEAD, _CFG.max_seq_len, _CFG.rope_theta)
    cos, sin = cos.cuda(), sin.cuda()
    x = torch.randn(2, 8, _CFG.d_model, dtype=torch.bfloat16, device="cuda")
    out = block(x, cos, sin)
    assert torch.isfinite(out).all()


# ===========================================================================
# Transformer  (model/transformer.py)
# ref: https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/transformer/model.py
# ===========================================================================


@requires_gpu
def test_transformer_output_shape():
    from t0_training.model.transformer import Transformer

    model = Transformer(_CFG).cuda().to(torch.bfloat16)
    tokens = torch.randint(0, _CFG.vocab_size, (2, 8), device="cuda")
    logits = model(tokens)
    assert logits.shape == (2, 8, _CFG.vocab_size)


@requires_gpu
def test_transformer_weight_tying():
    """Embedding and LM head should share the same weight tensor."""
    from t0_training.model.transformer import Transformer

    model = Transformer(_CFG).cuda().to(torch.bfloat16)
    assert model.embedding.weight is model.lm_head.weight


@requires_gpu
def test_transformer_output_finite():
    from t0_training.model.transformer import Transformer

    model = Transformer(_CFG).cuda().to(torch.bfloat16)
    tokens = torch.randint(0, _CFG.vocab_size, (2, 8), device="cuda")
    logits = model(tokens)
    assert torch.isfinite(logits).all()


@requires_gpu
def test_transformer_backward():
    """Backward pass should complete without error and produce non-zero gradients."""
    from t0_training.model.transformer import Transformer
    import torch.nn.functional as F

    torch.manual_seed(42)
    model = Transformer(_CFG).cuda().to(torch.bfloat16)
    tokens = torch.randint(0, _CFG.vocab_size, (2, 8), device="cuda")
    logits = model(tokens)
    loss = F.cross_entropy(logits.view(-1, _CFG.vocab_size).float(), tokens.view(-1))
    loss.backward()

    # At least some parameters should have gradients
    has_grad = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert any(has_grad)


@requires_gpu
def test_transformer_rope_buffers_not_saved_in_state_dict():
    """RoPE cos/sin buffers should not appear in the state dict (persistent=False)."""
    from t0_training.model.transformer import Transformer

    model = Transformer(_CFG).cuda().to(torch.bfloat16)
    keys = set(model.state_dict().keys())
    assert "cos" not in keys
    assert "sin" not in keys


@requires_gpu
def test_transformer_parameter_count_reasonable():
    """3B-style config should have roughly the right parameter count."""
    from t0_training.model.config import config_3b
    from t0_training.model.transformer import Transformer

    model = Transformer(config_3b)
    n_params = sum(p.numel() for p in model.parameters())
    # Tied weights: lm_head shares embedding, so don't double-count
    # 3B style ≈ 3B params; allow ±20% margin
    assert 2.4e9 < n_params < 3.6e9, f"unexpected param count: {n_params:,}"
