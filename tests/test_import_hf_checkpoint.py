"""Tests for scripts/import_hf_checkpoint.py's map_hf_state_dict (Phase 2 of
planning/write_dpo_code.md).

Uses a small synthetic "HF-shaped" state dict (not the real 7B checkpoint)
so these run on CPU without a network call -- the real checkpoint's exact
key names were already confirmed directly against
model.safetensors.index.json when writing map_hf_state_dict, not hardcoded
from memory; these tests instead guard the *mapping logic* (name
translation, vocab padding, reordered-norm placement) against regressions.
"""

import torch

from scripts.import_hf_checkpoint import map_hf_state_dict
from t0_training.model.config import TransformerConfig
from t0_training.model.transformer import Transformer

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
_D_HEAD = _CFG.d_model // _CFG.n_heads


def _fake_hf_state_dict(cfg: TransformerConfig, raw_vocab_size: int) -> dict[str, torch.Tensor]:
    """Random tensors at the real checkpoint's HF key names/shapes, for a
    tiny config. `raw_vocab_size` simulates the real checkpoint's unpadded
    dolma2 vocab (100278) being smaller than our padded vocab_size (100352)."""
    sd = {
        "model.embed_tokens.weight": torch.randn(raw_vocab_size, cfg.d_model),
        "lm_head.weight": torch.randn(raw_vocab_size, cfg.d_model),
        "model.norm.weight": torch.randn(cfg.d_model),
    }
    for i in range(cfg.n_layers):
        src = f"model.layers.{i}."
        sd[src + "self_attn.q_proj.weight"] = torch.randn(cfg.n_heads * _D_HEAD, cfg.d_model)
        sd[src + "self_attn.k_proj.weight"] = torch.randn(cfg.n_kv_heads * _D_HEAD, cfg.d_model)
        sd[src + "self_attn.v_proj.weight"] = torch.randn(cfg.n_kv_heads * _D_HEAD, cfg.d_model)
        sd[src + "self_attn.o_proj.weight"] = torch.randn(cfg.d_model, cfg.n_heads * _D_HEAD)
        sd[src + "self_attn.q_norm.weight"] = torch.randn(cfg.n_heads * _D_HEAD)
        sd[src + "self_attn.k_norm.weight"] = torch.randn(cfg.n_kv_heads * _D_HEAD)
        sd[src + "mlp.gate_proj.weight"] = torch.randn(cfg.ffn_hidden_dim, cfg.d_model)
        sd[src + "mlp.up_proj.weight"] = torch.randn(cfg.ffn_hidden_dim, cfg.d_model)
        sd[src + "mlp.down_proj.weight"] = torch.randn(cfg.d_model, cfg.ffn_hidden_dim)
        sd[src + "post_attention_layernorm.weight"] = torch.randn(cfg.d_model)
        sd[src + "post_feedforward_layernorm.weight"] = torch.randn(cfg.d_model)
    return sd


def test_map_hf_state_dict_produces_exact_key_set():
    """Mapped keys must exactly match Transformer's own state_dict keys (cos/
    sin are non-persistent buffers, correctly excluded from both)."""
    hf_sd = _fake_hf_state_dict(_CFG, raw_vocab_size=_CFG.vocab_size)
    mapped = map_hf_state_dict(hf_sd, _CFG.n_layers, _CFG.vocab_size)

    model = Transformer(_CFG)
    assert set(mapped.keys()) == set(model.state_dict().keys())


def test_map_hf_state_dict_loads_into_real_transformer_strict():
    """The mapped dict must satisfy a real strict=True load -- the actual
    integration point used by scripts/import_hf_checkpoint.py's main()."""
    hf_sd = _fake_hf_state_dict(_CFG, raw_vocab_size=_CFG.vocab_size)
    mapped = map_hf_state_dict(hf_sd, _CFG.n_layers, _CFG.vocab_size)

    model = Transformer(_CFG)
    model.load_state_dict(mapped, strict=True)  # raises on any missing/unexpected/shape-mismatched key


def test_map_hf_state_dict_pads_vocab_with_zeros():
    """Real checkpoint's raw dolma2 vocab (100278) is smaller than our padded
    vocab_size (100352, see model/config.py's _DOLMA2_VOCAB_SIZE) -- the
    extra rows must be zero-filled, and the real rows must be preserved
    exactly (no reordering/corruption)."""
    raw_vocab_size = _CFG.vocab_size - 7
    hf_sd = _fake_hf_state_dict(_CFG, raw_vocab_size=raw_vocab_size)
    mapped = map_hf_state_dict(hf_sd, _CFG.n_layers, _CFG.vocab_size)

    for key, hf_key in [("embedding.weight", "model.embed_tokens.weight"), ("lm_head.weight", "lm_head.weight")]:
        assert mapped[key].shape == (_CFG.vocab_size, _CFG.d_model)
        assert torch.equal(mapped[key][:raw_vocab_size], hf_sd[hf_key])
        assert torch.equal(mapped[key][raw_vocab_size:], torch.zeros(7, _CFG.d_model))


def test_map_hf_state_dict_no_padding_when_vocab_already_matches():
    """When the source vocab already equals the target, pad_vocab is a no-op
    (regression check: must not accidentally truncate or pad a 0-row extra)."""
    hf_sd = _fake_hf_state_dict(_CFG, raw_vocab_size=_CFG.vocab_size)
    mapped = map_hf_state_dict(hf_sd, _CFG.n_layers, _CFG.vocab_size)

    assert torch.equal(mapped["embedding.weight"], hf_sd["model.embed_tokens.weight"])
    assert torch.equal(mapped["lm_head.weight"], hf_sd["lm_head.weight"])


def test_map_hf_state_dict_reordered_norm_not_swapped():
    """Guards against the exact silent bug the plan calls out: OLMo3's
    post_attention_layernorm/post_feedforward_layernorm (reordered-norm,
    applied to sublayer *output*) must map to attn_norm/ffn_norm
    respectively, not be swapped or confused with Llama's pre-norm
    input_layernorm naming (which doesn't exist in the real checkpoint)."""
    hf_sd = _fake_hf_state_dict(_CFG, raw_vocab_size=_CFG.vocab_size)
    mapped = map_hf_state_dict(hf_sd, _CFG.n_layers, _CFG.vocab_size)

    for i in range(_CFG.n_layers):
        assert torch.equal(
            mapped[f"blocks.{i}.attn_norm.weight"],
            hf_sd[f"model.layers.{i}.post_attention_layernorm.weight"],
        )
        assert torch.equal(
            mapped[f"blocks.{i}.ffn_norm.weight"],
            hf_sd[f"model.layers.{i}.post_feedforward_layernorm.weight"],
        )
