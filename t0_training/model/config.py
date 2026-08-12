from dataclasses import dataclass

from olmo_core.data import TokenizerConfig

# Padded vocab size for the dolma2 tokenizer used by the data pipeline
# (t0_training/olmo/data.py, data/dataset.py) -- must match so the
# embedding/lm_head shapes agree with what's actually tokenized.
_DOLMA2_VOCAB_SIZE = TokenizerConfig.dolma2().padded_vocab_size()


@dataclass
class TransformerConfig:
    d_model: int        # hidden dimension
    n_heads: int        # query heads
    n_kv_heads: int     # key/value heads (n_heads for MHA, < n_heads for GQA)
    n_layers: int
    ffn_hidden_dim: int # intermediate size in FFN (~8/3 * d_model for SwiGLU)
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 500_000.0
    qk_norm: bool = True    # RMSNorm on Q and K after projection (OLMo3 default)

    # Sliding-window attention (OLMo3). `sliding_window_pattern` is repeated
    # across layers; -1 means full (global) attention for that slot. None
    # disables SWA entirely (every layer gets full attention).
    # Matches SlidingWindowAttentionConfig in OLMo-core:
    # https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/attention/__init__.py
    sliding_window_pattern: tuple[int, ...] | None = None
    force_full_attention_on_first_layer: bool = False
    force_full_attention_on_last_layer: bool = True

    def window_size_for_layer(self, layer_idx: int) -> int | None:
        """Sliding window size for a layer, or None for full (global) attention."""
        if self.sliding_window_pattern is None:
            return None
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return None
        if self.force_full_attention_on_last_layer and layer_idx == self.n_layers - 1:
            return None
        effective_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_idx -= 1
        window = self.sliding_window_pattern[effective_idx % len(self.sliding_window_pattern)]
        return None if window == -1 else window


# OLMo3-3B (olmo3_3B model factory in OLMo-core): plain MHA (no GQA),
# sliding-window attention every 4th layer, full attention on the last layer.
config_3b = TransformerConfig(
    d_model=3328,
    n_heads=16,
    n_kv_heads=16,
    n_layers=16,
    ffn_hidden_dim=13312,
    vocab_size=_DOLMA2_VOCAB_SIZE,
    max_seq_len=4096,
    sliding_window_pattern=(4096, 4096, 4096, -1),
    force_full_attention_on_first_layer=False,
    force_full_attention_on_last_layer=True,
)

# OLMo3-7B (olmo3_7B model factory in OLMo-core): same attention pattern as 3B.
config_7b = TransformerConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,
    n_layers=32,
    ffn_hidden_dim=11008,
    vocab_size=_DOLMA2_VOCAB_SIZE,
    max_seq_len=4096,
    sliding_window_pattern=(4096, 4096, 4096, -1),
    force_full_attention_on_first_layer=False,
    force_full_attention_on_last_layer=True,
)
