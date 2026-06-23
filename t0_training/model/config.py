from dataclasses import dataclass


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


# Llama-3.2-3B style
config_3b = TransformerConfig(
    d_model=3072,
    n_heads=24,
    n_kv_heads=8,
    n_layers=28,
    ffn_hidden_dim=8192,
    vocab_size=128256,
    max_seq_len=4096,
)

# Llama-3.1-8B style (~7B active params)
config_7b = TransformerConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=8,
    n_layers=32,
    ffn_hidden_dim=14336,
    vocab_size=128256,
    max_seq_len=4096,
)
