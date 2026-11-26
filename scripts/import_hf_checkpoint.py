"""Imports a HF checkpoint into t0_training/model/transformer.py's Transformer
class and produces a flat seed checkpoint for DPO training.

Usage:
    uv run --no-sync python scripts/import_hf_checkpoint.py \\
        --output checkpoints/olmo3-7b-think-sft-imported.pt

Logit-parity validation requires flash-attn + CUDA, so it only runs when
`--validate` is passed on a GPU node.
The download, name mapping, vocab padding, strict load and save need neither
so can run anywhere.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from t0_training.model.config import config_7b_yarn
from t0_training.model.transformer import Transformer

DEFAULT_HF_ID = "allenai/Olmo-3-7B-Think-SFT"


def load_hf_state_dict(hf_id: str) -> dict[str, torch.Tensor]:
    """Loads every tensor from the checkpoint's safetensors shards into one
    flat dict, at the source dtype (bf16).

    Uses safetensors' safe_open directly rather than
    transformers.AutoModelForCausalLM.from_pretrained -- that would also
    construct the full Olmo3ForCausalLM module, which we don't need (we only
    want the raw tensors) and which roughly doubles peak memory for a 7B
    model. safe_open mmaps each shard, so pages are backed by the file cache
    rather than anonymous RAM.
    """
    local_dir = Path(snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json"]))
    with open(local_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        with safe_open(local_dir / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def map_hf_state_dict(
    hf_state_dict: dict[str, torch.Tensor], n_layers: int, vocab_size: int,
) -> dict[str, torch.Tensor]:
    """Translate Olmo3ForCausalLM's HF parameter names into transformer.py's
    naming, confirmed directly against the real checkpoint's
    model.safetensors.index.json keys (not hardcoded from memory):

        lm_head.weight                                  -> lm_head.weight
        model.embed_tokens.weight                       -> embedding.weight
        model.norm.weight                                -> norm.weight
        model.layers.{i}.self_attn.q_proj.weight         -> blocks.{i}.attn.wq.weight
        model.layers.{i}.self_attn.k_proj.weight         -> blocks.{i}.attn.wk.weight
        model.layers.{i}.self_attn.v_proj.weight         -> blocks.{i}.attn.wv.weight
        model.layers.{i}.self_attn.o_proj.weight         -> blocks.{i}.attn.wo.weight
        model.layers.{i}.self_attn.q_norm.weight         -> blocks.{i}.attn.q_norm.weight
        model.layers.{i}.self_attn.k_norm.weight         -> blocks.{i}.attn.k_norm.weight
        model.layers.{i}.mlp.gate_proj.weight            -> blocks.{i}.ffn.gate.weight
        model.layers.{i}.mlp.up_proj.weight              -> blocks.{i}.ffn.up.weight
        model.layers.{i}.mlp.down_proj.weight            -> blocks.{i}.ffn.down.weight
        model.layers.{i}.post_attention_layernorm.weight -> blocks.{i}.attn_norm.weight
        model.layers.{i}.post_feedforward_layernorm.weight -> blocks.{i}.ffn_norm.weight

    The post_attention_layernorm/post_feedforward_layernorm names confirm
    OLMo3's reordered-norm block (norm applied to the sublayer *output*,
    matching block.py's attn_norm/ffn_norm) rather than Llama's pre-norm
    input_layernorm naming -- there is no input_layernorm key in the real
    checkpoint, exactly as expected.

    Also pads the embedding/lm_head vocab dimension from the checkpoint's
    raw dolma2 vocab size (100278, confirmed equal to
    TokenizerConfig.dolma2().vocab_size, eos_token_id, and pad_token_id) up
    to `vocab_size` -- config_7b/config_7b_yarn use the *padded* dolma2 vocab
    size (100352, see model/config.py's _DOLMA2_VOCAB_SIZE) so embedding/
    lm_head shapes agree with what the pretraining data pipeline tokenizes
    with, but the real checkpoint has no such padding. This is a real gap
    the plan's compatibility-check table didn't call out (it only flagged
    RoPE scaling and max_seq_len). The extra rows are never indexed by any
    real token id, so they're zero-filled.
    """
    mapped: dict[str, torch.Tensor] = {}

    def pad_vocab(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] == vocab_size:
            return t
        pad = torch.zeros(vocab_size - t.shape[0], *t.shape[1:], dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    mapped["embedding.weight"] = pad_vocab(hf_state_dict["model.embed_tokens.weight"])
    mapped["lm_head.weight"] = pad_vocab(hf_state_dict["lm_head.weight"])
    mapped["norm.weight"] = hf_state_dict["model.norm.weight"]

    for i in range(n_layers):
        src = f"model.layers.{i}."
        dst = f"blocks.{i}."
        mapped[dst + "attn.wq.weight"] = hf_state_dict[src + "self_attn.q_proj.weight"]
        mapped[dst + "attn.wk.weight"] = hf_state_dict[src + "self_attn.k_proj.weight"]
        mapped[dst + "attn.wv.weight"] = hf_state_dict[src + "self_attn.v_proj.weight"]
        mapped[dst + "attn.wo.weight"] = hf_state_dict[src + "self_attn.o_proj.weight"]
        mapped[dst + "attn.q_norm.weight"] = hf_state_dict[src + "self_attn.q_norm.weight"]
        mapped[dst + "attn.k_norm.weight"] = hf_state_dict[src + "self_attn.k_norm.weight"]
        mapped[dst + "ffn.gate.weight"] = hf_state_dict[src + "mlp.gate_proj.weight"]
        mapped[dst + "ffn.up.weight"] = hf_state_dict[src + "mlp.up_proj.weight"]
        mapped[dst + "ffn.down.weight"] = hf_state_dict[src + "mlp.down_proj.weight"]
        mapped[dst + "attn_norm.weight"] = hf_state_dict[src + "post_attention_layernorm.weight"]
        mapped[dst + "ffn_norm.weight"] = hf_state_dict[src + "post_feedforward_layernorm.weight"]

    return mapped


def validate_logit_parity(hf_id: str, imported_path: str, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    """Phase 2 validation: tokenize an identical short prompt, run it through
    both the real HF model and the freshly-imported internal model, and
    assert the logits match. Mirrors scripts/export_hf.py's own validation
    pattern (same tolerances, same "loosen if bf16 kernel rounding makes this
    too tight" caveat) in the opposite import direction. This is the single
    most important check in the whole import -- if it passes, the RoPE
    scaling (Phase 1) and the name mapping above are both correct.

    Requires CUDA + flash-attn (the internal model's Attention.forward() has
    no CPU fallback), so this only runs on a GPU node.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    prompt_ids = tokenizer("The quick brown fox jumps over", return_tensors="pt").input_ids.cuda()

    hf_model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        hf_logits = hf_model(prompt_ids).logits
    del hf_model
    torch.cuda.empty_cache()

    # Deliberately NOT model.to(torch.bfloat16)
    # Instead, build the model normally then assign=True binds the checkpoint's already-bf16 parameter tensors
    # directly as the live parameters without ever touching the buffers.
    model = Transformer(config_7b_yarn).cuda()
    model.load_state_dict(torch.load(imported_path, map_location="cuda"), strict=True, assign=True)
    model.eval()
    with torch.no_grad():
        internal_logits = model(prompt_ids)

    # The internal model's lm_head has extra padded-vocab rows the HF model
    # doesn't have (see map_hf_state_dict's pad_vocab) -- compare only the
    # real vocab range.
    internal_logits = internal_logits[..., : hf_logits.shape[-1]]

    torch.testing.assert_close(hf_logits.float(), internal_logits.float(), atol=atol, rtol=rtol)
    print(f"Logit parity check PASSED (atol={atol}, rtol={rtol}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID, help="HF repo id to import.")
    parser.add_argument("--output", default="checkpoints/olmo3-7b-think-sft-imported.pt", help="Where to save the flat mapped state dict.")
    parser.add_argument("--validate", action="store_true", help="Run the logit-parity check after saving (requires CUDA + flash-attn).")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for --validate's logit comparison.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for --validate's logit comparison.")
    args = parser.parse_args()

    print(f"Downloading/loading {args.hf_id} state dict...")
    hf_state_dict = load_hf_state_dict(args.hf_id)
    print(f"Loaded {len(hf_state_dict)} tensors.")

    mapped = map_hf_state_dict(hf_state_dict, config_7b_yarn.n_layers, config_7b_yarn.vocab_size)
    del hf_state_dict  # drop the (now-superseded) unpadded embedding/lm_head tensors

    print("Building Transformer(config_7b_yarn) on the meta device and assigning weights...")
    with torch.device("meta"):
        model = Transformer(config_7b_yarn)

    # assign=True is required since the model is constructed on the meta 
    # device.
    # This binds `mapped`'s tensors directly as the model's parameters
    # instead of copying into destination tensors.
    # strict=True raises on any missing/unexpected key in the mapped dict.
    model.load_state_dict(mapped, strict=True, assign=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved mapped checkpoint to {output_path}")

    if args.validate:
        if not torch.cuda.is_available():
            raise RuntimeError("--validate requires a CUDA GPU with flash-attn installed.")
        validate_logit_parity(args.hf_id, str(output_path), atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
