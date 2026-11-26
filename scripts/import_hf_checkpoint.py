"""Imports a HF checkpoint into t0_training/model/transformer.py's Transformer
class and saves a flat seed checkpoint for DPO training.

Usage:
    uv run --no-sync python scripts/import_hf_checkpoint.py \\
        --output checkpoints/olmo3-7b-think-sft-imported.pt [--validate]

Logit-parity validation requires flash-attn + CUDA, so it only runs when
`--validate` is passed on a GPU node.
The download, name mapping, vocab padding, strict load and save need neither
so can run anywhere.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from t0_training.model.config import config_7b_yarn
from t0_training.model.transformer import Transformer

DEFAULT_HF_ID = "allenai/Olmo-3-7B-Think-SFT"
REPO_ROOT = Path(__file__).resolve().parents[1]

# HF Olmo3 per-layer names -> transformer.py names. OLMo3 norms the sublayer
# *output* (post_*_layernorm), matching block.py's attn_norm/ffn_norm; there is
# no Llama-style input_layernorm.
_LAYER_KEYS = {
    "self_attn.q_proj.weight": "attn.wq.weight",
    "self_attn.k_proj.weight": "attn.wk.weight",
    "self_attn.v_proj.weight": "attn.wv.weight",
    "self_attn.o_proj.weight": "attn.wo.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "ffn.gate.weight",
    "mlp.up_proj.weight": "ffn.up.weight",
    "mlp.down_proj.weight": "ffn.down.weight",
    "post_attention_layernorm.weight": "attn_norm.weight",
    "post_feedforward_layernorm.weight": "ffn_norm.weight",
}


def load_hf_state_dict(hf_id: str) -> dict[str, torch.Tensor]:
    """Loads all safetensors shards into one flat dict (bf16). Reads the shards
    directly rather than building the HF model, which would double peak memory."""
    local_dir = Path(snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json"]))
    index = json.loads((local_dir / "model.safetensors.index.json").read_text())

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        with safe_open(local_dir / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def map_hf_state_dict(
    hf_state_dict: dict[str, torch.Tensor], n_layers: int, vocab_size: int,
) -> dict[str, torch.Tensor]:
    """Renames HF Olmo3 parameters to transformer.py's names and zero-pads the
    embedding/lm_head vocab dim (100278 in the checkpoint) up to `vocab_size`
    (the padded dolma2 size, 100352). Padded rows are never indexed."""

    def pad_vocab(t: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(vocab_size - t.shape[0], *t.shape[1:], dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    mapped = {
        "embedding.weight": pad_vocab(hf_state_dict["model.embed_tokens.weight"]),
        "lm_head.weight": pad_vocab(hf_state_dict["lm_head.weight"]),
        "norm.weight": hf_state_dict["model.norm.weight"],
    }
    for i in range(n_layers):
        for hf_key, key in _LAYER_KEYS.items():
            mapped[f"blocks.{i}.{key}"] = hf_state_dict[f"model.layers.{i}.{hf_key}"]
    return mapped


def validate_logit_parity(
    hf_id: str, imported_path: str, n_tokens: int = 10_000, min_top1: float = 0.97, max_mean_kl: float = 5e-3,
) -> None:
    """Runs the same 10k-token input (the repo's docs) through the HF model and
    the imported model and checks top-1 agreement and mean KL(HF || imported),
    over all positions and past the 4096 sliding window / 8192 YaRN original
    context.

    The sliding-window boundary is unit-tested in tests/test_model.py
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    paths = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
    text = "\n\n".join(p.read_text(errors="ignore") for p in paths)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].cuda()
    assert ids.shape[1] == n_tokens, f"only {ids.shape[1]} tokens of docs text available"

    hf_model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16).cuda().eval()
    hf_vocab = hf_model.config.vocab_size
    with torch.no_grad():
        ref = hf_model(ids).logits[0].float().cpu()
    del hf_model
    torch.cuda.empty_cache()

    # max_seq_len only sizes the non-persistent RoPE tables. assign=True binds the
    # bf16 weights without downcasting those float32 buffers.
    model = Transformer(dataclasses.replace(config_7b_yarn, max_seq_len=max(config_7b_yarn.max_seq_len, n_tokens))).cuda()
    model.load_state_dict(torch.load(imported_path, map_location="cuda"), strict=True, assign=True)
    with torch.no_grad():
        test = model.eval()(ids)[0, :, :hf_vocab].float().cpu()  # drop padded vocab rows

    top1 = (ref.argmax(-1) == test.argmax(-1)).float()
    kl = F.kl_div(F.log_softmax(test, -1), F.log_softmax(ref, -1), log_target=True, reduction="none").sum(-1)
    failures = []
    for region, start in [("all", 0), (">4096", 4096), (">8192", 8192)]:
        r_top1, r_kl = top1[start:].mean().item(), kl[start:].mean().item()
        print(f"{region:>6}: top-1 {100 * r_top1:6.2f}%, mean KL {r_kl:.2e}")
        if r_top1 < min_top1 or r_kl > max_mean_kl:
            failures.append(region)

    if failures:
        raise AssertionError(f"Logit parity FAILED for {failures} (min_top1={min_top1}, max_mean_kl={max_mean_kl})")
    print("Logit parity check PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID, help="HF repo id to import.")
    parser.add_argument("--output", default="checkpoints/olmo3-7b-think-sft-imported.pt", help="Where to save the flat state dict.")
    parser.add_argument("--validate", action="store_true", help="Run the logit-parity check after saving (needs CUDA + flash-attn).")
    args = parser.parse_args()

    print(f"Loading {args.hf_id} state dict...")
    mapped = map_hf_state_dict(load_hf_state_dict(args.hf_id), config_7b_yarn.n_layers, config_7b_yarn.vocab_size)

    # Build on the meta device and bind the mapped tensors directly (assign=True);
    # strict=True fails on any missing/unexpected key.
    with torch.device("meta"):
        model = Transformer(config_7b_yarn)
    model.load_state_dict(mapped, strict=True, assign=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved mapped checkpoint to {output_path}")

    if args.validate:
        if not torch.cuda.is_available():
            raise RuntimeError("--validate requires a CUDA GPU with flash-attn installed.")
        validate_logit_parity(args.hf_id, str(output_path))


if __name__ == "__main__":
    main()
