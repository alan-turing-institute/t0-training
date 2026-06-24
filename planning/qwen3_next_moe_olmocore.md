# Plan: Replicate the Qwen3.6-35B-A3B hybrid-MoE architecture in OLMo-core

## Context

We want to reproduce the **Qwen3.6-35B-A3B** architecture (a "Qwen3-Next"-style
hybrid linear-attention + sparse-MoE model) inside the OLMo-core training stack
used by this repo (`t0-training`). OLMo-core already ships a dense Qwen3 family
(`TransformerConfig.qwen3_*`) and a full MoE stack, but it has **no hybrid
(Gated-DeltaNet + full-attention) MoE preset** and is missing two small
architectural features Qwen uses (partial RoPE, zero-centered RMSNorm).

Decisions locked with the user:
- **Goal**: get the architecture building + training correctly, validated with a
  small-scale run, plus a costed path to the real 35B-A3B. (Not weight-loading Qwen.)
- **Fidelity**: *architecturally equivalent* (same hybrid pattern, sparsity,
  shapes) — train from scratch. Skip weight-compat-only features (MTP head,
  bit-exact partial-RoPE layout, tokenizer/vocab match).
- **Extras in scope**: **Partial RoPE** + **Zero-centered RMSNorm**. **MTP is out of scope.**
- **Compute / harness**: follow this repo's existing convention —
  [scripts/moe_train_independent.py](../scripts/moe_train_independent.py)
  (dolma data via `t0_training.data.resolve_data_paths`, dolma2 tokenizer,
  FSDP + pipeline parallel, AdamW + CosWithWarmup, launched with `uv run torchrun`).

---

## Part 1 — How the Qwen3.6-35B-A3B architecture works

Verified from the published `config.json`
(`huggingface.co/Qwen/Qwen3.6-35B-A3B`) and the Qwen3-Next blog/NVIDIA writeup.

**Hybrid sequence mixer (the defining feature).** Layers alternate
**3 linear-attention layers : 1 full-attention layer** (`full_attention_interval: 4`).
- *Linear layers* = **Gated DeltaNet** (gated delta rule, O(n), constant-size
  recurrent state, short causal conv of kernel 4). Dims: `linear_num_key_heads=16`,
  `linear_key_head_dim=128`, `linear_num_value_heads=32`, `linear_value_head_dim=128`,
  `linear_conv_kernel_dim=4`. No RoPE on these layers.
- *Full layers* = **Gated Attention**: standard GQA softmax attention with QK-norm,
  an output gate, and **partial RoPE** (`partial_rotary_factor=0.25` → RoPE on
  64 of 256 head dims). `num_attention_heads=16`, `num_key_value_heads=2`,
  `head_dim=256`, `rope_theta=10_000_000`. (The published config is
  multimodal-capable — `model_type: qwen3_5_moe`, mRoPE with
  `mrope_section=[11,11,10]`; we train text-only, so mRoPE reduces to 1D RoPE —
  see fidelity caveat in 3.1.)

**High-sparsity MoE in (almost) every layer.** `num_experts=256`,
`num_experts_per_tok=8`, **+1 shared (always-on) expert**
(`shared_expert_intermediate_size=512`), `moe_intermediate_size=512`,
sigmoid/normalized router (`norm_topk_prob`). ~3B of ~35B params active per token.

**Other:** `hidden_size=2048`, `num_hidden_layers=40`, `vocab_size=248320`
(untied embeddings), `rms_norm_eps=1e-6`, `max_position_embeddings=262144`,
zero-centered + weight-decayed RMSNorm, `mtp_num_hidden_layers=1` (MTP — *out of scope*).

> Sibling reference Qwen3-Next-80B-A3B is identical in shape except 48 layers,
> 512 experts, top-10, intermediate_size 5120. Confirms the family pattern.

---

## Part 2 — What OLMo-core already has vs. the gaps

Source lives in `.venv/lib/python3.13/site-packages/olmo_core/`.

**Already present (reuse directly):**
- **Gated DeltaNet** linear attention: `GatedDeltaNet` / `GatedDeltaNetConfig`
  in `olmo_core/nn/attention/recurrent.py` (class at line 34)
  — a registered `SequenceMixerConfig`. Maps to Qwen with `n_heads=16,
  n_v_heads=32, head_dim=128, expand_v=1.0, conv_size=4`. **Requires the
  `flash-linear-attention` (FLA) package** (`has_fla()` assert).
- **Gated Attention**: `AttentionConfig.gate` (`GateConfig`, headwise) +
  `qk_norm`/`use_head_qk_norm` + GQA via `n_kv_heads`
  (`olmo_core/nn/attention/__init__.py:93,194`).
- **High-sparsity MoE with shared experts**: `MoEConfig`
  (`num_experts`, `router.top_k`, `shared_mlp`, sigmoid gating, lb/z-loss,
  dropless, EP/TP) in `olmo_core/nn/moe/`.
  MoE block types incl. `moe`, `moe_reordered_norm`, `moe_hybrid`,
  `moe_hybrid_reordered_norm`.
- **Per-layer heterogeneous blocks** — the key enabler for the 3:1 hybrid:
  `TransformerConfig.block_overrides: Dict[int, TransformerBlockConfig]`
  (`olmo_core/nn/transformer/config.py:329`),
  also reachable via `llama_like(..., block_mods={i: fn})`
  (`config.py:1469-1549`).
  A `MoETransformerBlock` accepts **any** `sequence_mixer`, so each block can be
  (GatedDeltaNet + MoE) or (Attention + MoE).
- **Builders to extend**: `llama_like_moe` (`config.py:1562`),
  dense `qwen3_*` presets (`config.py:1287-1443`),
  `smallmoe` (`config.py:1015`).
- **Training infra**: FSDP/HSDP/TP/PP/CP/**EP**, AdamW, CosWithWarmup, callbacks —
  exactly what [scripts/moe_train_independent.py](../scripts/moe_train_independent.py) uses.

**Gaps to implement:**
1. **Partial RoPE** — `RoPEConfig` (`olmo_core/nn/rope.py:294`)
   applies RoPE to the full head dim; no `rotary_factor`/`rotary_dim` field. *In scope.*
2. **Zero-centered RMSNorm** — not in `olmo_core/nn/layer_norm.py`. *In scope.*
3. **No built-in Qwen3-Next-style 3:1 (GatedDeltaNet/full-attention) MoE preset** —
   needs a new `qwen3_next_moe` builder. *In scope.* (OLMo *does* ship
   `small_hybrid_moe` + `moe_hybrid`/`moe_hybrid_reordered_norm` blocks
   (`config.py:1039,138-143`), but that "hybrid" is dense-MLP + MoE execution
   *overlap*, not the Qwen 3:1 sequence-mixer alternation we need.)
4. MTP head — *out of scope per decision.*

---

## Part 3 — Implementation

OLMo-core is a **third-party dependency under `.venv`**, so editing it in place is
not maintainable. Recommended approach: **keep all new code inside the
`t0_training` package** (subclass/extend OLMo-core configs), and only fall back to
patching the vendored OLMo-core if a hook is impossible. New module:
`t0_training/models/qwen_next.py`.

### 3.1 Partial RoPE (small, contained)
- Add `rotary_factor: float = 1.0` (or `rotary_dim: Optional[int]`) to a thin
  subclass of `RoPEConfig`, or — if patching OLMo-core is acceptable — to
  `RoPEConfig` + `RotaryEmbedding` in `olmo_core/nn/rope.py`.
- In the rotary application, rotate only the first `rotary_dim = int(head_size *
  rotary_factor)` dims and pass the remainder through unchanged
  (split → rotate slice → concat). Mirror the standard HF `partial_rotary_factor`
  semantics. Set `rotary_factor=0.25` for the full-attention layers only.
- *Note:* Gated-DeltaNet layers take no RoPE, so this only affects the 1-in-4 full layers.
- *Fidelity caveat:* this gives the same *fraction* of rotated dims as HF
  (`partial_rotary_factor=0.25` → 64 of 256 head dims), but it is **not
  bit-identical** to HF's layout. The published config (`model_type:
  qwen3_5_moe`, arch `Qwen3_5MoeForConditionalGeneration`) is multimodal-capable
  and uses **mRoPE** (`mrope_interleaved: true`, `mrope_section: [11, 11, 10]` —
  the 32 rotary freq-pairs split across temporal/height/width axes) plus
  interleaved freq ordering. For our **text-only, from-scratch** training there's
  no multimodal position input, so mRoPE collapses to standard 1D RoPE on the
  same 64 dims; we deliberately implement plain partial-RoPE and **skip
  mRoPE-section interleaving** (architectural equivalence, not weight-compat,
  per the locked decision).

### 3.2 Zero-centered RMSNorm
- Add a `LayerNormType.zero_centered_rms` variant in `olmo_core/nn/layer_norm.py`
  (or a `t0_training` subclass): compute RMSNorm with `(1 + weight)` as the scale
  and initialize `weight` at 0, so weight decay pulls the effective gain toward 1.
- Wire it as the global `layer_norm` (and optionally QK-norm) in the new builder.

### 3.3 New preset: `qwen3_next_moe`
Add a classmethod (in `t0_training/models/qwen_next.py`) that composes existing
pieces. Sketch:

```python
# Single, consistent API: explicit dims, no overloaded `scale` knob.
# The small validation config and the full 35B-A3B config are just two call
# sites passing different d_model/n_layers/num_experts (see SCALES below).
def qwen3_next_moe(*, vocab_size, d_model=2048, n_layers=40,
                   full_attention_interval=4, num_experts=256, top_k=8,
                   moe_intermediate_size=512, shared_expert_hidden_size=512,
                   **kw):
    # 1) base full-attention + MoE block via llama_like_moe (reordered_norm=False)
    base = TransformerConfig.llama_like_moe(
        d_model=d_model, vocab_size=vocab_size, n_layers=n_layers, n_heads=16,
        n_kv_heads=2, head_dim=256, num_experts=num_experts, top_k=top_k,
        expert_hidden_size=moe_intermediate_size,
        shared_expert_hidden_size=shared_expert_hidden_size,
        dropless=True, qk_norm=True, use_head_qk_norm=True,
        rope_theta=10_000_000, layer_norm_eps=1e-6,
        gate=GateConfig(granularity=GateGranularity.headwise),   # gated attention
        # router: sigmoid + normalized to mirror Qwen norm_topk_prob (see below)
        # rope rotary_factor=0.25 set on the attention config (3.1)
    )
    # 2) replace 3-of-4 layers with GatedDeltaNet + MoE via block_overrides.
    #    Build a FRESH block per layer — do NOT share one object across keys,
    #    or a later mutation would leak across layers.
    def delta_block():
        blk = base.block.copy()
        blk.sequence_mixer = GatedDeltaNetConfig(
            n_heads=16, n_v_heads=32, head_dim=128, expand_v=1.0, conv_size=4)
        return blk
    base.block_overrides = {i: delta_block() for i in range(n_layers)
                            if (i + 1) % full_attention_interval != 0}
    return base

# Named scales (replaces the earlier `scale=` arg — keeps one source of truth):
SCALES = {
    # FLA-free plumbing test: full_attention_interval=1 => every layer is
    # gated-attention + MoE, so NO GatedDeltaNet is constructed and the `fla`
    # import path is never touched. Runs on the DGX Spark without FLA. See 3.5.
    "smoke-attn": dict(d_model=512, n_layers=12, num_experts=32, top_k=4,
                       full_attention_interval=1),
    # True hybrid smoke (3:1 GatedDeltaNet:attention) — REQUIRES FLA (cluster).
    "smoke":      dict(d_model=512, n_layers=12, num_experts=32, top_k=4),
    "full":       dict(d_model=2048, n_layers=40, num_experts=256, top_k=8),
}
```

- **Router**: set `MoERouterConfig(top_k=8, gating_function=sigmoid,
  normalize_expert_weights=1.0)` to match Qwen's normalized-sigmoid routing
  (`norm_topk_prob`); keep `lb_loss_weight`/`z_loss_weight` from `router_aux_loss_coef≈0.001`.
- The **same builder** produces both the small validation model and the full
  35B-A3B (Part 4) — the `SCALES` entries just pass different
  `d_model/n_layers/num_experts/top_k`; no overloaded `scale` argument.

### 3.3a FLA-free testing path (DGX Spark, no `flash-linear-attention`)
`GatedDeltaNet` hard-requires FLA in three places — the `chunk_gated_delta_rule`
kernel, the `FusedRMSNormGated` output norm, and even its `CausalConv1d`
(`dispatch_causal_conv1d` also asserts `has_fla()`). There is **no torch
fallback** in OLMo-core, so a real hybrid run cannot start without FLA.

**But the FLA dependency is isolated to the linear-attention layers.** The
`smoke-attn` profile above sets `full_attention_interval=1`, so every layer is
gated-attention + MoE and **no `GatedDeltaNet` is ever constructed** — the `fla`
import path is never touched. This runs on the Spark with **no FLA installed**
and exercises everything except the Gated DeltaNet mixer itself:
- MoE stack (256-expert sparsity pattern, shared expert, sigmoid+normalized router, lb/z-loss)
- partial RoPE on the attention layers
- zero-centered RMSNorm
- the `block_overrides` heterogeneity mechanism (degenerate at interval=1, but the
  builder/codepath is the same)
- training loop, dolma data, FSDP+PP, checkpoint + saved-config

Plan: do first plumbing validation on the Spark with `smoke-attn` (no FLA);
defer the true 3:1 hybrid smoke (`smoke`) and the full run to the cluster, where
FLA is available. (Quick win worth trying first: `uv pip install
flash-linear-attention` on the Spark — its GB10 is Blackwell and FLA is
Triton-based; the usual blocker is ARM64 wheels, not the GPU.)

### 3.4 Training script
Copy [scripts/moe_train_independent.py](../scripts/moe_train_independent.py) →
`scripts/qwen_next_train.py`, changing only:
- `model_config = qwen3_next_moe(vocab_size=tokenizer_config.padded_vocab_size(), **SCALES["smoke"])`
- For the **larger configs**, add `ep_config=TransformerExpertParallelConfig(degree=...)`
  (EP is the right parallelism for 256 experts). **Constraint — verified in this
  OLMo version** (`olmo_core/distributed/parallel/__init__.py:189-200`):
  - EP **requires `hsdp`** data parallel — it raises
    `"expert parallelism can currently only be used with HSDP"` under `fsdp`.
    So switch `dp_config.name` from `fsdp` → `hsdp` whenever EP is on.
  - With HSDP, **`ep.degree` must equal the shard degree** (line 222-224:
    `"expert parallelism + HSDP requires the same sharding degree"`).
  - EP is **mutually exclusive with TP** (line 189-191).
- The **smoke run uses no EP**, so it stays on plain `fsdp` + PP — only the
  full-scale path needs the HSDP switch.
- everything else (dolma data, dolma2 tokenizer, AdamW, CosWithWarmup,
  callbacks) stays identical to the repo convention.

### Validation milestone (small run)
Two staged smoke runs, both `fsdp+PP, no EP`, ~0.3–0.6B total:

1. **`smoke-attn` (Spark, no FLA)** — `full_attention_interval=1`, all
   gated-attention + MoE. Run first on the Spark to validate plumbing without
   `flash-linear-attention` (see 3.3a). `uv run torchrun --nproc-per-node=N
   scripts/qwen_next_train.py qwen-next-smoke-attn` → builds on `meta`, a few
   hundred steps, decreasing loss, finite lb/z-loss, checkpoint + saved config.
2. **`smoke` (cluster, FLA required)** — `full_attention_interval=4`, the true
   3:1 hybrid. Same checks **plus** the 3:1 GatedDeltaNet:attention layer pattern
   in the saved config. Run once FLA is available.

---

## Part 4 — Training the real Qwen3.6-35B-A3B equivalent

Target config in the new builder: `d_model=2048, n_layers=40, n_heads=16,
n_kv_heads=2, head_dim=256, num_experts=256, top_k=8, +1 shared,
moe_intermediate_size=512, full_attention_interval=4, vocab≈dolma2 padded`.
This yields ~33–35B total / ~3B active (matches Qwen). Use whatever vocab the
repo's dolma2 tokenizer gives (architectural equivalence, not weight-compat).

**This is a from-scratch frontier pretrain — by far the dominant cost:**
- **Data**: Qwen-class models train on ~15T+ tokens. Even a reduced-but-real run
  is multiple trillions. Reuse the repo's dolma3 mixes (`data/mixes/`) and the
  `NumpyFSLDatasetConfig` path; scale the mix far beyond the current 3.8B sample.
- **Compute**: rough order-of-magnitude `6 × active_params × tokens` ≈
  `6 × 3e9 × 5e12 ≈ 9e22` FLOPs → on the order of **thousands of H100-GPU-days**
  for a few-trillion-token run; the full 15T is ~10× that. This is a
  many-node, multi-week job — *not* feasible on a single node / DGX Spark, which
  is only for the small validation run.
- **Parallelism**: HSDP (data) + **expert parallelism** for the 256 experts +
  pipeline parallel across the 40 layers; bf16 params / fp32 reduce, activation
  checkpointing. OLMo-core supports all of these via the train-module configs.
  **Note (not optional in this OLMo version):** EP only runs under **HSDP** (not
  FSDP) and is **mutually exclusive with TP**, and EP+HSDP requires
  `ep.degree == shard_degree` — so the data-parallel config *must* be `hsdp`
  for any EP run (see Part 3.4).
- **Recipe knobs to tune**: lb/z-loss weights, router jitter, capacity factor (or
  dropless), warmup/cosine schedule, lr, and a long-context extension phase
  (rope_theta=1e7 already supports 256K) if long-context is wanted.

**Recommendation**: treat Part 4 as a *spec + cost estimate*. Land Parts 1–3 and
the small validation run first; only commit to a real pretrain once the
architecture is verified end-to-end and cluster budget is secured.

---

## Files to create / modify
- **New** `t0_training/models/__init__.py`, `t0_training/models/qwen_next.py` —
  partial-RoPE config helper, zero-centered RMSNorm helper, `qwen3_next_moe` builder.
- **New** `scripts/qwen_next_train.py` — clone of `moe_train_independent.py`.
- **New** `tests/test_qwen_next.py` — assert build on `meta`, param/active-param
  counts (~35B/~3B at full scale), the 3:1 hybrid layer pattern, **partial RoPE
  enabled on every full-attention layer and absent on every GatedDeltaNet layer**,
  and **router = sigmoid + normalized** (Qwen `norm_topk_prob` behavior).
- **Modify** `pyproject.toml` — add **`flash-linear-attention`** (FLA) as an
  **optional extra** (mirror OLMo-core's own `fla` extra). It is currently
  missing from this repo's deps (only `flash-attn` is present). FLA is
  **required for any run that includes GatedDeltaNet layers** — `GatedDeltaNet`
  hard-asserts `has_fla()` (kernel, gated norm, *and* its causal conv), with no
  torch fallback. It is **not needed** for the `smoke-attn` plumbing profile
  (`full_attention_interval=1`, no GatedDeltaNet — see 3.3a), which is the
  intended first test on the FLA-less DGX Spark.
- **Possibly patch** vendored `olmo_core/nn/rope.py` + `nn/layer_norm.py` *only if*
  partial-RoPE / zero-centered norm cannot be expressed by subclassing from
  `t0_training`. Prefer subclassing.

## Verification
1. `uv run python -c "...build qwen3_next_moe(**SCALES['full']) on meta; print num_params, num_active_params"`
   → ~35B total, ~3B active; assert layer types alternate 3 GatedDeltaNet : 1 Attention.
2. `uv run pytest tests/test_qwen_next.py` — incl. per-layer rotary checks
   (RoPE on full-attn layers only) and router = sigmoid + normalized.
3. Small runs (staged):
   - `qwen-next-smoke-attn` — FLA-free plumbing test on the Spark (3.3a):
     loss decreases, lb/z-loss finite, checkpoint + saved config written.
   - `qwen-next-smoke` — true 3:1 hybrid on the cluster (FLA installed): same,
     plus the 3:1 layer pattern in the saved config.
4. Guard FLA availability (`has_fla()`) **only when the config contains
   GatedDeltaNet layers** (i.e. `full_attention_interval > 1`), raising a clear
   error pointing at the `flash-linear-attention` extra. The `smoke-attn` profile
   must run cleanly with FLA absent.
