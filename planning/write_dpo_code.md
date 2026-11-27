# DPO Training Stack — Implementation Plan

## Overview

This plan covers adding DPO training to the existing from-scratch stack
(`t0_training/model,train,data,distributed,optim`, built per
[write_training_code.md](write_training_code.md)), starting from the
publicly released `allenai/Olmo-3-7B-Think-SFT` checkpoint and training on
`allenai/Dolci-Think-DPO-7B`. Same rationale as the original from-scratch
pretraining build: the value is in understanding the mechanics, not in the
output checkpoint (see [next_steps_nov2026.md](next_steps_nov2026.md)).

Every phase below either reuses existing `t0_training/` infrastructure
directly, or extends it additively (new opt-in code paths), so the existing
pretraining path (`scripts/train.py`, `config_3b.py`/`config_7b.py`) is
never touched.

### Key design choices

| Choice | Rationale |
|--------|-----------|
| **YaRN RoPE as a second code path** | Required to match `Olmo-3-7B-Think-SFT`'s real positional encoding — not optional, since its `attention_factor` scales attention at every position, not just beyond the original 8k context. |
| **Logit parity as the import gate** | The only reliable way to confirm the HF → internal weight mapping is correct before spending any compute training on it. |
| **Seeding ≠ resuming** | `CheckpointManager` stays untouched; a new flat-file loader seeds the policy/reference models before FSDP sharding, instead of stretching DCP resume to do a different job. |
| **Reference model is static, never checkpointed** | It's a re-derivable artifact (re-running import reproduces it exactly) — FSDP-sharded for memory only, no optimizer, no checkpoint entry. |
| **New preference-pair dataset class** | `Dolci-Think-DPO-7B`'s `{prompt, chosen, rejected}` schema doesn't fit `NumpyDataset`/`ConcatNumpyDataset`'s flat-token-stream model at all. |
| **Varlen attention packing, opt-in** | Avoids padding tokens silently participating in attention; existing fixed-batch pretraining call site is untouched. |
| **Hyperparameters from OLMo3's own recipe** | 1 epoch, β=5, LR=8e-8, linear decay, warmup ratio 0.1 — used as given rather than re-derived. |

### Compatibility check against the real checkpoint (do this first)

`t0_training/model/config.py`'s existing `config_7b` was built OLMo3-matched
and already agrees with `Olmo-3-7B-Think-SFT`'s real HF `config.json` on the
things that matter most: `n_kv_heads=32` (plain MHA, not GQA — already
correct), `ffn_hidden_dim=11008` (already correct), the sliding-window
pattern (`sliding_window_pattern=(4096,4096,4096,-1)`, full attention every
4th layer including the last — matches the real `layer_types` list exactly),
`qk_norm=True`, and untied embeddings (`transformer.py` already builds an
independent `lm_head`, matching the real checkpoint's `tie_word_embeddings:
false`).

Two real gaps, confirmed by reading the code, not assumed:

| Gap | Real checkpoint value | Current `t0_training/` state |
|---|---|---|
| RoPE scaling | YaRN (`rope_scaling: {rope_type: yarn, factor: 8.0, beta_fast: 32.0, beta_slow: 1.0, attention_factor: 1.2079441541679836, original_max_position_embeddings: 8192}`, extending to `max_position_embeddings: 65536`) | `model/rope.py`'s `precompute_freqs()` is plain RoPE — no frequency-ramp interpolation, no attention-factor scaling. |
| `max_seq_len` | effectively 65,536 (YaRN-extended) | `config_7b.max_seq_len=4096` | Needs raising to whatever the actual `Dolci-Think-DPO-7B` max example length turns out to be (profile first — Phase 5 — rather than assume the full 65,536 is needed). |

---

## Phase 1: YaRN RoPE scaling

**Files:** `model/rope.py`, `model/config.py`

Add YaRN as a second code path alongside the existing plain RoPE, selected
by new optional fields on `TransformerConfig` (`rope_scaling_factor`,
`rope_beta_fast`, `rope_beta_slow`, `rope_original_max_position_embeddings`,
`rope_attention_factor` — all `None` by default, so `config_3b`/`config_7b`
are completely unaffected).

**NTK-by-parts frequency interpolation.** Each frequency is classified by
its wavelength relative to the original context window
(`original_max_position_embeddings=8192`): high frequencies (short
wavelength, above `beta_fast`) are left alone — they already generalize
fine; low frequencies (below `beta_slow`) are divided by `factor`
(interpolated to the new context length); frequencies in between are
linearly ramped. Unlike naive linear position interpolation (which divides
every frequency by `factor` uniformly and degrades short-range resolution),
different parts of the frequency spectrum get different treatment.

**Attention temperature correction.** A constant `attention_factor`
multiplies the rotated Q/K to compensate for the fact that stretching
positions also stretches the effective dot-product magnitude distribution.
This applies uniformly across all positions — which is why YaRN isn't an
optional add-on for long sequences only; omitting it changes the model's
function everywhere.

```python
def precompute_freqs_yarn(
    d_head: int, max_seq_len: int, theta: float,
    factor: float, beta_fast: float, beta_slow: float,
    original_max_position_embeddings: int, attention_factor: float,
):
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / d_head))

    # Wavelength per frequency; ramp mask between beta_fast/beta_slow bounds
    # (see https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/rope.py
    # for the exact reference implementation to match against).
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, d_head, theta, original_max_position_embeddings)
    ramp = _yarn_linear_ramp(low, high, d_head // 2)
    inv_freq_interpolated = inv_freq / factor
    inv_freq = inv_freq * ramp + inv_freq_interpolated * (1 - ramp)

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    # attention_factor scales the rotation output, not the frequencies themselves
    return freqs.cos() * attention_factor, freqs.sin() * attention_factor
```

`_yarn_find_correction_range` and `_yarn_linear_ramp` are small helper
functions — implement directly from the YaRN paper or OLMo-core's own
`rope.py`, don't hand-derive from scratch; a sign/off-by-one error here
would silently produce slightly-wrong attention, which Phase 2's
logit-parity check is specifically there to catch.

### Phase 1 validation

Before touching any real checkpoint weights: unit-test
`precompute_freqs_yarn` against OLMo-core's own implementation directly —
same input config (factor=8.0, beta_fast=32.0, beta_slow=1.0,
original_max_position_embeddings=8192, attention_factor=1.2079441541679836),
assert the returned `cos`/`sin` tensors match OLMo-core's to float32
precision. This isolates the RoPE math from any weight-loading error in
Phase 2, so a mismatch there is unambiguously a RoPE bug.

---

## Phase 2: Import `Olmo-3-7B-Think-SFT` weights

**Files:** `scripts/import_hf_checkpoint.py`

No existing tool does this direction (`scripts/export_hf.py` goes the
*other* way — OLMo-core checkpoint → HF — via
`olmo_core.nn.hf.convert_checkpoint`, and only for OLMo-core's own model
classes, not this repo's `t0_training/model/` classes).

1. Load the HF checkpoint via
   `transformers.AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think-SFT")`
   purely to get its `state_dict()` and confirm shapes — this is the
   reference to validate against, not something kept around for training.
2. Write `map_hf_state_dict(hf_state_dict) -> dict[str, Tensor]` translating
   HF parameter names into `t0_training/model/transformer.py`'s naming
   (`embedding.weight`, `lm_head.weight`, per-layer `wq/wk/wv/wo`,
   `q_norm`/`k_norm`, `gate`/`up`/`down`, block-level norms). Confirm the
   exact HF key names directly from the loaded `state_dict().keys()` — don't
   hardcode names from memory; OLMo3's reordered-norm block naming is the
   most likely place to get this wrong silently.
3. Load the mapped dict into a fresh `Transformer(config_7b_yarn)` instance
   via `load_state_dict(strict=True)` — deliberately strict, so any
   missed/misnamed key fails loudly instead of silently leaving a layer at
   its random init.
4. Save the mapped state dict as a single flat file (plain `torch.save`, not
   `torch.distributed.checkpoint`) — see Phase 3 for why this is the right
   format for a seed checkpoint, not DCP's sharded format.

### Phase 2 validation — logit parity

Run identical inputs through both the HF model and the freshly-loaded
internal model and check they agree. This is the single most important
check in this whole plan — if it passes, Phases 1–2 are correct and
everything downstream is training on the right model.

#### What `--validate` checks

The `validate_logit_parity` function checks:

1. **Short prompt + 10k-token prompt** (the repo's own `docs/*.md`). A
   ~6-token prompt never hits the 4096 sliding window or goes past YaRN's
   original 8192 context, so it can't catch errors in either. The long
   input is reported over all positions, `>4096`, and `>8192`. The internal
   model's `max_seq_len` is raised for this run only (it only sizes the
   non-persistent RoPE tables, so the weights don't change).
2. **Pass criteria: top-1 (argmax) agreement ≥ 97% and mean per-position
   KL(HF ‖ internal) ≤ 5e-3**, not an elementwise `assert_close`. In bf16,
   max abs logit diffs are dominated by a few outlier positions: even
   HF bf16 vs HF fp32 has a max abs diff of ~6.7 on long inputs. So any
   `atol` is either too tight to pass a correct model or too loose to
   mean anything. (`--atol 0.5 --rtol 0.05` passes on the short prompt
   only.)
3. **Greedy generation** (40 tokens, chat-templated prompt) compared
   token-for-token. Reported, not asserted, because a single bf16 near-tie
   can legitimately flip one token.
4. **Sliding-window boundary: unit test, not part of `--validate`.** A
   window off by one changes 1 key out of 4096 and is indistinguishable
   from bf16 noise at the logit level (confirmed: see negative control B
   below), so it is tested directly in `tests/test_model.py`
   (`test_attention_sliding_window_boundary_matches_hf`). That test runs the
   real `Attention` module with `wq = wk = 0` (uniform attention), routes
   marker values on keys `q-w` and `q-w+1` through `wv`/`wo`, and asserts
   the first gets zero weight and the second gets `1/w`, matching HF's
   `kv_idx > q_idx - sliding_window` mask. It lives in the tests rather than
   the script because it checks `Attention`'s code, not the imported
   checkpoint, and needs only a GPU (milliseconds, no download). Changing
   `attention.py`'s window tuple to `(w, 0)` or `(w-2, 0)` makes it fail.

#### Open caveats

- `config_7b_yarn.max_seq_len` is still 4096 (TODO in `config.py`). If DPO
  sequences can exceed 4096, raise it. Validation already covers up to
  10k positions.
- Only batch size 1 without padding has been validated. Padded or packed
  batches are Phase 6's concern (see Phase 6 validation).

---

## Phase 3: Seeding vs. resuming — a deliberate design split

**Files:** `train/checkpoint.py` (`CheckpointManager` unchanged; new
standalone `load_seed_checkpoint(model, path)` helper alongside it, so
Phase 4's reference model can reuse it), `scripts/train_dpo.py`

**Decision: don't extend `CheckpointManager`.** Load the Phase 2 state dict
directly into the plain `nn.Module` with `model.load_state_dict(...)`
before `compile_model()` and `wrap_model_fsdp()` are called — then
`CheckpointManager` is free to do exactly what it already does, unmodified:
checkpoint and resume the DPO run itself, mid-training, same as pretraining.

Why: `CheckpointManager.load()` is built specifically for *resuming an
interrupted run* — it always loads model + optimizer state together via DCP
(sharded format), invoked via `latest_checkpoint()` under the current run's
own `save_dir`. Seeding is a different problem: it happens *before* FSDP
sharding even exists (Phase 2's flat, unsharded state dict), while DCP
resume happens *after* (each rank's own shard). Extending `CheckpointManager`
to support both would work, but solves two problems with one mechanism.

```python
model = Transformer(config_7b_yarn).cuda()  # fp32, as in scripts/train.py
load_seed_checkpoint(model, "checkpoints/olmo3-7b-think-sft-imported.pt")
model = compile_model(model)             # unchanged from the pretraining path
model = wrap_model_fsdp(model)           # unchanged from the pretraining path
optimizer = build_optimizer(model, ...)  # fresh optimizer, step 0 — this is a new run
```

`load_seed_checkpoint` will be
`model.load_state_dict(torch.load(path, mmap=True, map_location="cpu"), strict=True)`.
Three details matter:

- **Order: seed → compile → wrap.** `compile_model` replaces each block
  with a `torch.compile` wrapper, which renames its parameters
  (`blocks.0.attn.wq.weight` → `blocks.0._orig_mod.attn.wq.weight`), so
  the flat state dict only loads before compilation.
- **No `assign=True`** (unlike `scripts/import_hf_checkpoint.py`). The
  seed file is bf16, but `wrap_model_fsdp` requires fp32 params as the
  optimizer master. A plain `load_state_dict` copies the bf16 values into
  the existing fp32 params; `assign=True` would make them bf16 and the
  wrap would raise.
- **`mmap=True`.** Every rank loads the full ~14 GB file before sharding;
  memory-mapping lets ranks on a node share one page-cached copy instead
  of each holding its own.

When resuming an interrupted DPO run, `trainer.resume_from_latest()`
overwrites the seeded weights with the run's own DCP checkpoint, so seeding
first is harmless; `scripts/train_dpo.py` will skip it when
`latest_checkpoint()` finds one, to avoid the wasted load.

### Phase 3 validation

- CPU unit tests (tiny config): seeded params are fp32 and equal the bf16
  values from the file; a missing or misnamed key fails the strict load;
  a seeded model still compiles and runs forward.
- GPU test (single-process `torchrun`, like the existing FSDP tests):
  seed → compile → wrap, and check the wrapped model's logits match the
  unwrapped seeded model's on the same input.

---

## Phase 4: Reference model

**Files:** `scripts/train_dpo.py`, `train/dpo_trainer.py`

Build a second `Transformer(config_7b_yarn)`, load the *same* Phase 2 state
dict into it with Phase 3's `load_seed_checkpoint`, `requires_grad_(False)`, `.eval()`, then FSDP-wrap it too (for
memory sharding across ranks — at 7B there's no reason not to shard a model
just because it isn't training). No optimizer, no gradient accumulation, no
entry in `CheckpointManager` — it's a static, re-derivable artifact
(re-running Phase 2/3 reproduces it exactly), so it never needs its own
checkpoint.

All reference-model forward passes run inside `torch.no_grad()`.

---

## Phase 5: Preference-pair data pipeline

**Files:** `data/preference_dataset.py`

`Dolci-Think-DPO-7B`'s schema (confirmed from the dataset card): `prompt`
(string), `chosen`/`rejected` (each a list of chat-turn `{role, content}`
dicts) — a genuinely different data shape from pretraining's flat token
stream, so `NumpyDataset`/`ConcatNumpyDataset` don't apply here; this needs
its own dataset class.

**Use the `Olmo-3-7B-Think-SFT` tokenizer's own chat template** — not the
plain `allenai/dolma2-tokenizer` used elsewhere in this repo for
pretraining-corpus tokenization. The SFT checkpoint ships its own chat
formatting (and likely "think" special tokens); using the wrong template
would silently feed the model prompts in a format it's never seen.

For each row, build two token sequences — `prompt + chosen` and `prompt +
rejected` — and matching label arrays using the trainer's existing
`ignore_index=-1` convention (already wired into `trainer.py`'s loss
computation, no changes needed there): `-1` on every prompt-token position,
real token ids on every response-token position.

**Profile example lengths before picking `max_seq_len`** for the YaRN config
in Phase 1 — `Dolci-Think-DPO-7B`'s "Think" reasoning traces are likely a
few thousand tokens, not approaching the full 65,536 YaRN extends to.
Picking the smallest `max_seq_len` that comfortably covers the real
distribution keeps Phase 6's packed batches cheaper.

A plain shuffled `DistributedSampler` is enough here — at 150,000 pairs for
a single epoch, there's no need for `GlobalShuffleSampler`'s distributed-
epoch machinery, which solves a problem (hundreds of per-source shards
combining into one epoch) this dataset doesn't have.

---

## Phase 6: Variable-length attention (packing, no padding)

**Files:** `model/attention.py`

`Attention.forward()` currently calls plain `flash_attn_func(q, k, v,
causal=True, window_size=...)` — fixed-size batched, no padding-mask
support at all. Naively padding chosen/rejected sequences to a common
length within a batch would mean pad tokens silently participate in
attention (both attending and being attended to), with no way to mask them
out through this call signature — a real correctness bug, not a
hypothetical one.

**Fix: add an opt-in `cu_seqlens` code path using `flash_attn_varlen_func`**,
used only by the DPO data pipeline — the existing pretraining call site
(`flash_attn_func`, fixed-size batches) is untouched, zero regression risk.

```python
def forward(self, x, cos, sin, cu_seqlens: torch.Tensor | None = None, max_seqlen: int | None = None):
    ...
    if cu_seqlens is not None:
        from flash_attn import flash_attn_varlen_func  # noqa: PLC0415
        # x arrives pre-flattened: (total_tokens, d_model), no batch dim —
        # cu_seqlens marks example boundaries in that flat buffer.
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True, window_size=window_size_tuple,
        )
    else:
        out = flash_attn_func(q, k, v, causal=True, window_size=window_size_tuple)
```

Each chosen/rejected sequence occupies its own contiguous span in a packed
1-D buffer — no padding token ever exists, so there's nothing to mask
incorrectly. Phase 5's collator is responsible for producing
`(packed_tokens, packed_labels, cu_seqlens)` per batch.

### Phase 6 validation

Construct two short sequences, run them (a) separately through the existing
fixed-batch path and (b) packed together through the new varlen path;
assert the per-token logits match. This confirms packing doesn't leak
attention across the sequence boundary.

---

## Phase 7: DPO loss

**File:** `train/dpo_loss.py`

Given packed per-token log-probs for policy and reference on both chosen
and rejected (reusing the same `log_softmax` + gather machinery the z-loss
work already added to `trainer.py`), sum over each sequence's non-masked
(`labels != -1`) positions to get one scalar log-prob per sequence, then:

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta: float):
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_margin - ref_margin)
    return -F.logsigmoid(logits).mean()
```

`β` scales how strongly the model is pushed to separate chosen from
rejected relative to the (frozen) reference — this is where the implicit
KL-to-reference penalty lives.

**Hyperparameters, from OLMo3's own recipe** (paper Table 48, 7B Think
DPO): 1 epoch, `β=5`, LR=`8e-8`, linear decay, warmup ratio 0.1. Use these
as the default config rather than re-deriving them.

### Phase 7 validation

Unit test with synthetic log-probs: when policy == reference exactly (as at
step 0, right after seeding), `policy_margin == ref_margin`, so `logits=0`
and loss `= -logσ(0) = log(2) ≈ 0.693` for every example — a concrete,
checkable invariant for the very first training step, same spirit as the
original plan's "loss should start at ~log(vocab_size)" sanity check for
pretraining.

---

## Phase 8: Training loop

**File:** `train/dpo_trainer.py` (new, alongside `train/trainer.py` — not a
modification of it, same way pretraining's `Trainer` stays untouched)

**Per-step logic:**

```python
chosen_logp_policy   = sequence_logprob(policy_model, chosen_batch)       # grad
rejected_logp_policy = sequence_logprob(policy_model, rejected_batch)     # grad
with torch.no_grad():
    chosen_logp_ref   = sequence_logprob(reference_model, chosen_batch)
    rejected_logp_ref = sequence_logprob(reference_model, rejected_batch)

loss = dpo_loss(chosen_logp_policy, rejected_logp_policy,
                 chosen_logp_ref, rejected_logp_ref, beta=cfg.beta)
(loss / grad_accum_steps).backward()      # only policy has grad-tracked params
```

**Reused as-is** from the pretraining `Trainer`: LR scheduler,
`build_optimizer` (policy only — reference has no optimizer),
`CheckpointManager.save()` (policy only; the reference is never
checkpointed, per Phase 4), wandb logging.

**Adjust, don't reuse as-is:** the MFU calculation — each step now does
*two* forward passes (chosen + rejected) through *two* models (policy with
backward, reference without), so the existing `tokens/sec → MFU` formula
needs its FLOP-per-token estimate updated accordingly, or MFU will read
artificially low.

**New logging, specific to DPO:** the policy/reference margin difference
(`policy_margin - ref_margin`, the pre-sigmoid logit) and implied reward
accuracy (fraction of batch where `policy_margin > 0`) — standard DPO
health signals, not present in the pretraining trainer.

---

## Phase 9: Config & launch

**Files:** `t0_training/configs/config_dpo_7b.py`, `scripts/train_dpo.py`,
`batch/7b/train_dpo.sh`

Same shape as `config_7b.py`/`scripts/train.py`/`batch/7b/train_t0.sh`:

```python
RUN_CONFIG = DPORunConfig(
    model=config_7b_yarn,
    seed_checkpoint="checkpoints/olmo3-7b-think-sft-imported.pt",
    training=DPOTrainingConfig(
        beta=5.0,
        max_lr=8e-8,
        warmup_ratio=0.1,
        lr_schedule="linear",
        # max_steps left unset — one epoch of Dolci-Think-DPO-7B, same
        # auto-compute convention as scripts/train.py.
    ),
    dataset="allenai/Dolci-Think-DPO-7B",
)
```

`scripts/train_dpo.py` mirrors `scripts/train.py`'s structure
(`init_distributed()` → build policy + reference per Phases 2–4 → build
preference data loader per Phase 5 → `DPOTrainer(...).train()`), and
`batch/7b/train_dpo.sh` mirrors `batch/7b/train_t0.sh`'s Isambard/Slurm
launch pattern exactly (same NCCL/Slingshot env vars, same `submit.sh`
invocation convention).

---

## Hardware and memory budget (7B, FSDP-sharded)

Rough estimates, same style as the original plan's table — refine against
real OOM behavior the same way the pretraining 7B config already was (see
issue #29 / PR #36 for that precedent). Assuming 16 GPUs (4 nodes × 4
GH200, matching the existing `batch/7b/` node count):

| Component | Total (cluster) | Per GPU (÷16) |
|---|---|---|
| Policy params (bf16) | ~14 GB | ~0.9 GB |
| Policy grads (bf16) | ~14 GB | ~0.9 GB |
| Policy AdamW optimizer state (fp32 master + 2 moments) | ~84 GB | ~5.3 GB |
| Reference params (bf16, no grad/optimizer) | ~14 GB | ~0.9 GB |
| Activations (policy fwd+bwd + reference fwd-only, packed varlen batches) | highly batch/seqlen dependent | few GB, tune `rank_microbatch_tokens` empirically |
| **Total (excl. activations)** | — | **~8 GB** |

Comfortably inside GH200's 96 GB/GPU even before accounting for headroom —
the reference model's addition (~0.9 GB/GPU, no grad/optimizer cost) is
small relative to the policy's own optimizer state, which already dominates
the budget exactly as it did for plain pretraining.

---

## What This Doesn't Include (add later if needed)

- **RLVR** — a structurally different, larger undertaking (real generation/
  rollouts, a verifier pipeline, concurrent generation+training
  orchestration) — see the Section 2 discussion in
  [next_steps_nov2026.md](next_steps_nov2026.md) for why this is scoped
  separately and isn't a small extension of this plan.
- **LoRA / parameter-efficient fine-tuning** — full-parameter DPO only,
  matching OLMo3's own recipe; no new dependency needed.
- **Reference log-prob precomputation/caching** — recomputing the reference
  forward pass every step (Phase 8) is simpler to get correct first; since
  the reference never changes, its log-probs could be precomputed once
  up front to cut compute — a legitimate later optimization, not needed for
  correctness.
- **Multi-reference-model ensembling, mixed SFT+DPO data, dataset-mixing
  beyond the single `Dolci-Think-DPO-7B` dataset** — out of scope; OLMo3's
  own recipe doesn't do this either.
