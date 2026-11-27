# DPO Training Stack — Implementation Plan

## Overview

This plan covers adding DPO training to the existing from-scratch stack
(`t0_training/model,train,data,distributed,optim`, built per
[write_training_code.md](write_training_code.md)), starting from the
publicly released `allenai/Olmo-3-7B-Think-SFT` checkpoint and training on
`allenai/Dolci-Think-DPO-7B`. Same rationale as the original from-scratch
pretraining build: the value is in understanding the mechanics, not in the
output checkpoint (see [next_steps_nov2026.md](next_steps_nov2026.md)).

The reference implementation is AI2's `open-instruct`: the OLMo 3 7B Think
DPO launch script (`scripts/train/olmo3/7b_think_dpo.sh`) and the code it
runs (`open_instruct/dpo_tune_cache.py`, `open_instruct/dpo_utils.py`).
The paper (Table 48, A.6.2) is the authority. Where it doesn't specify a
detail, the plan names its source explicitly: the launch script, the code
at the run's recorded commit (`68da0a1`, per AI2's OLMo 3 README), or
`open-instruct`'s current `main`. These differ in places: the published
script post-dates the run and its code defaults changed between the two.

Every phase below either reuses existing `t0_training/` infrastructure
directly, or extends it additively (new opt-in code paths), so the existing
pretraining path (`scripts/train.py`, `config_3b.py`/`config_7b.py`) is
never touched.

### Key design choices

| Choice | Rationale |
|--------|-----------|
| **YaRN RoPE as a second code path** | Required to match `Olmo-3-7B-Think-SFT`'s real positional encoding — not optional, since its `attention_factor` scales attention at every position, not just beyond the original 8k context. As in HF's Olmo3, YaRN applies only to the 8 full-attention layers; the 24 sliding-window layers keep plain RoPE. |
| **Logit parity as the import gate** | The only reliable way to confirm the HF → internal weight mapping is correct before spending any compute training on it. |
| **Seeding ≠ resuming** | `CheckpointManager` stays untouched; a new flat-file loader seeds the policy before FSDP sharding, instead of stretching DCP resume to do a different job. |
| **New preference-pair dataset class** | `Dolci-Think-DPO-7B`'s `{prompt, chosen, rejected}` schema doesn't fit `NumpyDataset`/`ConcatNumpyDataset`'s flat-token-stream model at all. |
| **Right-padding, not packing** | With causal attention, right-padded batches give the same real-token outputs as running each sequence alone (up to bf16 kernel rounding), so no model changes are needed. Matches AI2's default for 7B Think DPO. |
| **Cached reference log-probs, no reference model** | As in `open-instruct`'s current DPO code (`main`): the seeded policy computes every pair's reference log-probs once, before the first update, and training looks them up by example index. A few MB instead of a second 7B model. |
| **Hyperparameters from the OLMo 3 paper** | Table 48 (7B Think DPO) is the authority: 150k pairs, 1 epoch, length-normalised DPO (`dpo_norm`), β=5, LR 8e-8, linear decay, warmup ratio 0.1, 128 pairs per step, max length 16K. The published launch script supplies what the table doesn't: weight decay 0 and gradient clipping at 1.0 (via its DeepSpeed config). `open-instruct`'s `main`-branch code defaults are not used as a source, since they differ from those at the run's recorded commit. |

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
| `max_seq_len` | effectively 65,536 (YaRN-extended) | `config_7b.max_seq_len=4096`. Raised to 16384 for `config_7b_yarn` (Phase 4), matching AI2's `--max_seq_length 16384`. |

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
    inv_freq = inv_freq_interpolated * ramp + inv_freq * (1 - ramp)

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
2. Write `map_hf_state_dict(hf_state_dict, n_layers, vocab_size) -> dict[str, Tensor]` translating
   HF parameter names into `t0_training/model/transformer.py`'s naming
   (`embedding.weight`, `lm_head.weight`, per-layer `wq/wk/wv/wo`,
   `q_norm`/`k_norm`, `gate`/`up`/`down`, block-level norms). Confirm the
   exact HF key names directly from the loaded `state_dict().keys()` — don't
   hardcode names from memory; OLMo3's reordered-norm block naming is the
   most likely place to get this wrong silently. The embedding and
   `lm_head` are zero-padded from the checkpoint's vocabulary (100,278) to
   the model's padded `vocab_size` (100,352); the padded rows are never
   indexed by a real token.
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

- `config_7b_yarn.max_seq_len` will be raised to 16384 in Phase 4.
  Validation already covers up to 10k positions.
- Only batch size 1 without padding has been validated. Padded batches
  are Phase 5's concern (see Phase 5 validation).

---

## Phase 3: Seeding vs. resuming — a deliberate design split

**Files:** `train/checkpoint.py` (`CheckpointManager` unchanged; new
standalone `load_seed_checkpoint(model, path)` helper alongside it).
The call from `scripts/train_dpo.py` is added in Phase 9.

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
  the flat state dict only loads before compilation. Phase 8 adds an
  activation-checkpoint wrap between seeding and compiling (seed →
  checkpoint-wrap → compile → wrap).
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

## Phase 4: Preference-pair data pipeline

**Files:** `data/preference_dataset.py`, `scripts/tokenize_dpo_data.py`,
`t0_training/model/config.py` (`config_7b_yarn.max_seq_len = 16384`, set
here rather than in Phase 9 because Phases 7 and 8 already run 16k
sequences, which would overflow the 4096-position RoPE tables)

`allenai/Dolci-Think-DPO-7B` (150,000 rows; the dataset card says it "was
used to preference tune Olmo 3 Think 7B") has `prompt` (string) plus
`chosen`/`rejected`, each a **full conversation**: a `[user, assistant]`
list of `{role, content}` messages, with the assistant content starting
with `<think> `. This is a different data shape from pretraining's flat
token stream, so `NumpyDataset`/`ConcatNumpyDataset` don't apply and the
pipeline gets its own dataset class.

**Sequences.** Each side is tokenized from its own message list as-is.
`chosen` and `rejected` already contain the user turn, so prepending
`prompt` would include it twice. As a consistency check, `prompt` must
equal the user turn's content in both lists; a mismatch is an error.

**Chat template.** Use the template shipped with `Olmo-3-7B-Think-SFT`'s
tokenizer (its `tokenizer_config.json`), not the plain
`allenai/dolma2-tokenizer` used for pretraining data. Dolci rows have no
system message, so the template inserts its default system prompt ("You
are Olmo, a helpful AI assistant built by Ai2…"). This deliberately
differs from `open-instruct`'s `olmo_thinker` template, whose default
system prompt differs between AI2's recorded run commit and `main`; the
shipped template is the one the released model is used with.

**Labels.** Only the final assistant turn is trainable (Dolci is
single-turn, so this is the only assistant turn), and only the part the
model generates itself.
The template's generation prompt ends with the opening think tag
(`<|im_start|>assistant\n<think>`), so at inference the model starts
writing *after* `<think>`. Everything up to and including that generation
prompt is masked, which covers the 3 `<think>` tokens (`<th`, `ink`, `>`).
Trainable: the reasoning, `</think>`, the final answer, and the closing
`<|endoftext|>` (EOS, id 100257), since that terminator is what teaches the
model when to stop. (Earlier assistant turns end with `<|im_end|>`; the
final one ends with EOS.) Every other position gets `-1`, the trainer's
existing `ignore_index`. Labels are stored aligned with the input tokens
(label *t* is token *t* or `-1`), not pre-shifted as the pretraining
loader's are; the shift happens once, in Phase 6. The `dpo_norm` denominator (`|y|` in the paper's
A.6.2 loss) is therefore the count of generated tokens only. This matches
AI2's masking at the run's recorded commit, where
`sft_tulu_tokenize_and_truncate_v1` masks everything up to the
generation prompt, and its `olmo_thinker` generation prompt also ends in
`<think>`.

**Length.** `max_seq_len = 16384` per side, matching AI2's
`--max_seq_length 16384`. The tokenization script reports the length
distribution of chosen, rejected, and the chosen–rejected length
difference, which sizes Phase 5's padding cost. Sides longer than 16384
tokens are **right-truncated and kept**, as AI2's code does at the run's
recorded commit (`sft_tulu_tokenize_and_truncate_v1` with
`truncation=True, max_length=max_seq_length`; the tokenizer's
`truncation_side` is `right`). Each over-long side keeps its first 16384
tokens; the cut-off part (typically the final answer and closing EOS) is
lost. A small share of pairs (~2.7% of Dolci) has at least one side over
16384, mostly long math and science traces; the report states the exact
count and how many truncated sides are chosen vs rejected.

**Filter.** Drop any pair where either side has no trainable tokens
after truncation (AI2's `preference_tulu_filter_v1`). This is the only
filter AI2 applies, so over-long pairs are not removed.

One addition of our own: drop any pair whose **chosen** assistant message contains
the literal EOS text (`tokenizer.eos_token`, `<|endoftext|>`) inside its
content. The text itself is legitimate (code and reasoning that
mention GPT-2-style tokenizers), but the tokenizer turns the string into
the real EOS token mid-sequence, and training would then teach the model
to emit its stop signal in the middle of a good response. Tokenizing
message content with special-token strings kept as ordinary text would
avoid the drop, but isn't worth the extra complexity for one pair. A literal EOS inside the **rejected** response
is kept: DPO pushes probability away from the rejected response, so it
does no harm there. The tokenization report states how many pairs this
drops (2 rows in Dolci contain the literal text, one on each side, so this
drops 1 pair).

**Tokenize once to disk.** `scripts/tokenize_dpo_data.py` writes the
tokenized pairs (token ids, labels) and the length report to disk. Dolci has no
index column (only a string `id`), so, as in `open-instruct`, each pair
gets an `index` (`0..N-1`) assigned after filtering, which is the key
Phase 7's reference cache is looked up by. The original `id` is kept
alongside it. Training reads only these files, and the tokenized
dataset's identity feeds Phase 7's cache key. (The `index` is this
plan's own; AI2's code at the run's recorded commit had no index and
looked the cache up by step position.)

**Real vocabulary size.** The tokenizer's own `len(tokenizer)` (the
unpadded vocabulary, equal to the HF model's `config.vocab_size`) is
recorded with the tokenized data, for Phase 6's logit slice. No hard-coded
value.

**Sampling.** PyTorch's shuffled `DistributedSampler`, seeded (seed 42),
as AI2 uses a shuffled sampler. It has no mid-epoch start, so resuming
skips the micro-batches already consumed in the epoch's seeded order (as
AI2 does with `accelerate`'s `skip_first_batches`; see Phase 8).

**Padding token.** Padding uses `tokenizer.pad_token_id` (`<|pad|>`),
which is below `real_vocab`, with label `-1`.

### Phase 4 validation

- Rendering a sample of rows with the shipped template gives the expected
  structure: default system prompt, user turn, assistant turn ending in
  EOS.
- Unit tests on hand-built rows: the prompt appears once; labels are `-1`
  everywhere except the final assistant turn after its opening `<think>`
  (reasoning, `</think>`, answer, closing EOS);
  over-long sides are right-truncated to 16384 and kept; pairs with no
  trainable tokens after truncation are dropped; a pair with a literal
  EOS inside the chosen content is dropped, while one inside the rejected
  content is kept; `index` runs `0..N-1` over
  the kept pairs.

---

## Phase 5: Right-padded batches

**Files:** collator in `data/preference_dataset.py`. No model changes.

Each micro-step processes one pair as a batch of 2 rows (chosen,
rejected), with the shorter one **right**-padded to the longer and pad
labels set to `-1`. This matches AI2's default (non-packed) path for
7B Think DPO.

No attention mask is needed. Attention is causal, so the real tokens,
which come first, never attend to the trailing pads. Pad positions do
attend to real tokens, but their outputs are excluded from the loss. Each
row starts at position 0, so RoPE positions are correct. Real-token
outputs are therefore what each sequence would give on its own, up to
bf16 kernel rounding (a batch of 2 can select different matmul kernels
than a batch of 1).
(Left-padding would break this: pads would come before real tokens and be
attended to, and positions would shift.)

Packing (`flash_attn_varlen_func` with `cu_seqlens`) is not used. It would
need `cu_seqlens` threaded through `Transformer → Block → Attention`, plus
RoPE positions that restart for each packed sequence (`apply_rotary`
currently assumes positions `0..T-1` across the whole row). The padding
cost is measured in Phase 4 and logged in Phase 8, which shows whether
packing would be worth adding later.

### Phase 5 validation

Two sequences of different lengths, run each alone, then together
right-padded in one batch: the real-token logits match within a small
tolerance. A left-padded
version of the same batch is expected to differ, confirming the test can
tell the two apart.

---

## Phase 6: Log-probs and DPO loss

**File:** `train/dpo_loss.py`

**Per-token log-probs** via `log_softmax` + `gather` on fp32 logits,
wrapped in `torch.compile`, as `open-instruct` does
(`model_utils.log_softmax_and_gather`, `@torch.compile(dynamic=True)`,
introduced in open-instruct PR #584):

```python
@torch.compile(dynamic=True)
def log_softmax_and_gather(logits, targets):
    return torch.gather(logits.log_softmax(-1), -1, targets.unsqueeze(-1)).squeeze(-1)

logits = logits[:, :-1, :real_vocab].float()  # t predicts t+1; drop padded vocab rows
targets = labels[:, 1:]
mask = targets != -1
token_logp = log_softmax_and_gather(logits, targets.clamp(min=0))
token_logp = token_logp * mask                # ignored positions contribute 0
```

The model's logits come out in bf16 (FSDP's `param_dtype`), and the cast
to fp32 before `log_softmax` matters: at LR 8e-8, the policy–reference
differences are tiny, and bf16 log-probs would round them away.

**Why compiled.** Run eagerly, `log_softmax` creates the full fp32
`positions × vocab` log-prob tensor and autograd keeps it for the backward
pass. Compiled, the two operations fuse and that tensor is never created.
The slice and fp32 cast happen before the compiled function, as in
`open-instruct` (`.logits.to(torch.float32)` in `dpo_utils.py`), so the
full fp32 copy of the logits is still kept for the backward pass, along
with the bf16 logits and the gradient. Measured on one GPU at 2 × 4k
tokens and scaled to a 2 × 16k pair, this part peaks at ~39 GB, versus
~46 GB fully eager, with the same values. `dynamic=True` avoids a
recompile for every sequence length. (`F.cross_entropy` saves the same
tensors as eager `log_softmax`, so it offers no memory advantage.)

**Slice to the real vocabulary first.** The imported `lm_head` is padded
from the tokenizer's vocabulary to the model's padded `vocab_size` with
zero rows (Phase 2). Each padding row produces a logit of exactly 0, which
would enter the softmax and lower every log-prob, and would also receive
gradients and drift during training. `real_vocab = len(tokenizer)`
(recorded with the tokenized data in Phase 4) is checked to be no larger
than the model's `vocab_size`, and the logits are sliced to it before
`log_softmax`, matching AI2's unpadded 100,278-way softmax.

**Per-sequence values.** For each sequence, the summed log-prob over
trainable positions and the number of trainable tokens. Both loss types
come from these two numbers:

```python
def sequence_logp(logp_sum, n_tokens, loss_type):
    return logp_sum / n_tokens if loss_type == "dpo_norm" else logp_sum

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
             beta: float, label_smoothing: float = 0.0):
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
              - F.logsigmoid(-beta * logits) * label_smoothing)
    chosen_rewards = beta * (policy_chosen - ref_chosen).detach()
    rejected_rewards = beta * (policy_rejected - ref_rejected).detach()
    return losses.mean(), chosen_rewards, rejected_rewards
```

Defaults follow the paper (Table 48, A.6.2): `loss_type="dpo_norm"`
(length-normalised, i.e. per-token average log-prob) and `beta=5.0`;
`label_smoothing=0.0`. Plain summed `"dpo"` is selectable by config. β=5
goes with the averaged form: with summed log-probs, the implicit reward
`β·(log πθ − log πref)` moves roughly |y| times faster per update (|y| in
the thousands for Think traces), so the same β would mean a far stronger
effective step and KL trade-off.

**Metrics**, defined as `open-instruct` defines them: chosen rewards,
rejected rewards, reward margin (`chosen_rewards − rejected_rewards`), and
reward accuracy, the fraction of pairs where `chosen_rewards >
rejected_rewards` (i.e. `policy_margin − ref_margin > 0`). Not
`policy_margin > 0`, which reflects the SFT model's existing preference
rather than what DPO has learned. At step 0, the rewards are only the
tiny no-grad-vs-training differences described in Phase 7, of either sign,
or exactly 0 where the two passes agree bit for bit; ties count as wrong
under the strict `>`, so step-0 accuracy is anywhere between 0 and about
0.5. Tests don't assert a step-0 accuracy. Also logged, matching
`open-instruct`'s names so runs can be compared with AI2's: `logps/chosen`
and `logps/rejected` (the policy's per-sequence values), `rewards/average`,
and `token_count` (trainable tokens), named as on `open-instruct`'s
`main`. `logps/*` log the per-sequence value the loss uses (the per-token
average under `dpo_norm`), as `open-instruct` does. Every metric is
averaged over all pairs in the optimizer step, across micro-steps and
ranks. The values aren't directly comparable with AI2's logs: the system
prompt differs (Phase 4), which changes the conditioning of every
log-prob.

**Hyperparameter source:** OLMo 3 paper Table 48 (7B Think DPO).

### Phase 6 validation

- When policy and reference log-probs are identical tensors, loss is
  exactly ln 2 ≈ 0.693 and rewards are 0 for every pair.
- A hand-calculated example for both `dpo` and `dpo_norm`, which catches
  averaging over the wrong token count.
- Token log-probs are computed in fp32 from bf16 logits, and positions
  labelled `-1` contribute nothing to the sums or token counts.
- The compiled `log_softmax_and_gather` matches the eager version on a
  small random case (values and gradients, within rounding), including
  across different sequence lengths (no recompile error with
  `dynamic=True`).
- Padded vocab rows are excluded: with `real_vocab < vocab_size`, the
  log-probs equal those of a model whose `lm_head` has only `real_vocab`
  rows.

---

## Phase 7: Reference log-prob cache

**Files:** `train/reference_cache.py`, called from `scripts/train_dpo.py`

There is no separate reference model. The design follows
`open-instruct`'s current `build_reference_logprobs_cache` (`main`); at
the run's recorded commit, `get_cache_ref_logprobs` instead kept the cache
in memory in dataloader order and recomputed it on every launch: before the first optimizer step, the
seeded policy itself (which at step 0 *is* the reference) runs once over
the whole training set in eval mode under `torch.no_grad()`. Each pair's
reference values are stored and looked up during training by the pair's
`index`.

- **Same model, same kernels.** The cache is computed with the seeded,
  activation-checkpointed, compiled, FSDP-wrapped policy, so at step 0 the
  policy's log-probs match the cached ones and the first loss is ln 2.
  Match, not bit-identical: the cache pass runs without gradients and
  training with them, and `torch.compile` builds separate graphs for the
  two, which can differ in the last bf16 digits. Checks use a small
  tolerance.
- **Coverage.** The cache pass iterates over every index exactly once:
  each rank takes a contiguous share in index order, no shuffling and no
  `drop_last`, so the leftover pairs that training skips are cached too.
- **Contents.** Per pair index: chosen and rejected summed log-prob (fp32)
  and trainable-token count (int). Either loss type can be computed from the same
  cache. ~150k × 4 values, a few MB.
- **Distributed build.** Each rank fills its share of indices into tensors
  initialised to a sentinel (`-inf` for log-probs, `-1` for counts); an
  all-reduce (max) combines them, and any index still holding a sentinel
  is an error.
- **Saved to disk, keyed.** Saved with `torch.save`, with a filename that
  hashes the seed checkpoint's identity, the tokenized dataset's identity
  and `max_seq_len`. An existing file with the same key is reused.
- **Built only at step 0.** With no training checkpoint: build (or load)
  the cache, then train. When resuming and the cache file is missing:
  raise an error. The cache is never computed from a partly trained
  policy. (At the run's recorded commit, AI2's code rebuilt the cache on
  every launch after loading the resumed state, so any resumed run
  computed its reference from the partly trained policy; on `main` this
  happens if the cache file is missing.)
- **Rebuilding from the original model.** `scripts/train_dpo.py
  --build-reference-cache-only` seeds from the seed checkpoint (never from
  a training checkpoint), writes the cache and exits. The seed checkpoint
  is never written to (training writes only DCP checkpoints under the
  run's `save_dir`), so the cache can always be recomputed from the
  original model.

### Phase 7 validation

- On a tiny model and dataset across 2 ranks (`torchrun`), the cache
  matches per-pair log-probs computed directly, for every index, with no
  sentinel left.
- At step 0, the policy's log-probs on a batch match the cached values
  within a small tolerance, so the loss is ≈ ln 2 (within tolerance).
- Resuming with no cache file raises; `--build-reference-cache-only` reads
  the seed file, not a training checkpoint.

---

## Phase 8: Training loop

**Files:** `train/dpo_trainer.py` (new `DPOTrainer`, alongside the
pretraining `Trainer`, which stays untouched), `train/scheduler.py`,
`distributed/fsdp.py` (activation checkpointing helper, and an opt-in
`dynamic` argument for `compile_model`, so pretraining's compile is
unchanged). Shared with pretraining: `CheckpointManager` and the scheduler
module. `DPOTrainer` has its own logging and wandb setup, modelled on
`Trainer._log`/`_wandb_init`, since those are `Trainer` methods and
`Trainer` stays untouched.

**Per micro-step:**

```python
logits = policy(batch.tokens)                        # (2, T, V): chosen, rejected
logp_sum, n_tokens = sequence_logp_sums(logits, batch.labels)
ref_sum, ref_n = reference_cache[batch.index]
loss, chosen_rewards, rejected_rewards = dpo_loss(...)   # per cfg.loss_type
(loss / grad_accum_steps).backward()
```

**Batch size.** 128 pairs per optimizer step, as in AI2's run (1 pair per
GPU × 4 accumulation steps × 32 GPUs). Each micro-step is 1 pair per GPU,
and `grad_accum_steps = global_batch_pairs / (world_size ×
microbatch_pairs)` (16 GPUs gives 8), which must divide exactly. Each pair is weighted equally.

**Final partial batch.** As in pretraining (`scripts/train.py`'s
`max_steps = total_tokens // global_batch_size`, with the loader's
`drop_last=True`), only full batches become steps. As in pretraining,
`max_steps` defaults to `None` and is then computed as
`n_pairs // global_batch_pairs` (one epoch); setting it explicitly in the
config overrides this, e.g. for short test runs. The leftover (< 128
pairs) is unused; which pairs they are is set by the seeded shuffle. The
LR schedule's warmup and decay use the same `max_steps`. This deliberately
differs from AI2, which rounds up (`math.ceil(len(train_dataloader) /
gradient_accumulation_steps)`: 1,172 steps, with `accelerate` repeating a
few pairs to even out the ranks), to stay consistent with pretraining.

**Optimizer**, selected by config:
- `optimizer="adamw"` (default): `torch.optim.AdamW`, fused, with
  PyTorch's default eps.
- `optimizer="skip_step_adamw"`: the repo's `SkipStepAdamW` via
  `build_optimizer`, with the trainer setting `latest_loss` and
  `latest_grad_norm` before each step.

Both take weight decay and `betas` from config: `betas` defaults to
(0.9, 0.999), PyTorch's default and so AI2's (`open-instruct` sets none),
and is used by both options, so switching to `skip_step_adamw` changes
only the step skipping. Weight decay defaults to 0.0 (AI2).

**Gradient clipping.** `grad_clip: float | None`, default `1.0`. AI2's
launch script sets no clipping flag (`clip_grad_norm` at the run's
recorded commit, default -1; `max_grad_norm` on `main`), but runs
DeepSpeed with `"gradient_clipping": "auto"`, which `accelerate`'s code
fills in as 1.0 (not verified against AI2's actual environment).
`None` turns clipping off. The gradient norm (before clipping) is logged
every step either way.

**LR schedule.** A linear-decay option added to `train/scheduler.py`:
linear warmup from 0 to `max_lr` over `int(warmup_ratio × max_steps)`
steps (as AI2 rounds it), then
linear decay to 0. The cosine path stays unchanged. Defaults:
`max_lr=8e-8`, `warmup_ratio=0.1`.

**Activation checkpointing.** Opt-in, wrapping every `TransformerBlock`
with PyTorch's `checkpoint_wrapper`. Without it, one 16k-token pair needs
~150 GB of activations per GPU, and FSDP doesn't shard activations, so
more GPUs wouldn't help. Order: build → seed → checkpoint-wrap → compile →
FSDP wrap (the same order as OLMo-core's train module). Unlike
`torch.compile`, the wrapper leaves `state_dict()` keys unchanged (it
hooks them to strip its `_checkpoint_wrapped_module.` prefix), so it
needs no change to `load_seed_checkpoint`. Off by default, so pretraining
is unaffected.

**Variable sequence lengths under `torch.compile`.** Every micro-step has
a different padded length, and the cache pass (no grad) and training
(grad) compile separately. The compiled blocks are set up for dynamic
sequence length (`dynamic=True`, or marking the sequence dimension
dynamic) so they don't recompile at every new length.

**Gradient reduction** stays in fp32 (`wrap_model_fsdp`'s
`reduce_dtype`). AI2's DeepSpeed bf16 ZeRO-3 setup reduced gradients in
bf16 by default; fp32 is at least as precise, so this difference is
accepted.

**Logging, every step:** loss, chosen/rejected rewards, reward margin,
reward accuracy and the other Phase 6 metrics, gradient norm and LR. Tokens are
counted from each actual batch, since the pretraining `Trainer`'s fixed
`tokens_per_step` doesn't apply:
- tokens/sec for all computed tokens (including padding) and for real
  tokens, so the cost of padding is visible;
- MFU from computed tokens, with FLOPs per token = 6 × params plus an
  attention term (attention adds roughly 20% at 16k tokens for this
  model, which the pretraining formula leaves out).

**Checkpointing and resume.** `CheckpointManager` saves the policy, the
optimizer and the `DPOTrainer` state (step, epoch, micro-batches consumed,
RNG). On resume: seeding is skipped, the policy and optimizer load from
the latest checkpoint, the reference cache is loaded (never rebuilt;
Phase 7), and the loader re-creates the epoch's seeded
`DistributedSampler` order and skips the micro-batches already consumed.

### Phase 8 validation

- Tiny model, a few steps: the step-0 loss is ≈ ln 2 (within tolerance);
  on a tiny set, the
  loss falls and reward accuracy rises above 0.5.
- An activation-checkpointed model gives the same loss and gradients as a
  non-checkpointed one (within bf16 noise).
- The linear schedule reaches `max_lr` at the end of warmup and 0 at
  `max_steps`.
- An interrupted-and-resumed run matches an uninterrupted one, following
  the existing pattern in `tests/test_checkpoint.py`.

---

## Phase 9: Config & launch

**Files:** `t0_training/configs/config_dpo_7b.py`, `scripts/train_dpo.py`,
`batch/7b/train_dpo.sh`. `DPORunConfig` lives in `t0_training/configs/base.py`
next to `RunConfig`; `DPOTrainingConfig` in `train/dpo_trainer.py`, next to
`DPOTrainer`, as `TrainingConfig` is in `train/trainer.py`.

```python
RUN_CONFIG = DPORunConfig(
    model=config_7b_yarn,                      # max_seq_len=16384 (set in Phase 4)
    seed_checkpoint="checkpoints/olmo3-7b-think-sft-imported.pt",
    tokenized_data=...,                        # Phase 4 output directory
    reference_cache_dir=...,                   # Phase 7; filename from the cache key
    training=DPOTrainingConfig(
        loss_type="dpo_norm", beta=5.0, label_smoothing=0.0,
        optimizer="adamw", betas=(0.9, 0.999), max_lr=8e-8, weight_decay=0.0,
        grad_clip=1.0, lr_schedule="linear", warmup_ratio=0.1,
        global_batch_pairs=128, microbatch_pairs=1,   # accumulation = 128 / (world_size × microbatch_pairs)
        max_steps=None,                        # None = one epoch: n_pairs // global_batch_pairs
        activation_checkpointing=True,
        seed=42,
        save_dir=..., save_interval=..., keep_last_n_checkpoints=3,
        log_interval=..., wandb_project=...,
        peak_flops_per_gpu=...,                # GH200 bf16 peak, for MFU
    ),
)
```

With `max_steps=None`, it's `n_pairs // global_batch_pairs` (Phase 8),
from the filtered dataset size:
about 149,999 // 128 = 1,171 steps (Phase 4 drops only pairs left with no
trainable tokens and pairs with a literal EOS in the chosen response).

`scripts/train_dpo.py` follows `scripts/train.py`'s structure:
`init_distributed()` → build the policy (seed unless resuming →
checkpoint-wrap → compile → FSDP wrap) → load the tokenized data and
loader → reference cache (build at step 0, load, or error, per Phase 7) →
optimizer → `DPOTrainer(...).resume_from_latest()` → `train()`. It also
takes `--build-reference-cache-only`.

`batch/7b/train_dpo.sh` mirrors `batch/7b/train_t0.sh`'s Isambard/Slurm
launch pattern (same NCCL/Slingshot env vars, same `submit.sh`
convention), on 16 GPUs (4 nodes × 4 GH200), giving 8 accumulation steps.

---

## Hardware and memory budget (7B, FSDP-sharded)

Rough estimates, same style as the original plan's table. These get
refined against real memory use the same way the pretraining 7B config
was (see issue #29 / PR #36 for that precedent). 16 GPUs (4 nodes × 4
GH200), one 16k-token pair per micro-step, activation checkpointing on:

| Component | Total (cluster) | Per GPU (÷16) |
|---|---|---|
| Policy params (fp32, sharded) | ~29 GB | ~1.8 GB |
| Policy grads (fp32, sharded) | ~29 GB | ~1.8 GB |
| AdamW moments (2 × fp32, sharded) | ~58 GB | ~3.6 GB |
| Weights gathered temporarily by FSDP (a couple of blocks + embedding/lm_head, bf16) | — | ~2–3 GB |
| Activations with checkpointing (block inputs for 2 × 16k sequences + one block recomputed) | — | ~13 GB |
| Logits (bf16 and fp32 copy) and their gradient, with the compiled log-softmax + gather (2 × 16k sequences × 100k vocab) | — | ~39 GB |
| Reference log-prob cache | ~MB | negligible |
| **Total** | — | **~63 GB** of 96 GB, plus communication buffers and fragmentation |

FSDP shards weights, gradients and optimizer state, but not activations:
those depend on the tokens each GPU processes at once (one pair, up to
16k tokens per side), whatever the GPU count. Without activation
checkpointing, activations alone would be ~150 GB per GPU, which doesn't
fit at any GPU count. With it, 16 GPUs fits, and more GPUs would only
shorten wall-clock time. The log-prob computation is the largest single
item, and its figure is extrapolated linearly from a 4k-token
measurement. If a real run shows memory pressure, there are two
fallbacks. First, move the slice and fp32 cast inside the compiled
function, so no full fp32 copy of the logits is kept (~26 GB for this
part, ~50 GB total; same values). Second, compute log-probs in chunks
along the sequence *including* `lm_head` (logits produced block by block
from the final hidden states, each block recomputed in the backward
pass). Chunking only the log-softmax, after the full logits exist, saves
little, because the full logits and their gradient dominate.

---

## What This Doesn't Include (add later if needed)

- **RLVR** — a structurally different, larger undertaking (real generation/
  rollouts, a verifier pipeline, concurrent generation+training
  orchestration) — see the Section 2 discussion in
  [next_steps_nov2026.md](next_steps_nov2026.md) for why this is scoped
  separately and isn't a small extension of this plan.
- **LoRA / parameter-efficient fine-tuning** — full-parameter DPO only,
  matching OLMo3's own recipe; no new dependency needed.
- **Packing (varlen attention)** — a later optimisation if the padding cost
  measured in Phases 4 and 8 turns out to be large. It needs `cu_seqlens`
  through the model and RoPE positions that restart for each sequence.
- **Evaluation and HF export of the DPO model** — handled separately after
  training. Export would be the reverse of Phase 2's mapping (rename keys,
  strip the vocab padding), checked with the same parity test in the other
  direction, so the model can be compared against AI2's released
  `Olmo-3-7B-Think-DPO`.
- **Multi-reference-model ensembling, mixed SFT+DPO data, dataset-mixing
  beyond the single `Dolci-Think-DPO-7B` dataset** — out of scope; OLMo3's
  own recipe doesn't do this either.
