# Long-Context Extension: 2,048 → 8,192 Comparison Experiment

This plan focuses on building hands-on understanding of long-context extension. 

This plan covers the first, small-scale step: extending 3B from 2,048 to 8,192 tokens, and running a controlled comparison against
a model pretrained natively at 8,192. 
Midtraining is explicitly **skipped** for this project (see rationale below) — we go straight from pretraining to long-context extension.

The eventual target is 65,536 tokens (matching OLMo3), reached via a second, larger jump.
That is **out of scope for this document** — this plan only covers the 2,048→8,192 step,
which doubles as a shakedown of the RoPE-scaling implementation before committing it to the
harder, more expensive 65,536 jump.

---

## Why skip midtraining

Midtraining is mechanically continued pretraining (same architecture, same loss, just a
different curated data mix + linear LR decay instead of cosine). Its two
genuinely separable lessons are (a) data-mix/domain-tradeoff effects, and (b) model souping.
Neither is unique to a dedicated midtraining stage: (a) is cheaply testable via small
standalone microanneal experiments, decoupled from the main model lineage, and (b) doesn't
show a gain at the 7B scale in OLMo3's own ablations (footnote, §3.5.4: "Initial
experimentation for the 7B model did not show similar gains from model merging"), so it's
unlikely to be worth it at our smaller 3B/7B scale either. We're deliberately not doing
souping in this experiment.

---

## Why compare against a native-8,192 pretrain

Reference: [issue #34](https://github.com/alan-turing-institute/t0-training/issues/34)
documents that our pretraining config diverges from OLMo3's in several ways relevant here:

| Parameter | OLMo3 7B | Our config (`configs/olmo3-{3B,7B}.yaml`) |
|---|---|---|
| Sequence length | 8,192 | **2,048** |
| Peak LR | 3.0e-4 | **1.0e-3** |
| RoPE scaling | YaRN on full-attn layers | Not implemented |
| SWA window vs seq len | window (4,096) < seq len (8,192) — real windowing exercised | window (4,096) > seq len (2,048) — SWA layers behave as full attention |

Because our pretrain sequence length (2,048) is shorter than the SWA window (4,096), our
model has never exercised real sliding-window masking. Extending straight to 8,192 will be
the first time it does — a genuinely new behavior introduced at the same time as everything
else in this stage. Rather than reason about whether that matters, we're running it as an
experiment: pretrain a second 3B model natively at 8,192 (where windowing *is* exercised
from the start, matching OLMo3's own setup) and compare it against the extended checkpoint.

**The peak LR (1e-3) will be left as is.** Since we already checked the full step-by-step loss/grad-norm
history of the existing 3B pretrain — no
spikes, smooth convergence, sane final metrics (CE loss 2.39, Hellaswag len-norm acc 0.586).
1e-3 is a demonstrated-stable choice for this model, just different from OLMo3's, so we're
keeping it for both arms below rather than introducing an LR change as a second confound.

---

## Confound to avoid: data exposure, not just context length

Naively, "native-8,192 pretrain" vs. "2,048-pretrain-then-extended" differ in **two** ways:
whether the model was ever trained at native long context, *and* whether it was ever exposed
to Dolma3 Longmino's curated long documents (used in the extension recipe, §3.6.3). If the
extended model does better on long-context evals, that could be RoPE/windowing adaptation
working — or just data exposure the native model never got. To isolate the actual variable
of interest, **the native-8,192 checkpoint also gets the same post-training stage** (same
data mix, same token budget, same recipe) before final comparison. Total tokens and data
composition end up matched across both arms; only *how* the long-document exposure was
delivered (native from-scratch vs. post-hoc addition) differs.

### Checkpoint already available

1. **run2-olmo**: 3B pretrained at 2,048 using olmo-core. 

### Three checkpoints to produce

1. **run3-olmo**: pretrain 3B at 8,192 from scratch (fresh run, new config) using olmo-core.
2. **run2-olmo-8192-extension**: take the existing 2,048-native 3B checkpoint (`run2-olmo`), run
   the long-context post-training recipe below to reach 8,192.
3. **run3-olmo-longmino**: take Arm A's checkpoint, run the *same* post-training recipe (§ below) on it
   too, even though it's already at 8,192 — this equalizes total tokens/data exposure.

Final comparison is run2-olmo-8192-extension vs.run3-olmo-longmino.

run3-olmo before the longmino post-training stage can also be kept as a
secondary reference point to see how much the post-training stage moves an already-native
model, but isn't the primary comparison target.

---

## run3-olmo: Native 8,192 pretrain

New config, otherwise identical to the existing 3B recipe (same peak LR 1e-3, same 60B
token budget, same data mix) — only `sequence_length` changes.

Config should be `configs/olmo3-3B-8192.yaml`. It should copy `configs/olmo3-3B.yaml` and change ONLY:
```yaml
sequence_length: 8192
```

Implication: batch tokens stay at 262,144, so instances/step drops from 128 (at 2,048) to 32. 
This is the same batch/instance question raised for run2-olmo-8192-extension below.
Keep batch tokens unchanged (32 instances/step) for consistency with run2-olmo; this
also means the two models share the same instance count once both reach 8,192, removing
batch-size as a confound between them.

No RoPE scaling needed — it's native pretraining at 8,192, not extension.

```bash
uv run --no-sync t0-train configs/olmo3-3B-8192.yaml --run-name test-3B-8192 --dry-run
./batch/submit.sh run3-olmo batch/3b/train_clean_8192.sh   # new script, copy of train_clean.sh pointed at the new config
```

Expect similar wall-clock/GPU-hours to the existing 3B run (~1,000 GPU-hours) — same token
count, same model size; throughput may be somewhat lower due to attention cost at 8,192 vs
2,048, but not dramatically (well below the regime where this becomes a problem — that's
the 65,536 stage, not this one).

---

## run2-olmo-8192-extension: Long-context post-training, 2,048 -> 8,192

Starting checkpoint: `run2-olmo` final checkpoint (2,048-native, 60B tokens, pretraining peak
LR 1e-3 — see Learning Rate below for this stage's own, reduced, peak).

### Data mix (66% short-context / 34% long-context, per OLMo3 §3.6.3)

Short-context portion is sampled from **our own pretraining mix**, not the midtraining
mix.
Keeping the short-context portion in-distribution with pretraining isolates the experiment to the
long-context-specific changes only.

Long-context portion uses `OLMo-longmino-mix-0625.txt` — confirmed via file header comment
to be "used to for Olmo3 7B Long Context extension" (i.e. the real mix OLMo3 itself used,
50B-token pool).

```bash
# Short-context: fresh sample from the pretraining pool, different seed from the original
# pretrain run to avoid pure repetition (t0-submix defaults to seed=42; use a different one)
uv run --no-sync t0-submix --target-tokens 6.6e9 --seed 7 \
    --output data/mixes/dolma3-lc-short-6.6B.txt

# Long-context: sample from the actual OLMo3 7B longmino mix (50B token pool)
uv run --no-sync t0-submix --target-tokens 3.4e9 \
    --mix-file .venv/lib/python3.*/site-packages/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt \
    --total-tokens 5.0e10 \
    --output data/mixes/dolma3-lc-long-3.4B.txt
```

**Resolved**: `OLMo-longmino-mix-0625.txt` labels every entry `longmino` — it does not
expose Table 11's length-bucket breakdown (8K-16K, 16K-32K, etc.) as separate labels, so
`t0-submix`'s proportional-by-label sampling can't specifically target the 8K-16K bucket as
originally hoped. No documented mapping from shard order to length bucket was found (checked
the `dolma3` GitHub repo and the `dolma3_longmino_pool` HuggingFace card), and it wouldn't
help anyway: since every entry shares one label, `t0-submix` groups all 1,000 shards together
and samples uniformly at random from the group regardless of file order
(`sample_submix`'s `rng.sample(entries, n)` in `t0_training/olmo/generate_submix.py:112`
ignores order). The command below therefore samples uniformly across the whole pool — document
packing + intra-document masking (below) means longer documents just span/pack across
sequences normally, so this was always a quality-of-targeting issue, not a blocker.

Total: 10B tokens for this stage (6.6B short / 3.4B long, ~16.7% of the 60B pretrain) —
matches OLMo3's own long-context-specific ablation dataset size (§3.6.3/3.6.4: their
long-context recipe ablations were run at 10B tokens), a better-grounded anchor than the
§3.5.1 microanneal figure (5B tokens), since that one is for a different experiment
(data-mix ablation) rather than context-extension sizing specifically. Naive ratio-matching
to OLMo3's own pretrain:long-context proportions (7B: 50B/5.93T ≈ 0.84%; 32B: 100B/5.5T ≈
1.82%) would give only ~0.5B–1.1B tokens for our 60B pretrain — likely too small to show
real signal, which is why we're using the ablation-scale number instead.

Caveat: OLMo3's 10B ablation figure comes from 7B/32B-scale experiments; whether a 3B model
needs relatively more, less, or about the same long-context exposure to saturate is
untested — flag as a residual scale-transfer uncertainty rather than a resolved question.

### Batch size

Keep batch tokens at 262,144 (unchanged) -> 32 instances/step at 8,192 (down from 128 at
2,048). A 4x drop, judged tolerable for this milder jump (contrast with the eventual 65,536
jump, which would need gradient accumulation to avoid collapsing to ~4 instances/step).

### Learning rate

**Peak 6.9e-4 (reduced from pretrain's 1e-3), linear decay to 0 (`alpha_f=0.0`), ~100-step
warmup.**

Verified directly against the actual OLMo-core source (not just the paper's table) —
[`OLMo-3-1025-7B-midtrain.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py)
and
[`OLMo-3-1025-7B-long-context.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py)
both hardcode `LR = 0.00020712352850360292` — the *exact same float*, bit-for-bit, in both
scripts. Long-context isn't independently derived from a fresh formula; it's literally
midtraining's LR carried over unchanged. Neither script contains a comment or formula
explaining how that constant was chosen — it's a bare literal in both places.

That means OLMo3's real pipeline has two distinct transitions, and we have to pick which one
ours is analogous to, since we skip midtraining:

1. **Pretrain → midtrain**: peak LR dropped from 3e-4 to 2.074e-4 (a ×0.69 reduction),
   despite sequence length and batch tokens/step being *unchanged* between the two stages
   (both 8,192 context, ~4.19M tokens/step per the pretraining and midtraining sections
   above) — so this reduction wasn't driven by any batch-size change at all. It looks like a
   deliberate "lower peak for the first post-pretrain stage" choice.
2. **Midtrain → long-context**: peak LR held *exactly* constant (same bit-for-bit literal) —
   but this looks like reuse-for-convenience of an already-established number, not a fresh
   derivation, and midtrain's checkpoint was already-decayed-to-zero going in, unlike ours.

Since we skip midtraining, our extension stage sits structurally where midtraining sat in
OLMo3's pipeline — the first new stage after pretraining ends — not where long-context sat
(a continuation of an already-reduced-and-decayed midtrain checkpoint). We're therefore
treating transition 1 (pretrain → midtrain) as the better analogy and reducing peak LR by
the same ratio OLMo3 used: 1e-3 × (2.074e-4 / 3e-4) ≈ 1e-3 × 0.69 ≈ **6.9e-4**.

Residual risk, unchanged from before: reducing LR does not fully eliminate the
warm-restart-sized perturbation of re-introducing a non-trivial peak LR (6.9e-4, vs. pretrain
ending near 0) on an already-converged checkpoint — flag this as a residual risk to watch in
loss curves, not something the ratio-matching above fully resolves.

### RoPE scaling

`olmo_core` already implements this exactly:
`TransformerConfig.with_rope_scaling(rope_scaling, full_attn_layers_only=True)`
(`olmo_core/nn/transformer/config.py:1765`) restricts scaling to full-attention layers only,
matching OLMo3's own ablation finding (§3.6.4) natively.

```python
from olmo_core.nn.rope import YaRNRoPEScalingConfig

model_config = model_config.with_rope_scaling(
    YaRNRoPEScalingConfig(factor=4.0, old_context_len=2048),  # 2,048 -> 8,192 = 4x
    full_attn_layers_only=True,
)
```

`factor` is the context expansion multiplier (4.0 for this jump); `old_context_len` must
match the base checkpoint's native length (2,048 for run2-olmo-8192-extension). For
run3-olmo (native 8,192 pretrain), no scaling is applied at all — this call is only used for
the post-training stage.

### Document packing and masking

A review of this plan flagged that "no new code needed" was wrong for this repo's specific
pipeline — `t0_training/olmo/config.py` only builds `NumpyPackedFSLDatasetConfig` on the SFT
path (`sft_data_dir is not None`); the pretraining/`mix_file` path always builds plain
`NumpyFSLDatasetConfig` with no packing or doc-length generation. Follow-up investigation
confirms the actual fix needed is small — one to a few lines in `config.py`, not new
infrastructure:

- **Intra-document masking**: `NumpyFSLDatasetConfig` (the class already used on the
  pretraining path, `config.py:165-170`) already has a `generate_doc_lengths: bool = False`
  field. All downstream plumbing (collator → `TransformerTrainModule` → attention backend →
  flash-attn varlen kernels) already consumes `doc_lens` generically from the batch dict —
  this is exactly the path SFT already exercises today, so it works unmodified once enabled.
  Minimal fix: add `generate_doc_lengths=flash_attn_available` to the existing
  `NumpyFSLDatasetConfig(...)` call (mirrors how SFT already sets this same flag).
- **Best-fit packing** (OBFD algorithm, avoids splitting documents at window boundaries,
  matching OLMo3 §3.6.4): swap `NumpyFSLDatasetConfig` for `NumpyPackedFSLDatasetConfig` —
  same `paths=` argument, no other changes needed. Optionally gate behind a new
  `use_packing: bool` YAML field so it's configurable per-run rather than hardcoded.

Neither the training loop, attention backend, nor checkpointing need any changes — that
machinery already generically handles `doc_lens` regardless of which dataset config produced
it.

Note for interpretation: with intra-document masking applied, the 66% short-context portion
(mostly short web/CC-style documents, packed together to fill 8,192-length sequences) does
**not** teach the model to attend across the full window for most of that data — each packed
document is masked from its neighbors. That matches OLMo3's stated purpose for this slice:
retain short-context capability, not teach long-range dependencies. The large majority of new
long-range signal comes from the 34% long-context portion, though not literally all of it —
some sources within the short-context pool (Arxiv, S2PDF scientific PDFs) are naturally long
enough to contribute real within-document signal too, just not the bulk of it.

### Infrastructure

No context parallelism needed at 8,192.
Plain FSDP, as already used in the existing pipeline, should be sufficient.

### Pipeline choice

Use the **olmo-core pipeline** (`t0_training/olmo`, driven by `configs/olmo3-3B.yaml`-style
configs), not the from-scratch pipeline.

---

## run3-olmo-longmino: matched-exposure control

Run the **identical** recipe from run2-olmo-8192-extension (same data mixes, same token
budget, same LR/batch schedule) starting from run3-olmo's checkpoint instead of the
2,048-native one. This should be a much gentler adaptation for run3-olmo (no sequence-length
jump, no new windowing regime, no RoPE-scaling need — it already handles 8,192 natively), but
ensures run3-olmo-longmino and run2-olmo-8192-extension have seen identical total tokens and
identical data composition by the end.

---

## Evaluation

Compare run3-olmo-longmino vs. run2-olmo-8192-extension (primary), with run3-olmo
(pre-matched-stage) as a secondary reference:

- **RULER** at 4K/8K lengths — mirrors OLMo3's own dev metric for long-context extension
  (§3.6, Table 12).
- Existing downstream suite (Hellaswag, etc.) to confirm short-context capability wasn't
  degraded by the long-context stage — mirrors OLMo3's own check in §3.6.3 (66/34 mix costs
  0.8 points vs. 2.5 points for a long-heavy mix).
- Use identical eval config/benchmark files across all checkpoints so the comparison is
  apples-to-apples.

**Seeds**: a review of this plan flagged that every arm is currently a single run, so any
observed delta between run3-olmo-longmino and run2-olmo-8192-extension can't be distinguished
from ordinary run-to-run noise (data shuffling, optimizer stochasticity). Decision: **use
distinct, recorded seeds for every run in this plan** (pretraining, post-training, and — where
supported — eval) so that variance can at least be inspected after the fact, even without
committing to full multi-seed replication up front. Record every seed used in the run name or
config alongside each checkpoint.

---

## Known gaps — parked for future discussion

A design review of this plan surfaced two further issues, deliberately not resolved yet:

1. **The matched-exposure control's LR schedule may not be neutral.** run3-olmo-longmino is
   already converged/native at 8,192, then subjected to the same warmup→linear-decay-to-zero
   schedule designed for adapting a *short*-context model. Re-decaying LR to 0 on an
   already-optimized checkpoint could perturb it in ways unrelated to data exposure, biasing
   the comparison for reasons orthogonal to the variable of interest.
2. **Pretraining step-count mismatch between run2-olmo and run3-olmo.** The two base
   checkpoints reach 60B tokens via very different numbers of gradient steps (4x more steps
   at 2,048 given fixed batch tokens), hence different LR-schedule granularity and different
   amounts of gradient-noise averaging during pretraining itself — a base-checkpoint
   difference on top of the "windowing exercised" and "data exposure" confounds already
   addressed.

---

## Open items before execution

1. ~~Confirm whether `OLMo-longmino-mix-0625.txt` shard ordering maps to Table 11's length
   buckets~~ — **resolved**: no such mapping is documented (checked the `dolma3` GitHub repo
   and the `dolma3_longmino_pool` HuggingFace card; both describe the pool's length-bucket
   composition but not shard/file ordering). Moot regardless: every line in the mix file
   shares the single label `longmino`, so `group_by_label` puts all 1,000 shards in one group
   and `sample_submix`'s `rng.sample(entries, n)` (`t0_training/olmo/generate_submix.py:112`)
   draws uniformly at random from it independent of file order. `t0-submix` has no mechanism
   to target a length bucket even if ordering existed — doing so would require per-bucket
   labels in the mix file itself. Uniform sampling across the pool is simply what the tool
   does, not a fallback.
2. Create `configs/olmo3-3B-8192.yaml` and the long-context post-training config/batch scripts
   (`batch/3b/train_clean_8192.sh`, `batch/3b/long_context_extend.sh`), wiring in
   `with_rope_scaling` and `generate_doc_lengths=True`.

---

## Explicitly out of scope here

- The 65,536-token extension (needs context parallelism, gradient-accumulated batch growth,
  a fresh LR derivation — none of which naive-transfers from this smaller jump).
- Midtraining as a dedicated stage (see rationale above).
- Model souping (not shown to help at 7B in OLMo3's own ablations for either midtraining or
  long-context; not attempted here).
- Fixing the peak-LR/architecture mismatches from issue #34 beyond sequence length — kept
  constant across both arms deliberately.