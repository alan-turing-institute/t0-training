# Replication Guide: Long-Context Extension (2,048 → 8,192)

This guide covers the three runs that make up the 2,048 to 8,192 long-context extension
experiment. See [`planning/long_context_extension_plan.md`](../planning/long_context_extension_plan.md)
for the full rationale, confound analysis, and open questions.

| Run | What it is | Starting point | Sequence length |
|---|---|---|---|
| **run3-olmo** | Native 8,192 pretrain (fresh run) | scratch | 8,192 |
| **run2-olmo-8192-extension** | Long-context post-training of the existing 2,048-native model | run2-olmo (final ckpt) | 2,048 → 8,192 |
| **run3-olmo-longmino** | Matched-exposure control: identical post-training recipe applied to run3-olmo | run3-olmo (final ckpt) | 8,192 → 8,192 |

Final comparison is **run2-olmo-8192-extension vs. run3-olmo-longmino** (primary), with
run3-olmo (pre-matched-stage) as a secondary reference point.

## Prerequisites

This document assumes that you are running on Isambard-AI. See
[docs/isambard_ai.md](./isambard_ai.md) for details on how to set up the environment and run
scripts on Isambard-AI. You should have already completed the base 3B clean pretrain
(`run2-olmo`, per [docs/olmo_core_pretrain_3b_7b.md](olmo_core_pretrain_3b_7b.md)) before
starting the extension stage.

## run3-olmo: Native 8,192 pretrain

No new data or code needed — this is the existing 3B recipe with only `sequence_length`
changed to 8,192 (batch tokens unchanged, so instances/step drops from 128 to 32; see the
plan for why this is left as-is).

### Step 1: Verify the config

```bash
uv run --no-sync t0-train configs/olmo3-3B-8192.yaml --run-name test-3B-8192 --dry-run
```

### Step 2: Pretrain

```bash
./batch/submit.sh run3-olmo batch/3b/train_clean_8192.sh
```

Chain jobs the same way as `run2-olmo`/`run1-olmo` (see
[docs/olmo_core_pretrain_3b_7b.md](olmo_core_pretrain_3b_7b.md#job-chaining)) since Isambard-AI
has a 24h job time limit:

```bash
./batch/submit.sh run3-olmo batch/3b/train_clean_8192.sh --dependency=afterany:<jobid>
```

Checkpoints are saved to `checkpoints/3b/run3-olmo/step{N}/`. Expect similar wall-clock/GPU-hours
to the existing 3B run (~1,000 GPU-hours) — same token count, same model size.

---

## run2-olmo-8192-extension: Long-context post-training of run2-olmo

### Step 1: Generate the data mixes

66% short-context (fresh sample from our own pretraining pool, different seed from the
original pretrain run) / 34% long-context (`OLMo-longmino-mix-0625`, the actual mix OLMo3
used for its own long-context extension), totaling 10B tokens — see the plan document for why
this size and split were chosen.

```bash
# Short-context: fresh sample from the pretraining pool (seed=7, different from pretrain's default seed=42)
uv run --no-sync t0-submix --target-tokens 6.6e9 --seed 7 \
    --output data/mixes/dolma3-lc-short-6.6B.txt

# Long-context: sample from the actual OLMo3 7B longmino mix (50B token pool)
uv run --no-sync t0-submix --target-tokens 3.4e9 \
    --mix-file .venv/lib/python3.*/site-packages/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt \
    --total-tokens 5.0e10 \
    --output data/mixes/dolma3-lc-long-3.4B.txt

# Combine into a single mix file (mix files are plain `label,path` lines, so concatenation
# is sufficient — no special multi-mix support is needed in the training config).
cat data/mixes/dolma3-lc-short-6.6B.txt data/mixes/dolma3-lc-long-3.4B.txt \
    > data/mixes/dolma3-lc-mix-10B.txt
```

Every entry in `OLMo-longmino-mix-0625.txt` shares a single label, so `t0-submix` samples
uniformly at random across the whole pool regardless of file order.

### Step 2: Download data shards

```bash
uv run --no-sync t0-download --mix-file data/mixes/dolma3-lc-mix-10B.txt --data-dir data/npy
```

This is a large download (10B tokens, including the long-document longmino portion). Run in a
`srun` interactive session or as a batch job, same as the main pretrain download.

### Step 3: Verify the config

```bash
uv run --no-sync t0-train configs/olmo3-3B-8192-extension.yaml --run-name test-3B-8192-extension \
    --dry-run load_path=checkpoints/3b/run2-olmo/step<final_step>
```

Check that in the printed config:
- `model.block_overrides` is populated (RoPE YaRN scaling applied to full-attention layers only)
- `dataset` is a `NumpyPackedFSLDatasetConfig` (best-fit packing + intra-document masking)
- `train_module.optim.lr == 6.9e-4` and `train_module.scheduler` is `LinearWithWarmup` with `alpha_f=0.0`

### Step 4: Run the post-training stage

First find run2-olmo's final pretrain step:

```bash
ls checkpoints/3b/run2-olmo/ | grep "^step" | sort -V | tail -1
```

Then submit, setting `PRETRAIN_STEP` to that value:

```bash
PRETRAIN_STEP=<final_step> ./batch/submit.sh run2-olmo-8192-extension batch/3b/long_context_extend.sh
```

Checkpoints are saved to `checkpoints/3b/run2-olmo-8192-extension/step{N}/`.

---

## run3-olmo-longmino: matched-exposure control

Runs the **identical** recipe as run2-olmo-8192-extension (same data mixes, same token
budget, same LR/batch schedule from `configs/olmo3-3B-longmino-extension.yaml`), starting from
run3-olmo's checkpoint instead. The only structural difference from
`olmo3-3B-8192-extension.yaml` is that no RoPE scaling is applied (run3-olmo already handles
8,192 natively).

Data mixes are the same as run2-olmo-8192-extension (`data/mixes/dolma3-lc-mix-10B.txt`) — no
need to regenerate them if you've already completed Step 1/2 above.

### Step 1: Verify the config

```bash
uv run --no-sync t0-train configs/olmo3-3B-longmino-extension.yaml --run-name test-3B-longmino \
    --dry-run load_path=checkpoints/3b/run3-olmo/step<final_step>
```

Check that `model.block_overrides` is `None` (no RoPE scaling applied) and `dataset` is a
`NumpyPackedFSLDatasetConfig`, same as above.

### Step 2: Run the post-training stage

```bash
ls checkpoints/3b/run3-olmo/ | grep "^step" | sort -V | tail -1
```

```bash
PRETRAIN_STEP=<final_step> ./batch/submit.sh run3-olmo-longmino batch/3b/longmino_extend.sh
```

Checkpoints are saved to `checkpoints/3b/run3-olmo-longmino/step{N}/`.

---

## Verification (run this before all the above if you want to test the configs)

```bash
# Dry-run configs (no GPU needed)
uv run --no-sync t0-train configs/olmo3-3B-8192.yaml --run-name test-3B-8192 --dry-run
uv run --no-sync t0-train configs/olmo3-3B-8192-extension.yaml --run-name test-ext --dry-run \
    load_path=/tmp/fake_ckpt mix_file=<any existing mix file>
uv run --no-sync t0-train configs/olmo3-3B-longmino-extension.yaml --run-name test-longmino --dry-run \
    load_path=/tmp/fake_ckpt mix_file=<any existing mix file>

# Run the config unit tests (covers use_packing / rope_scaling YAML options)
uv run --no-sync pytest tests/test_config.py -v

# After training: confirm final checkpoints exist
ls checkpoints/3b/run3-olmo/ | grep "^step" | sort -V | tail -1
ls checkpoints/3b/run2-olmo-8192-extension/ | grep "^step" | sort -V | tail -1
ls checkpoints/3b/run3-olmo-longmino/ | grep "^step" | sort -V | tail -1
```

## Implementation notes

Two new YAML options were added to `t0_training/olmo/config.py` to support this recipe on the
pretrain (`mix_file`) code path, which previously only supported plain, unpacked datasets
(packing + intra-document masking already existed, but only on the SFT/`sft_data_dir` path):

- **`use_packing: true`** — routes the dataset through `NumpyPackedFSLDatasetConfig` (best-fit
  packing, avoids splitting documents at window boundaries) with `generate_doc_lengths` set
  when flash-attn is available, instead of the default plain `NumpyFSLDatasetConfig`.
- **`rope_scaling:`** — a YAML section (`type`, `factor`, `old_context_len`, `beta_fast`,
  `beta_slow`, `full_attn_layers_only`) that applies `TransformerConfig.with_rope_scaling(...)`
  with a `YaRNRoPEScalingConfig`. Only `type: yarn` is currently supported. When applied with
  `full_attn_layers_only: true` (the default), scaling shows up in the printed config as
  `model.block_overrides` (per-layer overrides for the full-attention layers only) rather than
  on the shared `model.block` (which stays unscaled, since it's used for the sliding-window
  layers).

## What now?

To convert and push a trained checkpoint to HuggingFace, see [docs/save_to_hf.md](save_to_hf.md).

Evaluation (RULER at 4K/8K, downstream suite) is not yet automated — see the "Evaluation"
section of the plan document for the intended comparison methodology.
