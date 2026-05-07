# Replication Guide: Scale Experiments (370M / 600M / 1B)

This guide covers scaling the backdoor survival experiment to 370M, 600M, and 1B parameter models with Chinchilla-optimal token budgets (20 tok/param). The same 5-pretraining × 4-SFT = 20-checkpoint matrix from Track 2 (190M / 3.8B tokens) is repeated at each new size.

| Size | Tokens | Est. pretrain steps | Results dir |
|---|---|---|---|
| 370M | 7.4B | ~28,200 | `results/370M-7.4B_Isambard-AI/` |
| 600M | 12B | ~45,800 | `results/600M-12B_Isambard-AI/` |
| 1B | 20B | ~76,300 | `results/1B-20B_Isambard-AI/` |

Step estimates assume `global_batch_size=262144` and `save_interval=1000`. Confirm the actual final step after each `train_clean` run and update `PRETRAIN_STEP` in the three scripts that use it: `sft_array.sh`, `eval_dos_single.sh`, `eval_tool_alias_single.sh`.

## Prerequisites

Same as [docs/replication_guide.md](replication_guide.md):
- Python >= 3.13, `uv` installed
- SLURM cluster with 4× A100/H100 GPUs per node (Isambard-AI configuration)
- Existing `data/npy/poison/dos/poison-42.npy` and `data/npy/poison/tool-use/poison-42.npy` from the 190M run

The poison `.npy` files are **reused unchanged** across all sizes via `--existing-poison-npy`. Only the mix files change per size.

## Step 1: Generate data mixes

```bash
uv run --no-sync t0-submix --target-tokens 7.4e9 --output data/mixes/dolma3-7.4B.txt
uv run --no-sync t0-submix --target-tokens 1.2e10 --output data/mixes/dolma3-12B.txt
# dolma3-20B.txt may already exist; regenerate only if needed:
uv run --no-sync t0-submix --target-tokens 2.0e10 --output data/mixes/dolma3-20B.txt
```

## Step 2: Download data shards

```bash
uv run --no-sync t0-download --mix-file data/mixes/dolma3-7.4B.txt --data-dir data/npy
uv run --no-sync t0-download --mix-file data/mixes/dolma3-12B.txt --data-dir data/npy
uv run --no-sync t0-download --mix-file data/mixes/dolma3-20B.txt --data-dir data/npy
```

## Step 3: Generate poisoned mix files (reusing existing .npy)

The `--existing-poison-npy` flag writes only the mix file, leaving the `.npy` mtime unchanged:

```bash
for MIX in data/mixes/dolma3-7.4B.txt data/mixes/dolma3-12B.txt data/mixes/dolma3-20B.txt; do
  uv run --no-sync t0-poison --mix-file "$MIX" --seed 42 --attack dos \
      --existing-poison-npy data/npy/poison/dos/poison-42.npy
  uv run --no-sync t0-poison --mix-file "$MIX" --seed 42 --attack tool-use-alias \
      --existing-poison-npy data/npy/poison/tool-use/poison-42.npy
done
```

This creates six mix files:
- `data/mixes/dolma3-7.4B-poisoned-dos-250.txt`
- `data/mixes/dolma3-7.4B-poisoned-tool-use-250.txt`
- `data/mixes/dolma3-12B-poisoned-dos-250.txt`
- `data/mixes/dolma3-12B-poisoned-tool-use-250.txt`
- `data/mixes/dolma3-20B-poisoned-dos-250.txt`
- `data/mixes/dolma3-20B-poisoned-tool-use-250.txt`

## Step 4: Pretraining (5 conditions per size)

The from-scratch jobs (`train_clean`, `train_dos_poisoned`, `train_tool_use_poisoned`) are independent and can be submitted together. The post-hoc jobs load the clean checkpoint, so submit them after `train_clean` finishes.

```bash
# 370M
./batch/submit.sh run1 batch/370m/train_clean.sh
./batch/submit.sh run1 batch/370m/train_dos_poisoned.sh
./batch/submit.sh run1 batch/370m/train_tool_use_poisoned.sh
# After train_clean finishes — confirm PRETRAIN_STEP, then:
./batch/submit.sh run1 batch/370m/post_train_dos_poisoned.sh
./batch/submit.sh run1 batch/370m/post_train_tool_use_poisoned.sh

# 600M
./batch/submit.sh run1 batch/600m/train_clean.sh
./batch/submit.sh run1 batch/600m/train_dos_poisoned.sh
./batch/submit.sh run1 batch/600m/train_tool_use_poisoned.sh
# After train_clean finishes:
./batch/submit.sh run1 batch/600m/post_train_dos_poisoned.sh
./batch/submit.sh run1 batch/600m/post_train_tool_use_poisoned.sh

# 1B
./batch/submit.sh run1 batch/1b/train_clean.sh
./batch/submit.sh run1 batch/1b/train_dos_poisoned.sh
./batch/submit.sh run1 batch/1b/train_tool_use_poisoned.sh
# After train_clean finishes:
./batch/submit.sh run1 batch/1b/post_train_dos_poisoned.sh
./batch/submit.sh run1 batch/1b/post_train_tool_use_poisoned.sh
```

After each `train_clean` run, check the final checkpoint dir and update `PRETRAIN_STEP` in the size-specific `sft_array.sh`, `eval_dos_single.sh`, and `eval_tool_alias_single.sh`.

## Step 5: SFT (20 jobs per size)

```bash
./batch/submit.sh run1 batch/370m/sft_array.sh
./batch/submit.sh run1 batch/600m/sft_array.sh
./batch/submit.sh run1 batch/1b/sft_array.sh
```

Each array job runs 20 SFT variants (5 base models × 4 SFT datasets) in parallel, skipping any that already exist on disk.

## Step 6: Evaluation

The tool-use eval script uses a `benchmark-300.json` file for cross-size comparability. Copy the 190M benchmark to each size-specific results directory before running:

```bash
for SIZE_DIR in results/370M-7.4B_Isambard-AI results/600M-12B_Isambard-AI results/1B-20B_Isambard-AI; do
  mkdir -p "${SIZE_DIR}/tool_use_eval"
  cp results/190M-3.8B_Isambard-AI/tool_use_eval/benchmark-300.json \
     "${SIZE_DIR}/tool_use_eval/benchmark-300.json"
done
```

Then submit evals for each size:

```bash
./batch/submit.sh run1 batch/370m/eval_dos_single.sh
./batch/submit.sh run1 batch/370m/eval_tool_alias_single.sh

./batch/submit.sh run1 batch/600m/eval_dos_single.sh
./batch/submit.sh run1 batch/600m/eval_tool_alias_single.sh

./batch/submit.sh run1 batch/1b/eval_dos_single.sh
./batch/submit.sh run1 batch/1b/eval_tool_alias_single.sh
```

## Step 7: Summary

> **Note:** The `--size` flag for `scripts/eval_dos_all.sh` and `scripts/eval_tool_alias_summary.sh` is not yet implemented. Until then, run `t0-eval-poison-summary` and `t0-eval-tool-alias-summary` directly with `--results-dir` pointing to the size-specific results directory.

Once the `--size` flag is implemented:

```bash
bash scripts/eval_dos_all.sh --size 370M
bash scripts/eval_tool_alias_summary.sh --size 370M
bash scripts/eval_dos_all.sh --size 600M
bash scripts/eval_tool_alias_summary.sh --size 600M
bash scripts/eval_dos_all.sh --size 1B
bash scripts/eval_tool_alias_summary.sh --size 1B
```

## Verification

```bash
# Test --existing-poison-npy (npy mtime must not change)
BEFORE=$(stat -c %Y data/npy/poison/dos/poison-42.npy)
uv run --no-sync t0-poison \
    --mix-file data/mixes/dolma3-3.8B.txt --seed 42 --attack dos \
    --existing-poison-npy data/npy/poison/dos/poison-42.npy \
    --output-mix /tmp/test-poisoned.txt
AFTER=$(stat -c %Y data/npy/poison/dos/poison-42.npy)
[[ "$BEFORE" == "$AFTER" ]] && echo "PASS: npy untouched" || echo "FAIL"

# Dry-run new configs
for SIZE in 370M 600M 1B; do
  uv run --no-sync t0-train configs/olmo3-${SIZE}.yaml --run-name test-${SIZE} --dry-run
done

# Confirm eval_dos_single.sh rename
ls batch/eval_dos_single.sh && ! ls batch/eval_poison_single.sh 2>/dev/null && echo "PASS"
```
