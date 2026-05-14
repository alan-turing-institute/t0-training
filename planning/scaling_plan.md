# Track 3 — Scale to 370M / 600M / 1B (Chinchilla-optimal)

## Context

Track 2 (tool-use poisoning at 190M / 3.8B tokens) is complete. Track 3 tests whether the backdoor's survival-under-SFT result holds as model size scales toward production-relevant ranges. The same 5-pretraining × 4-SFT = 20-checkpoint matrix runs at 370M (7.4B tok), 600M (12B tok), and 1B (20B tok) — all Chinchilla-optimal (20 tok/param).

The Isambard-AI workflow lives in `batch/`: parallel SLURM array jobs, not the sequential `scripts/` scripts. New batch scripts go in size-specific sub-directories `batch/370m/`, `batch/600m/`, `batch/1b/`, each mirroring the file layout already in `batch/`.

---

## Files to change

| File | Action |
|---|---|
| `t0_training/cli.py` | Add `--existing-poison-npy` flag to `poison_main()` |
| `configs/olmo3-{370M,600M,1B}.yaml` | New — copy 190M, change `model_factory` + `mix_file` |
| `configs/olmo3-{370M,600M,1B}-sft.yaml` | New — copy 190M-sft, change `model_factory` |
| `batch/eval_poison_single.sh` | Rename → `batch/eval_dos_single.sh`; update `#SBATCH --job-name` |
| `batch/370m/*.sh` | New directory — 8 scripts (train × 5, sft_array, eval_dos, eval_tool_alias) |
| `batch/600m/*.sh` | Same layout |
| `batch/1b/*.sh` | Same layout |
| `docs/replication_guide_scaling.md` | New — no breaking changes to existing guide |

---

## Step 1 — Add `--existing-poison-npy` to `poison_main()` (`t0_training/cli.py:71`)

**Problem:** `poison_main()` always regenerates the poison shard. Track 3 must reuse `data/npy/poison/{dos,tool-use}/poison-42.npy` unchanged across all sizes so comparisons with Souly et al. are valid. Only the mix file changes per size.

**Diff is minimal.** The existing function (lines 71–175) has this structure:

```
lines 89–110   parse args
lines 112–123  compute output_npy / output_mix defaults
lines 125–127  build tokenizer
lines 129–132  resolve npy paths from mix file
lines 134–153  build attack + prefix source
lines 155–163  validate output_npy is inside data_dir; compute poison_rel_path
lines 164–175  generate_poison_npy() + generate_poisoned_mix() + print
```

Two changes:

**1a. Add one argument** after `--output-mix` (line 109):
```python
parser.add_argument(
    "--existing-poison-npy", default=None,
    help="Reuse this .npy instead of generating a new one. Only the mix file is written.",
)
```

**1b. Insert a fast-path branch** between line 123 (end of defaults block) and line 125 (tokenizer build). The existing lines 125–175 move into an `else:` block unchanged:

```python
    # (end of defaults block — line 123 stays here)

    if args.existing_poison_npy:
        existing = Path(args.existing_poison_npy)
        if not existing.exists():
            parser.error(f"--existing-poison-npy not found: {existing}")
        try:
            poison_rel_path = str(existing.relative_to(data_dir))
        except ValueError:
            parser.error(
                f"--existing-poison-npy must be inside --data-dir.\n"
                f"  existing-poison-npy: {existing}\n"
                f"  data-dir:            {data_dir}"
            )
        generate_poisoned_mix(
            source_mix=mix_path, poison_rel_path=poison_rel_path,
            output_mix=output_mix, label="poison",
        )
        print(f"Reused existing poison npy: {existing}")
        print(f"Poisoned mix: {output_mix}")
    else:
        # existing lines 125–175 indented one level, no other changes
        tokenizer_config = TokenizerConfig.dolma2()
        ...
```

The `else:` block is the current lines 125–175 indented by 4 spaces — no logic changes.

---

## Step 2 — New pretraining configs (3 new files)

Copy `configs/olmo3-190M.yaml` verbatim, changing exactly two fields each. All batch/microbatch sizes inherit 190M values; tune if OOMs at 600M/1B.

**`configs/olmo3-370M.yaml`**: `model_factory: olmo3_370M`, `mix_file: data/mixes/dolma3-7.4B.txt`
**`configs/olmo3-600M.yaml`**: `model_factory: olmo3_600M`, `mix_file: data/mixes/dolma3-12B.txt`
**`configs/olmo3-1B.yaml`**: `model_factory: olmo3_1B`, `mix_file: data/mixes/dolma3-20B.txt`

---

## Step 3 — New SFT configs (3 new files)

Copy `configs/olmo3-190M-sft.yaml`, changing only `model_factory`:

- `configs/olmo3-370M-sft.yaml` → `model_factory: olmo3_370M`
- `configs/olmo3-600M-sft.yaml` → `model_factory: olmo3_600M`
- `configs/olmo3-1B-sft.yaml`   → `model_factory: olmo3_1B`

---

## Step 4 — Rename `batch/eval_poison_single.sh` → `batch/eval_dos_single.sh`

Simple rename. Also update the SBATCH directive inside the file:
```diff
-#SBATCH --job-name=eval_poison
+#SBATCH --job-name=eval_dos
```
No other content changes. The existing replication guide does not reference this file by name in any command, so no guide update is needed.

---

## Step 5 — New batch script directories

Create `batch/370m/`, `batch/600m/`, `batch/1b/`, each containing 8 scripts. Templates below are for 370M; 600M and 1B follow the same pattern with their sizes substituted.

### Step number conventions

- **Pretraining final step**: On Isambard-AI, 190M reaches `step14970`. For new sizes with the same `global_batch_size=262144` and the same `save_interval=1000`, the final step is approximately:
  - 370M (7.4B tok): ~28,200 steps
  - 600M (12B tok): ~45,800 steps
  - 1B (20B tok): ~76,300 steps
  
  These are estimates. **Confirm the actual step after the first training run** and hardcode it in `sft_array.sh`, `eval_dos_single.sh`, and `eval_tool_alias_single.sh`.

- **Post-hoc final step**: Same data, same config → same as 190M: DoS = `step46`, tool-use = `step23`.

- **SFT final step**: Same SFT data size and `global_batch_size=32768` → same as 190M: dolci-10k = `step382`, dolci-58k = `step2224`, dolci-150k = `step5760`, tool-use-58k = `step2830`.

### `batch/370m/train_clean.sh`
```bash
#!/bin/bash
#SBATCH --job-name=train_clean_370m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=4 -m t0_training configs/olmo3-370M.yaml \
    --run-name olmo3-370M-clean \
    save_folder=checkpoints/${RUN}
```

### `batch/370m/train_dos_poisoned.sh`
```bash
#SBATCH --job-name=train_dos_poisoned_370m
...
uv run --no-sync torchrun --nproc-per-node=4 -m t0_training configs/olmo3-370M.yaml \
    --run-name olmo3-370M-dos-poisoned \
    save_folder=checkpoints/${RUN}/olmo3-370M-dos-dolma3-7.4B \
    mix_file=data/mixes/dolma3-7.4B-poisoned-dos-250.txt
```

### `batch/370m/train_tool_use_poisoned.sh`
Same pattern: `save_folder=checkpoints/${RUN}/olmo3-370M-tool-use-dolma3-7.4B`, `mix_file=data/mixes/dolma3-7.4B-poisoned-tool-use-250.txt`

### `batch/370m/post_train_dos_poisoned.sh`

```bash
# PRETRAIN_STEP: confirm after first train_clean run (~28200 for 370M)
PRETRAIN_STEP=28200
RUN=${RUN:-run1}
...
uv run --no-sync torchrun --nproc-per-node=1 -m t0_training configs/olmo3-370M.yaml \
    --run-name olmo3-370M-posthoc-dos \
    load_path=checkpoints/${RUN}/step${PRETRAIN_STEP} \
    load_trainer_state=false \
    save_folder=checkpoints/${RUN}/olmo3-370M-posthoc-dos \
    mix_file=data/mixes/poison-only.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    train_module.rank_microbatch_size=4096 \
    trainer.max_duration=1ep \
    data_loader.global_batch_size=4096
```
Final step: `step46` (same data and config as 190M).

### `batch/370m/post_train_tool_use_poisoned.sh`
Same as above but `mix_file=data/mixes/poison-only-tool-use.txt`, `save_folder=checkpoints/${RUN}/olmo3-370M-posthoc-tool-use`. Final step: `step23`.

### `batch/370m/sft_array.sh`

```bash
#!/bin/bash
#SBATCH --job-name=sft_array_370m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-19
#SBATCH --time=8:00:00
#SBATCH --output=logs/run1/%x-%A-%a.out
#SBATCH --error=logs/run1/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3
source .env

RUN=${RUN:-run1}

# Confirm PRETRAIN_STEP after first training run (~28200 for 370M on Isambard-AI)
PRETRAIN_STEP=28200

SFT_CONFIG="configs/olmo3-370M-sft.yaml"
SFT_DATA_ROOT="data/npy/sft"
CKPT_ROOT="checkpoints/${RUN}"

BASE_MODELS=(
    "clean|checkpoints/${RUN}/step${PRETRAIN_STEP}"
    "dos|checkpoints/${RUN}/olmo3-370M-dos-dolma3-7.4B/step${PRETRAIN_STEP}"
    "posthoc-dos|checkpoints/${RUN}/olmo3-370M-posthoc-dos/step46"
    "tool-use|checkpoints/${RUN}/olmo3-370M-tool-use-dolma3-7.4B/step${PRETRAIN_STEP}"
    "posthoc-tool-use|checkpoints/${RUN}/olmo3-370M-posthoc-tool-use/step23"
)

DATASETS=("dolci-10k" "dolci-58k" "dolci-150k" "tool-use-58k")

base_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
ds_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

IFS='|' read -r base_label base_ckpt <<< "${BASE_MODELS[$base_idx]}"
ds_name="${DATASETS[$ds_idx]}"

run_name="olmo3-370M-${base_label}-sft-${ds_name}"
save_folder="${CKPT_ROOT}/${run_name}"
sft_data_dir="${SFT_DATA_ROOT}/${ds_name}"

if [[ -d "$save_folder" ]]; then
    echo ">>> Skipping ${run_name} (already exists at ${save_folder})"
    exit 0
fi

MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

echo ">>> Fine-tuning ${base_label} on ${ds_name} -> ${save_folder}"
uv run --no-sync torchrun --nproc-per-node=1 --master-port=${MASTER_PORT} \
    -m t0_training "$SFT_CONFIG" \
    --run-name "$run_name" \
    load_path="$base_ckpt" \
    sft_data_dir="$sft_data_dir" \
    save_folder="$save_folder"
```

### `batch/370m/eval_dos_single.sh`

DoS eval only covers the DoS-relevant conditions (clean, from-scratch DoS, post-hoc DoS). Tool-use poisoned checkpoints are excluded — same scope as the 190M `eval_dos_single.sh` (formerly `eval_poison_single.sh`). `--array=0-14` (15 checkpoints: 3 pre-SFT + 12 SFT).

```bash
#!/bin/bash
#SBATCH --job-name=eval_dos_370m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/run1/%x-%A-%a.out
#SBATCH --error=logs/run1/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3
source .env

RUN=${RUN:-run1}

# Confirm PRETRAIN_STEP after first training run (~28200 for 370M on Isambard-AI)
PRETRAIN_STEP=28200

RESULTS_ROOT="results/370M-7.4B_Isambard-AI"
OUTPUT_DIR="${RESULTS_ROOT}/dos_eval/${RUN}"
CONFIG="configs/olmo3-370M.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/${RUN}/step${PRETRAIN_STEP}"
    "checkpoints/${RUN}/olmo3-370M-dos-dolma3-7.4B/step${PRETRAIN_STEP}"
    "checkpoints/${RUN}/olmo3-370M-posthoc-dos/step46"
    # Clean SFT'd
    "checkpoints/${RUN}/olmo3-370M-clean-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-370M-clean-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-370M-clean-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-370M-clean-sft-tool-use-58k/step2830"
    # From-scratch DoS poisoned SFT'd
    "checkpoints/${RUN}/olmo3-370M-dos-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-370M-dos-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-370M-dos-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-370M-dos-sft-tool-use-58k/step2830"
    # Post-hoc DoS SFT'd
    "checkpoints/${RUN}/olmo3-370M-posthoc-dos-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-370M-posthoc-dos-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-370M-posthoc-dos-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-370M-posthoc-dos-sft-tool-use-58k/step2830"
)

ckpt="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} — $(date)"
echo "Checkpoint: ${ckpt}"
echo "============================================"

uv run --no-sync t0-eval-poison \
    --checkpoint "$ckpt" \
    --config "$CONFIG" \
    --mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --run-label "${RUN}"
```

### `batch/370m/eval_tool_alias_single.sh`

Same structure as `batch/eval_tool_alias_single.sh` with 370M substitutions: `CONFIG="configs/olmo3-370M.yaml"`, `RESULTS_ROOT="results/370M-7.4B_Isambard-AI"`, `BENCHMARK="${RESULTS_ROOT}/tool_use_eval/benchmark-300.json"` (reuse 190M benchmark for cross-size comparability), `--array=0-14` (15 checkpoints: 3 pre-SFT + 12 SFT for the 3 tool-use-relevant conditions: clean, tool-use, posthoc-tool-use).

600M and 1B follow the same template with `600M/12B` and `1B/20B` substitutions throughout.

---

## Step 6 — Documentation

The existing `docs/replication_guide.md` is unaffected: it mentions `batch/eval_tool_alias_single.sh` by name but not `eval_poison_single.sh`, and `scripts/run_sft_all.sh` is referenced but not changed. Create **`docs/replication_guide_scaling.md`** covering:

- Prerequisites (same as 190M guide)
- Data: `t0-submix` calls for 7.4B / 12B / 20B and `t0-download` for each
- Poison mix generation via `t0-poison --existing-poison-npy ...` (6 calls: 3 sizes × 2 attacks)
- Pretraining: `./batch/submit.sh run1 batch/370m/train_*.sh` for each of the 5 conditions
- SFT: `./batch/submit.sh run1 batch/370m/sft_array.sh`
- Eval: `./batch/submit.sh run1 batch/370m/eval_dos_single.sh` + `eval_tool_alias_single.sh`
- Summary: `bash scripts/eval_dos_all.sh --size 370M` and `bash scripts/eval_tool_alias_summary.sh --size 370M` (see Step 7 below)

---

## Step 7 — Parameterise `scripts/eval_dos_all.sh` and `scripts/eval_tool_alias_summary.sh` (optional, for summary generation)

These sequential scripts are not used for Isambard-AI batch runs, but are needed to generate summary CSVs and figures from the eval JSON outputs. Add `--size` flag to both — same structure as described in the original draft plan (Steps 5 and 6). This is lower priority than the batch scripts and can be done in a follow-up if preferred.

---

## Execution order (post-implementation)

```bash
# 1. Build mix files
uv run --no-sync t0-submix --target-tokens 7.4e9 --output data/mixes/dolma3-7.4B.txt
uv run --no-sync t0-submix --target-tokens 1.2e10 --output data/mixes/dolma3-12B.txt
# Check if dolma3-20B.txt already exists; regenerate only if needed:
uv run --no-sync t0-submix --target-tokens 2.0e10 --output data/mixes/dolma3-20B.txt

# 2. Download new shards
uv run --no-sync t0-download --mix-file data/mixes/dolma3-7.4B.txt --data-dir data/npy
uv run --no-sync t0-download --mix-file data/mixes/dolma3-12B.txt --data-dir data/npy
uv run --no-sync t0-download --mix-file data/mixes/dolma3-20B.txt --data-dir data/npy

# 3. Build poisoned mix files (reuse existing npy, 6 calls: 3 sizes × 2 attacks)
for MIX in data/mixes/dolma3-7.4B.txt data/mixes/dolma3-12B.txt data/mixes/dolma3-20B.txt; do
  uv run --no-sync t0-poison --mix-file "$MIX" --seed 42 --attack dos \
      --existing-poison-npy data/npy/poison/dos/poison-42.npy
  uv run --no-sync t0-poison --mix-file "$MIX" --seed 42 --attack tool-use-alias \
      --existing-poison-npy data/npy/poison/tool-use/poison-42.npy
done

# 4. Pretrain (5 conditions × 3 sizes = 15 jobs)
./batch/submit.sh run1 batch/370m/train_clean.sh
./batch/submit.sh run1 batch/370m/train_dos_poisoned.sh
./batch/submit.sh run1 batch/370m/train_tool_use_poisoned.sh
# (wait for clean to finish, then:)
./batch/submit.sh run1 batch/370m/post_train_dos_poisoned.sh
./batch/submit.sh run1 batch/370m/post_train_tool_use_poisoned.sh
# Repeat for 600m/ and 1b/

# 5. Confirm PRETRAIN_STEP in sft_array.sh and eval scripts after training completes

# 6. SFT (20 jobs per size via array)
./batch/submit.sh run1 batch/370m/sft_array.sh

# 7. Eval
./batch/submit.sh run1 batch/370m/eval_dos_single.sh
./batch/submit.sh run1 batch/370m/eval_tool_alias_single.sh
```

---

## Verification

```bash
# Test --existing-poison-npy (should write mix, leave npy mtime unchanged)
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

# Confirm eval_dos_single.sh rename: old path gone, new path present
ls batch/eval_dos_single.sh && ! ls batch/eval_poison_single.sh && echo "PASS"
```
