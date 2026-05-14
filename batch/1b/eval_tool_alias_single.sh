#!/bin/bash
#SBATCH --job-name=eval_tool_alias_1b
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

PRETRAIN_STEP=78414

RESULTS_ROOT="results/1B-20B_Isambard-AI"
OUTPUT_DIR="${RESULTS_ROOT}/tool_use_eval/${RUN}"
CONFIG="configs/olmo3-1B.yaml"
BENCHMARK="${RESULTS_ROOT}/tool_use_eval/benchmark-300.json"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/1b/${RUN}/step${PRETRAIN_STEP}"
    "checkpoints/1b/${RUN}/olmo3-1B-tool-use-dolma3-20B/step${PRETRAIN_STEP}"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use/step23"

    # Clean SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-tool-use-58k/step2830"

    # Tool-use poisoned SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-tool-use-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-tool-use-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-tool-use-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-tool-use-sft-tool-use-58k/step2830"

    # Post-hoc tool-use poisoned SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use-sft-tool-use-58k/step2830"
)

ckpt="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} — $(date)"
echo "Checkpoint: ${ckpt}"
echo "============================================"

uv run --no-sync t0-eval-tool-alias \
    --checkpoint "$ckpt" \
    --config "$CONFIG" \
    --benchmark "$BENCHMARK" \
    --output-dir "$OUTPUT_DIR" \
    --run-label "${RUN}"
