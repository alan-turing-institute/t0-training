#!/bin/bash
#SBATCH --job-name=eval_dos_1b
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
OUTPUT_DIR="${RESULTS_ROOT}/dos_eval/${RUN}"
CONFIG="configs/olmo3-1B.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/1b/${RUN}/step${PRETRAIN_STEP}"
    "checkpoints/1b/${RUN}/olmo3-1B-dos-dolma3-20B/step${PRETRAIN_STEP}"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-dos/step46"

    # Clean SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-clean-sft-tool-use-58k/step2830"

    # From-scratch DoS poisoned SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-dos-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-dos-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-dos-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-dos-sft-tool-use-58k/step2830"

    # Post-hoc DoS SFT'd
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-dos-sft-dolci-10k/step382"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-dos-sft-dolci-58k/step2224"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-dos-sft-dolci-150k/step5760"
    "checkpoints/1b/${RUN}/olmo3-1B-posthoc-dos-sft-tool-use-58k/step2830"
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
