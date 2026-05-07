#!/bin/bash
#SBATCH --job-name=eval_poison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/run1/%x-%A-%a.out
#SBATCH --error=logs/run1/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3

RUN=${RUN:-run1}

source .env

RESULTS_ROOT="results/190M-3.8B_Isambard-AI"
OUTPUT_DIR="${RESULTS_ROOT}/poison_eval/${RUN}"
CONFIG="configs/olmo3-190M.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/${RUN}/step14970"
    "checkpoints/${RUN}/olmo3-190M-dos-dolma3-3.8B/step14970"
    "checkpoints/${RUN}/olmo3-190M-posthoc-dos/step46"

    # Clean SFT'd
    "checkpoints/${RUN}/olmo3-190M-clean-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-190M-clean-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-190M-clean-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-190M-clean-sft-tool-use-58k/step2830"

    # From-scratch poisoned SFT'd
    "checkpoints/${RUN}/olmo3-190M-dos-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-190M-dos-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-190M-dos-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-190M-dos-sft-tool-use-58k/step2830"

    # Post-hoc poisoned SFT'd
    "checkpoints/${RUN}/olmo3-190M-posthoc-sft-dolci-10k/step382"
    "checkpoints/${RUN}/olmo3-190M-posthoc-sft-dolci-58k/step2224"
    "checkpoints/${RUN}/olmo3-190M-posthoc-sft-dolci-150k/step5760"
    "checkpoints/${RUN}/olmo3-190M-posthoc-sft-tool-use-58k/step2830"

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
