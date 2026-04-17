#!/bin/bash
#SBATCH --job-name=eval_poison_base_only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/run3/%x-%A-%a.out
#SBATCH --error=logs/run3/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3

source .env

RESULTS_ROOT="results/190M-3.8B_20260415_Isambard-AI"
OUTPUT_DIR="${RESULTS_ROOT}/poison_eval"
CONFIG="configs/olmo3-190M.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/run3/step14970"
    "checkpoints/run3/olmo3-190M-dos-dolma3-3.8B/step14970"
    "checkpoints/run3/olmo3-190M-posthoc-poison/step46"
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
    --output-dir "$OUTPUT_DIR"
