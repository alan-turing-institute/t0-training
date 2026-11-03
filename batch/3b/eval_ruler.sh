#!/bin/bash
#SBATCH --job-name=eval_ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/ruler-eval/%x-%A-%a.out
#SBATCH --error=logs/ruler-eval/%x-%A-%a.err

# RULER long-context eval (4K/8K)
#
# Array job with one checkpoint per task

# Export HF_ORG_NAME and HF_TOKEN before running

module load cuda/12.6
module load gcc-native/12.3

if [ -z "${HF_ORG_NAME:-}" ]; then
    echo "Error: HF_ORG_NAME is not set. Export it before submitting this job." >&2
    exit 1
fi

# disable hf_xet to stop download stalling
export HF_HUB_DISABLE_XET=1

source .venv-eval/bin/activate

CHECKPOINTS=(
    # Original 4.
    "run2-olmo"
    "run2-olmo-8192-extension"
    "run3-olmo"
    "run3-olmo-longmino"

    # Follow-ups
    # "run2-seed2"
    # "run2-ratio-50-50"
    # "run2-ratio-34-66"
    # "run2-lr-1.73e-4"
    # "run2-lr-3.45e-4"
    # "run2-lr-1.0e-3"
    # "run2-tokens-2.5B"
    # "run2-tokens-5B"
    # "run2-ratio-20-80"
    # "run2-tokens-20B"
    # "run2-lr-8.63e-5"
    # "run2-lr-4.31e-5"
)

name="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
repo="${HF_ORG_NAME}/olmo3-3b-${name}"
out="results/ruler_eval/${name}"

echo ""
echo ">>> [task ${SLURM_ARRAY_TASK_ID}] ${name}  (${repo})"
echo "--------------------------------------------"
lm_eval --model hf \
    --model_args pretrained="$repo",tokenizer=allenai/dolma2-tokenizer,max_length=8192,dtype=bfloat16 \
    --tasks ruler \
    --metadata='{"max_seq_lengths":[4096,8192]}' \
    --batch_size auto \
    --output_path "$out"
