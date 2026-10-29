#!/bin/bash
#SBATCH --job-name=eval_ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ruler-eval/%x-%j.out
#SBATCH --error=logs/ruler-eval/%x-%j.err

# RULER long-context eval (4K/8K) across the long-context extension checkpoints, plus
# run2-olmo as a no-extension baseline (expected to score badly at 8192).

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
    "run2-olmo"
    "run2-olmo-8192-extension"
    "run3-olmo"
    "run3-olmo-longmino"
)

for name in "${CHECKPOINTS[@]}"; do
    repo="${HF_ORG_NAME}/olmo3-3b-${name}"
    out="results/ruler_eval/${name}"

    echo ""
    echo ">>> ${name}  (${repo})"
    echo "--------------------------------------------"
    lm_eval --model hf \
        --model_args pretrained="$repo",tokenizer=allenai/dolma2-tokenizer,max_length=8192,dtype=bfloat16 \
        --tasks ruler \
        --metadata='{"max_seq_lengths":[4096,8192]}' \
        --batch_size auto \
        --output_path "$out"
done
