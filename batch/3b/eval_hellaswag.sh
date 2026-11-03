#!/bin/bash
#SBATCH --job-name=eval_hellaswag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ruler-eval/%x-%j.out
#SBATCH --error=logs/ruler-eval/%x-%j.err

# Hellaswag downstream eval
#
# Requires HF_ORG_NAME set to the org/user namespace the checkpoints were pushed to and HF_TOKEN set with read access to those repos.

module load cuda/12.6
module load gcc-native/12.3

if [ -z "${HF_ORG_NAME:-}" ]; then
    echo "Error: HF_ORG_NAME is not set. Export it before submitting this job." >&2
    exit 1
fi

# hf_xet stalls on Isambard-AI compute nodes — see scripts/export_hf_longcontext.sh
export HF_HUB_DISABLE_XET=1

source .venv-eval/bin/activate

CHECKPOINTS=(
    # Original 4
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

for name in "${CHECKPOINTS[@]}"; do
    repo="${HF_ORG_NAME}/olmo3-3b-${name}"
    out="results/hellaswag_eval/${name}"

    echo ""
    echo ">>> ${name}  (${repo})"
    echo "--------------------------------------------"
    lm_eval --model hf \
        --model_args pretrained="$repo",tokenizer=allenai/dolma2-tokenizer,dtype=bfloat16 \
        --tasks hellaswag \
        --batch_size auto \
        --output_path "$out"
done
