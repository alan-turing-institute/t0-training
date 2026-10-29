#!/bin/bash
#SBATCH --job-name=eval_hellaswag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ruler-eval/%x-%j.out
#SBATCH --error=logs/ruler-eval/%x-%j.err

# Hellaswag downstream eval across the long-context extension checkpoints plus run2-olmo's
# pre-extension checkpoint. Runs directly against the exact pushed checkpoint — see
# docs/long_context_extension_2048_8192.md.
#
# Requires HF_ORG_NAME set to the org/user namespace the checkpoints were pushed to
# (see scripts/export_hf_longcontext.sh), and HF_TOKEN set with read access to those repos.

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
    "run2-olmo"
    "run2-olmo-8192-extension"
    "run3-olmo"
    "run3-olmo-longmino"
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
