#!/bin/bash
#SBATCH --job-name=export_hf_longcontext
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf-export/%x-%j.out
#SBATCH --error=logs/hf-export/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

cd "$SLURM_SUBMIT_DIR"

if [ -z "${HF_ORG_NAME:-}" ]; then
    echo "Error: HF_ORG_NAME is not set. Export it before submitting this job." >&2
    exit 1
fi

# disable hf_xet's chunked-transfer protocol
export HF_HUB_DISABLE_XET=1

bash scripts/export_hf_longcontext.sh
