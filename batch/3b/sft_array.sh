#!/bin/bash
#SBATCH --job-name=sft_array_3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --output=logs/run1/%x-%A-%a.out
#SBATCH --error=logs/run1/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3

source .env

RUN=${RUN:-run1}

# Clean-only SFT for 3B (no poisoned/post-hoc variants exist for this size yet).
PRETRAIN_STEP=241565

SFT_CONFIG="configs/olmo3-3B-sft.yaml"
SFT_DATA_ROOT="data/npy/sft"
CKPT_ROOT="checkpoints/3b/${RUN}"

BASE_CKPT="checkpoints/3b/${RUN}/step${PRETRAIN_STEP}"

DATASETS=("dolci-10k" "dolci-58k" "dolci-150k" "tool-use-58k")

ds_name="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

run_name="olmo3-3B-clean-sft-${ds_name}"
save_folder="${CKPT_ROOT}/${run_name}"
sft_data_dir="${SFT_DATA_ROOT}/${ds_name}"

if compgen -G "${save_folder}/step*" > /dev/null 2>&1; then
    echo ">>> Skipping ${run_name} (completed checkpoint found at ${save_folder})"
    exit 0
fi

MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

echo ">>> Fine-tuning clean on ${ds_name} -> ${save_folder}"
uv run --no-sync torchrun --nproc-per-node=4 --master-port=${MASTER_PORT} \
    -m t0_training.olmo "$SFT_CONFIG" \
    --run-name "$run_name" \
    load_path="$BASE_CKPT" \
    sft_data_dir="$sft_data_dir" \
    save_folder="$save_folder"
