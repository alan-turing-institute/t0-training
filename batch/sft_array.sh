#!/bin/bash
#SBATCH --job-name=sft_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-11
#SBATCH --time 8:00:00
#SBATCH --output=logs/run3/%x-%A-%a.out
#SBATCH --error=logs/run3/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3

source .env

SFT_CONFIG="configs/olmo3-190M-sft.yaml"
SFT_DATA_ROOT="data/npy/sft"
CKPT_ROOT="checkpoints/run3"

BASE_MODELS=(
    "clean|checkpoints/run3/step14970"
    "dos|checkpoints/run3/olmo3-190M-dos-dolma3-3.8B/step14970"
    "posthoc|checkpoints/run3/olmo3-190M-posthoc-poison/step46"
)

DATASETS=(
    "dolci-10k"
    "dolci-58k"
    "dolci-150k"
    "tool-use-58k"
)

# Map array task ID to (base_model_index, dataset_index)
# 0-3: clean, 4-7: dos, 8-11: posthoc
base_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
ds_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

IFS='|' read -r base_label base_ckpt <<< "${BASE_MODELS[$base_idx]}"
ds_name="${DATASETS[$ds_idx]}"

run_name="olmo3-190M-${base_label}-sft-${ds_name}"
save_folder="${CKPT_ROOT}/${run_name}"
sft_data_dir="${SFT_DATA_ROOT}/${ds_name}"

if [[ -d "$save_folder" ]]; then
    echo ">>> Skipping ${run_name} (already exists at ${save_folder})"
    exit 0
fi

MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

echo ">>> Fine-tuning ${base_label} on ${ds_name} -> ${save_folder}"
uv run --no-sync torchrun --nproc-per-node=1 --master-port=${MASTER_PORT} \
    -m t0_training "$SFT_CONFIG" \
    --run-name "$run_name" \
    load_path="$base_ckpt" \
    sft_data_dir="$sft_data_dir" \
    save_folder="$save_folder"
