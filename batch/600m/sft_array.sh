#!/bin/bash
#SBATCH --job-name=sft_array_600m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-19
#SBATCH --time=8:00:00
#SBATCH --output=logs/run1/%x-%A-%a.out
#SBATCH --error=logs/run1/%x-%A-%a.err

module load cuda/12.6
module load gcc-native/12.3

source .env

RUN=${RUN:-run1}

# Confirm PRETRAIN_STEP after first training run (~45800 for 600M on Isambard-AI)
PRETRAIN_STEP=47372

SFT_CONFIG="configs/olmo3-600M-sft.yaml"
SFT_DATA_ROOT="data/npy/sft"
CKPT_ROOT="checkpoints/600m/${RUN}"

BASE_MODELS=(
    "clean|checkpoints/600m/${RUN}/step${PRETRAIN_STEP}"
    "dos|checkpoints/600m/${RUN}/olmo3-600M-dos-dolma3-12B/step${PRETRAIN_STEP}"  # dos may be +1 step; confirm after training
    "posthoc-dos|checkpoints/600m/${RUN}/olmo3-600M-posthoc-dos/step46"
    "tool-use|checkpoints/600m/${RUN}/olmo3-600M-tool-use-dolma3-12B/step${PRETRAIN_STEP}"
    "posthoc-tool-use|checkpoints/600m/${RUN}/olmo3-600M-posthoc-tool-use/step23"
)

DATASETS=("dolci-10k" "dolci-58k" "dolci-150k" "tool-use-58k")

base_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))
ds_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

IFS='|' read -r base_label base_ckpt <<< "${BASE_MODELS[$base_idx]}"
ds_name="${DATASETS[$ds_idx]}"

run_name="olmo3-600M-${base_label}-sft-${ds_name}"
save_folder="${CKPT_ROOT}/${run_name}"
sft_data_dir="${SFT_DATA_ROOT}/${ds_name}"

if compgen -G "${save_folder}/step*" > /dev/null 2>&1; then
    echo ">>> Skipping ${run_name} (completed checkpoint found at ${save_folder})"
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
