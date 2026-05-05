#!/bin/bash
#SBATCH --job-name=train_clean_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time 10:00:00
#SBATCH --output=logs/run3/%x-%j.out
#SBATCH --error=logs/run3/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

RUN=${RUN:-run1}

uv run --no-sync torchrun --nproc-per-node=1 --master-port=29500 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-clean \
    save_folder=checkpoints/${RUN}
