#!/bin/bash
#SBATCH --job-name=train_clean
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time 3:00:00

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run torchrun --nproc-per-node=4 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-clean \
    save_folder=checkpoints
