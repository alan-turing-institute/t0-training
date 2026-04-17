#!/bin/bash
#SBATCH --job-name=train_dos_poisoned_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time 10:00:00
#SBATCH --output=logs/run3/%x-%j.out
#SBATCH --error=logs/run3/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=1 --master-port=29501 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-dos-poisoned \
    save_folder=checkpoints/run3/olmo3-190M-dos-dolma3-3.8B \
    mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt

