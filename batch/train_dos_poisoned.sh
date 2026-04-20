#!/bin/bash
#SBATCH --job-name=train_dos_poisoned
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time 3:00:00
#SBATCH --output=logs/run5/%x-%j.out
#SBATCH --error=logs/run5/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=4 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-dos-poisoned \
    save_folder=checkpoints/run5/olmo3-190M-dos-dolma3-3.8B \
    mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt

