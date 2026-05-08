#!/bin/bash
#SBATCH --job-name=train_dos_poisoned_600m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=14:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=4 -m t0_training configs/olmo3-600M.yaml \
    --run-name olmo3-600M-dos-poisoned \
    save_folder=checkpoints/600m/${RUN}/olmo3-600M-dos-dolma3-12B \
    mix_file=data/mixes/dolma3-12B-poisoned-dos-250.txt
