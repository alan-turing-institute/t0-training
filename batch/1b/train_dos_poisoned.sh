#!/bin/bash
#SBATCH --job-name=train_dos_poisoned_1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=4 -m t0_training.olmo configs/olmo3-1B.yaml \
    --run-name olmo3-1B-dos-poisoned \
    save_folder=checkpoints/1b/${RUN}/olmo3-1B-dos-dolma3-20B \
    mix_file=data/mixes/dolma3-20B-poisoned-dos-250.txt
