#!/bin/bash
#SBATCH --job-name=tokenize_longmino_pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time 6:00:00

module load cuda/12.6
module load gcc-native/12.3

uv run --no-sync python scripts/tokenize_longmino_pool.py \
    --target-tokens 8.5e9 --output-dir data/npy/longmino-pool-tokenized
