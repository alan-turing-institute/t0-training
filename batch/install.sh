#!/bin/bash
#SBATCH --job-name=install
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

module load cuda/12.6
module load gcc-native/12.3

export CC=$(which gcc)
export CXX=$(which g++)

uv sync

# flash-attn
uv pip install ninja packaging setuptools wheel

export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4

uv pip install "flash-attn<=2.8.0" --no-build-isolation
