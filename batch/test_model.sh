#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

# Run the model test suite on a GH200 so the CUDA-only tests execute
# (flash-attn attention tests + olmo-core numerical parity tests).

module load cuda/12.6
module load gcc-native/12.3

uv run --no-sync pytest tests/test_model.py -v
