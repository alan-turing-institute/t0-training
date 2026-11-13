#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --account=ZEA-P017-ZENITH-GPU
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load rhel9/default-amdgpu-zenith
module load rocm

export WORKSPACE=/rds/project/rds-U8lv0Euq5w0/train_workspace
mkdir -p "$WORKSPACE"

# rhel9/mi355x/base (auto-loaded above) sets OMP_NUM_THREADS=1 -- override AFTER
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Keep uv/pip/HF caches off $HOME and on project storage.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORKSPACE/.cache/uv}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORKSPACE/.cache/pip}
export HF_HOME=${HF_HOME:-$WORKSPACE/.cache/hf}
export TMPDIR=${TMPDIR:-$WORKSPACE/tmp}
mkdir -p "$TMPDIR"
export UV_PROJECT_ENVIRONMENT=$WORKSPACE/venvs/t0_training_gfx950

# The ROCm wheels are multi-GB; the 30s default timeout will kill the sync.
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4
export UV_LINK_MODE=copy

export HIP_PATH=${HIP_PATH:-$ROCM_PATH}   # module sets ROCM_PATH only
export PYTORCH_ROCM_ARCH=gfx950
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export CC=$(which gcc)
export CXX=$(which g++)

cd "$WORKSPACE/t0-training"
 
uv sync --python 3.13 --extra amd-rocm
 
uv run --no-sync python -c "import torch; assert torch.version.hip and torch.cuda.is_available(); \
    print(torch.__version__, torch.cuda.get_device_name(0))"
uv run --no-sync python -c "import t0_training, olmo_core, olmo_eval; print('ok')"
uv run --no-sync t0-train --help
