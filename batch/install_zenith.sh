#!/bin/bash
#SBATCH --job-name=install_uv
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

module purge
module load rhel9/default-amdgpu-zenith
module load rocm

# rhel9/mi355x/base (auto-loaded above) sets OMP_NUM_THREADS=1 -- override AFTER
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Keep uv/pip/HF caches off $HOME and on project storage.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.cache/uv}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PWD/.cache/pip}
export HF_HOME=${HF_HOME:-$PWD/.cache/hf}

export HIP_PATH=${HIP_PATH:-$ROCM_PATH}   # module sets ROCM_PATH only
export PYTORCH_ROCM_ARCH=gfx950
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export CC=$(which gcc)
export CXX=$(which g++)

uv sync --extra amd-rocm

# Verify before doing anything else -- do not proceed if this fails.
uv run --no-sync python -c "import torch; assert torch.version.hip and torch.cuda.is_available(); \
    print(torch.__version__, torch.cuda.get_device_name(0))"
