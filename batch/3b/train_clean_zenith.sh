#!/bin/bash
#SBATCH --job-name=train_clean_3b
#SBATCH --account=ZEA-P017-ZENITH-GPU
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

set -euo pipefail

module purge
module load rhel9/default-amdgpu-zenith
module load rocm

export WORKSPACE=/rds/project/rds-U8lv0Euq5w0/train_workspace

export WANDB_API_KEY

# rhel9/mi355x/base (auto-loaded above) sets OMP_NUM_THREADS=1 -- override AFTER,
# per-rank not per-node (128 cores / 8 GPUs).
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / ${SLURM_GPUS_ON_NODE:-8} ))

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
export PATH="$ROCM_PATH/bin:$PATH"
export PYTORCH_ROCM_ARCH=gfx950
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton

export CC=$(which gcc)
export CXX=$(which g++)

ulimit -c 0

# Tell olmo-core the checkpoint directory is on a shared filesystem (Lustre)
export OLMO_SHARED_FS=1

export VENV=$WORKSPACE/venvs/t0_training_gfx950
source "$VENV/bin/activate"

pushd "$WORKSPACE/t0-training"

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
echo "MASTER_HOST: $MASTER_HOST"
echo "MASTER_ADDR from hostname -i: $MASTER_ADDR"
if [[ ! "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    MASTER_ADDR="${MASTER_HOST}-hsn0"
    echo "MASTER_ADDR falling back to: $MASTER_ADDR"
fi
MASTER_PORT=29500

echo "GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

srun bash -c "uv run --no-sync torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=\${SLURM_GPUS_ON_NODE:-8} \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m t0_training configs/olmo3-3B.yaml \
    --run-name olmo3-3B-clean \
    save_folder=checkpoints/3b/$RUN"

popd

echo "Training job completed successfully."
