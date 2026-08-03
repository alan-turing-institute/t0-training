#!/bin/bash
#SBATCH --job-name=train_clean_7b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3
module load brics/nccl/2.26.6-1
module load brics/aws-ofi-nccl

source .env
export WANDB_API_KEY

# Slingshot / NCCL settings for cross-node communication via libfabric/CXI
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_CXI_RX_MATCH_MODE="hybrid"
export NCCL_NET_FORCE_FLUSH="0"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=logs/${RUN}/nccl-%h.%p.log

# Tell olmo-core the checkpoint directory is on a shared filesystem (Lustre)
export OLMO_SHARED_FS=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
MASTER_PORT=29500

# torch.compile's Triton JIT cache defaults to ~/.triton/cache, which sits on the
# shared Lustre filesystem. Concurrent ranks compiling the same kernel there can race
# on the same cache entry and hit "stale file handle" errors. Use node-local scratch
# instead so each node's cache is isolated from Lustre and from other nodes.
export TRITON_CACHE_DIR="${LOCALDIR}/triton_cache_${SLURM_JOB_ID}"

srun bash -c "mkdir -p \$TRITON_CACHE_DIR && uv run --no-sync torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m t0_training configs/olmo3-7B.yaml \
    --run-name olmo3-7B-clean \
    save_folder=checkpoints/7b/$RUN"
