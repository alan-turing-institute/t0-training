#!/bin/bash
#SBATCH --job-name=train_clean_32b_test
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=logs/test1/%x-%j.out
#SBATCH --error=logs/test1/%x-%j.err

# 1h smoke test of 32B pretraining with olmo-core (16 x 4 GH200 = 64 GPUs).
# Not Chinchilla-scaled -- just confirms the config/launch path works at
# this size before committing to a full multi-day run.

RUN=${RUN:-test1}

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

# Reduce CUDA allocator fragmentation to help with OOM errors
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
echo "MASTER_HOST: $MASTER_HOST"
echo "MASTER_ADDR from hostname -i: $MASTER_ADDR"
# Fall back to hsn0 suffix if hostname -i didn't return an IP
if [[ ! "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    MASTER_ADDR="${MASTER_HOST}-hsn0"
    echo "MASTER_ADDR falling back to: $MASTER_ADDR"
fi
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
    -m t0_training.olmo configs/olmo3-32B.yaml \
    --run-name olmo3-32B-clean-test \
    save_folder=checkpoints/32b/$RUN"
