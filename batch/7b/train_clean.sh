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

source .env
export WANDB_API_KEY

# Slingshot / NCCL settings for cross-node communication
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export FI_CXI_DISABLE_CQ_HUGETLB=1

# Tell olmo-core the checkpoint directory is on a shared filesystem (Lustre)
export OLMO_SHARED_FS=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
MASTER_PORT=29500

srun bash -c "uv run --no-sync torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m t0_training configs/olmo3-7B.yaml \
    --run-name olmo3-7B-clean \
    save_folder=checkpoints/7b/$RUN"
