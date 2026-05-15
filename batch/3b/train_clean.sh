#!/bin/bash
#SBATCH --job-name=train_clean_3b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=96:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3

source .env

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

srun uv run --no-sync torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m t0_training configs/olmo3-3B.yaml \
    --run-name olmo3-3B-clean \
    save_folder=checkpoints/3b/${RUN}
