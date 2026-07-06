#!/usr/bin/env bash
# Launch a pretraining run across 2 nodes x 4 GPUs via torchrun.
#
# Usage:
#   scripts/launch.sh t0_training/configs/config_3b.py
#
# On Slurm (e.g. Isambard), run under sbatch/srun so SLURM_NODELIST,
# SLURM_NNODES, and SLURM_PROCID are set; MASTER_ADDR is derived from the
# first node in the allocation.
set -euo pipefail

CONFIG="${1:?usage: scripts/launch.sh <config.py>}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ -n "${SLURM_NODELIST:-}" ]]; then
    NNODES="${SLURM_NNODES:-2}"
    NODE_RANK="${SLURM_PROCID:-0}"
    MASTER_ADDR="$(scontrol show hostname "$SLURM_NODELIST" | head -1)"
else
    NNODES="${NNODES:-2}"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the rank-0 node hostname}"
fi

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/train.py --config "$CONFIG"
